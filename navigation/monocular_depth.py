from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from read_pfm import read_pfm
from stereo_disparity import (
    DEFAULT_DATA_ROOT,
    StereoDepthConfig,
    build_montage,
    collision_decision,
    colorize,
    colorize_risk,
    convert_disparity_to_risk,
    ensure_dirs,
    find_frame_pairs,
    select_evenly,
    valid_estimate_mask,
    valid_gt_mask,
    write_csv,
    write_image,
    write_json,
)


DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "week02_monocular_depth"
DEFAULT_WEEK1_RESULTS = SCRIPT_DIR / "results" / "week01_stereo_depth" / "final"
DEFAULT_TORCH_HOME = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
DEFAULT_MIDAS_REPO = DEFAULT_TORCH_HOME / "hub" / "intel-isl_MiDaS_master"


@dataclass(frozen=True)
class MonocularDepthConfig:
    model_type: str = "MiDaS_small"
    device: str = "auto"
    midas_repo: Path = DEFAULT_MIDAS_REPO
    torch_home: Path = DEFAULT_TORCH_HOME


class LocalMiDaS:
    def __init__(self, config: MonocularDepthConfig) -> None:
        if not config.midas_repo.exists():
            raise FileNotFoundError(f"MiDaS repository not found: {config.midas_repo}")
        os.environ["TORCH_HOME"] = str(config.torch_home)
        self.device = self.resolve_device(config.device)
        self.model_type = config.model_type
        self.model = torch.hub.load(str(config.midas_repo), config.model_type, source="local", trust_repo=True)
        self.model.to(self.device)
        self.model.eval()
        transforms = torch.hub.load(str(config.midas_repo), "transforms", source="local", trust_repo=True)
        self.transform = transforms.small_transform if config.model_type == "MiDaS_small" else transforms.dpt_transform

    @staticmethod
    def resolve_device(name: str) -> torch.device:
        if name != "auto":
            return torch.device(name)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def predict_raw(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(image_rgb).to(self.device)
        start = time.perf_counter()
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        latency_ms = (time.perf_counter() - start) * 1000.0
        return prediction.detach().cpu().numpy().astype(np.float32), latency_ms


def finite_or_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def calibrate_scale(raw_predictions: list[np.ndarray], gt_disparities: list[np.ndarray]) -> float:
    ratios: list[float] = []
    for raw, gt in zip(raw_predictions, gt_disparities):
        mask = valid_gt_mask(gt) & np.isfinite(raw) & (raw > 0)
        if np.any(mask):
            ratios.append(float(np.median(gt[mask]) / max(float(np.median(raw[mask])), 1e-6)))
    if not ratios:
        raise RuntimeError("MiDaS calibration failed because no valid ground-truth overlap was found")
    return float(np.median(np.array(ratios, dtype=np.float64)))


def compare_to_stereo(frame_id: str, risk: np.ndarray, obstacle: np.ndarray, week1_dir: Path) -> dict[str, object]:
    stereo_risk_path = week1_dir / "arrays" / f"frame_{frame_id}_stereo_risk.npy"
    stereo_obstacle_path = week1_dir / "arrays" / f"frame_{frame_id}_obstacle_mask.npy"
    if not stereo_risk_path.exists() or not stereo_obstacle_path.exists():
        return {
            "risk_shape_match": False,
            "obstacle_shape_match": False,
            "risk_mae_vs_stereo": float("nan"),
            "risk_rmse_vs_stereo": float("nan"),
            "risk_agreement_0p5": float("nan"),
            "obstacle_iou_vs_stereo": float("nan"),
        }
    stereo_risk = np.load(stereo_risk_path)
    stereo_obstacle = np.load(stereo_obstacle_path)
    risk_shape_match = stereo_risk.shape == risk.shape
    obstacle_shape_match = stereo_obstacle.shape == obstacle.shape
    if not risk_shape_match or not obstacle_shape_match:
        return {
            "risk_shape_match": risk_shape_match,
            "obstacle_shape_match": obstacle_shape_match,
            "risk_mae_vs_stereo": float("nan"),
            "risk_rmse_vs_stereo": float("nan"),
            "risk_agreement_0p5": float("nan"),
            "obstacle_iou_vs_stereo": float("nan"),
        }
    diff = risk.astype(np.float32) - stereo_risk.astype(np.float32)
    mono_binary = risk >= 0.5
    stereo_binary = stereo_risk >= 0.5
    obstacle_union = np.logical_or(obstacle > 0, stereo_obstacle > 0)
    obstacle_intersection = np.logical_and(obstacle > 0, stereo_obstacle > 0)
    return {
        "risk_shape_match": True,
        "obstacle_shape_match": True,
        "risk_mae_vs_stereo": float(np.mean(np.abs(diff))),
        "risk_rmse_vs_stereo": float(np.sqrt(np.mean(diff**2))),
        "risk_agreement_0p5": float(np.mean(mono_binary == stereo_binary)),
        "obstacle_iou_vs_stereo": float(np.sum(obstacle_intersection) / max(float(np.sum(obstacle_union)), 1.0)),
    }


def metric_row(
    frame_id: str,
    baseline: str,
    model_type: str,
    estimate: np.ndarray,
    gt: np.ndarray,
    depth: np.ndarray,
    risk: np.ndarray,
    obstacle: np.ndarray,
    decision: str,
    clearances: dict[str, float],
    latency_ms: float,
    scale_factor: float,
    depth_config: StereoDepthConfig,
    alignment: dict[str, object],
) -> dict[str, object]:
    mask = valid_gt_mask(gt) & valid_estimate_mask(estimate, depth_config)
    if np.any(mask):
        error = np.abs(estimate[mask] - gt[mask])
        mae = float(np.mean(error))
        rmse = float(np.sqrt(np.mean(error**2)))
        bad_3px = float(np.mean(error > 3.0))
        bad_5pct = float(np.mean((error / np.maximum(gt[mask], 1e-6)) > 0.05))
    else:
        mae = rmse = bad_3px = bad_5pct = float("nan")
    depth_values = depth[np.isfinite(depth) & (depth > 0)]
    robust_min_depth = float(np.percentile(depth_values, 1)) if depth_values.size else float("nan")
    mean_depth = float(np.mean(depth_values)) if depth_values.size else float("nan")
    return {
        "frame_id": frame_id,
        "baseline": baseline,
        "method": f"Intel {model_type}",
        "mae_disparity_px": mae,
        "rmse_disparity_px": rmse,
        "bad_pixel_3px_rate": bad_3px,
        "bad_pixel_5pct_rate": bad_5pct,
        "valid_estimate_fraction": float(np.mean(valid_estimate_mask(estimate, depth_config))),
        "robust_min_depth_m": robust_min_depth,
        "mean_depth_m": mean_depth,
        "obstacle_fraction": float(np.mean(obstacle > 0)),
        "max_risk_fraction": float(np.mean(risk >= 0.95)),
        "decision": decision,
        "left_clearance_m": finite_or_zero(clearances["left"]),
        "center_clearance_m": finite_or_zero(clearances["center"]),
        "right_clearance_m": finite_or_zero(clearances["right"]),
        "center_alert_fraction": finite_or_zero(clearances["center_alert_fraction"]),
        "inference_latency_ms": latency_ms,
        "collision_threshold_m": depth_config.collision_threshold_m,
        "scale_factor": scale_factor,
        "risk_min": float(np.min(risk)),
        "risk_max": float(np.max(risk)),
        "risk_shape_match": alignment["risk_shape_match"],
        "obstacle_shape_match": alignment["obstacle_shape_match"],
        "risk_mae_vs_stereo": alignment["risk_mae_vs_stereo"],
        "risk_rmse_vs_stereo": alignment["risk_rmse_vs_stereo"],
        "risk_agreement_0p5": alignment["risk_agreement_0p5"],
        "obstacle_iou_vs_stereo": alignment["obstacle_iou_vs_stereo"],
    }


def summarize(rows: list[dict[str, object]], model_type: str) -> dict[str, object]:
    numeric_fields = [
        "mae_disparity_px",
        "rmse_disparity_px",
        "bad_pixel_3px_rate",
        "bad_pixel_5pct_rate",
        "valid_estimate_fraction",
        "robust_min_depth_m",
        "mean_depth_m",
        "obstacle_fraction",
        "max_risk_fraction",
        "center_alert_fraction",
        "inference_latency_ms",
        "risk_mae_vs_stereo",
        "risk_rmse_vs_stereo",
        "risk_agreement_0p5",
        "obstacle_iou_vs_stereo",
    ]
    summary: dict[str, object] = {
        "method": f"Intel {model_type}",
        "frames": len(rows),
        "forward_count": sum(row["decision"] == "FORWARD" for row in rows),
        "turn_left_count": sum(row["decision"] == "TURN_LEFT" for row in rows),
        "turn_right_count": sum(row["decision"] == "TURN_RIGHT" for row in rows),
        "slow_or_hold_count": sum(row["decision"] == "SLOW_OR_HOLD" for row in rows),
        "risk_shape_match_count": sum(row["risk_shape_match"] is True for row in rows),
        "obstacle_shape_match_count": sum(row["obstacle_shape_match"] is True for row in rows),
    }
    for field in numeric_fields:
        values = np.array([float(row[field]) for row in rows], dtype=np.float64)
        summary[f"avg_{field}"] = float(np.nanmean(values))
    return summary


def render_video(montage_dir: Path, output_path: Path, fps: float) -> str:
    frames = sorted(montage_dir.glob("frame_*_week2_montage.png"))
    if not frames:
        raise RuntimeError(f"No montage frames found in {montage_dir}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-pattern_type",
                "glob",
                "-i",
                str(montage_dir / "frame_*_week2_montage.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
        )
        return "ffmpeg/libx264/yuv420p"
    sample = cv2.imread(str(frames[0]))
    if sample is None:
        raise RuntimeError(f"Could not load montage frame: {frames[0]}")
    height, width = sample.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Could not load montage frame: {frame_path}")
        writer.write(frame)
    writer.release()
    return "opencv/mp4v"


def generate_monocular_output(
    data_root: Path | str = DEFAULT_DATA_ROOT,
    output_dir: Path | str = DEFAULT_RESULTS_ROOT,
    week1_dir: Path | str = DEFAULT_WEEK1_RESULTS,
    baseline: str = "ZM-Baseline-25",
    max_frames: int = 15,
    fps: float = 1.0,
    run_name: str | None = None,
    depth_config: StereoDepthConfig | None = None,
    mono_config: MonocularDepthConfig | None = None,
) -> Path:
    depth_config = depth_config or StereoDepthConfig()
    mono_config = mono_config or MonocularDepthConfig()
    data_root = Path(data_root)
    output_root = Path(output_dir)
    week1_dir = Path(week1_dir)
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_root / run_name
    dirs = ensure_dirs(output_path)
    frames = select_evenly(find_frame_pairs(data_root, baseline), max_frames)
    model = LocalMiDaS(mono_config)
    raw_predictions: list[np.ndarray] = []
    gt_disparities: list[np.ndarray] = []
    loaded_images: list[np.ndarray] = []
    latencies: list[float] = []
    selected_ids: list[str] = []
    for item in frames:
        frame_id = str(item["id"])
        selected_ids.append(frame_id)
        left = cv2.imread(str(item["left"]))
        if left is None:
            raise FileNotFoundError(f"Could not load monocular frame {item['left']}")
        gt, _ = read_pfm(item["gt"])
        raw, latency_ms = model.predict_raw(left)
        loaded_images.append(left)
        gt_disparities.append(gt)
        raw_predictions.append(raw)
        latencies.append(latency_ms)
    scale_factor = calibrate_scale(raw_predictions, gt_disparities)
    rows: list[dict[str, object]] = []
    for item, left, raw, gt, latency_ms in zip(frames, loaded_images, raw_predictions, gt_disparities, latencies):
        frame_id = str(item["id"])
        mono_disparity = raw * scale_factor
        depth, risk, obstacle = convert_disparity_to_risk(mono_disparity, depth_config)
        decision, clearances = collision_decision(depth, risk, depth_config)
        stereo_risk = np.load(week1_dir / "arrays" / f"frame_{frame_id}_stereo_risk.npy")
        gt_mask = valid_gt_mask(gt)
        alignment = compare_to_stereo(frame_id, risk, obstacle, week1_dir)
        np.save(dirs["arrays"] / f"frame_{frame_id}_mono_disparity.npy", mono_disparity)
        np.save(dirs["arrays"] / f"frame_{frame_id}_mono_depth.npy", depth)
        np.save(dirs["arrays"] / f"frame_{frame_id}_mono_risk.npy", risk)
        np.save(dirs["arrays"] / f"frame_{frame_id}_obstacle_mask.npy", obstacle)
        mono_disparity_vis = colorize(mono_disparity, cv2.COLORMAP_VIRIDIS, gt_mask)
        gt_disparity_vis = colorize(gt, cv2.COLORMAP_VIRIDIS, gt_mask)
        depth_vis = colorize(depth, cv2.COLORMAP_TURBO)
        risk_vis = colorize_risk(risk)
        stereo_risk_vis = colorize_risk(stereo_risk)
        obstacle_vis = cv2.cvtColor(obstacle * 255, cv2.COLOR_GRAY2BGR)
        annotated_risk = risk_vis.copy()
        cv2.rectangle(annotated_risk, (0, annotated_risk.shape[0] - 52), (annotated_risk.shape[1], annotated_risk.shape[0]), (0, 0, 0), -1)
        cv2.putText(
            annotated_risk,
            f"{decision} | center {clearances['center']:.2f} m",
            (12, annotated_risk.shape[0] - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_risk,
            f"threshold {depth_config.collision_threshold_m:.1f} m",
            (12, annotated_risk.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        write_image(dirs["frames"] / f"frame_{frame_id}_left.png", left)
        write_image(dirs["disparity"] / f"frame_{frame_id}_mono_disparity.png", mono_disparity_vis)
        write_image(dirs["disparity"] / f"frame_{frame_id}_gt_disparity.png", gt_disparity_vis)
        write_image(dirs["depth"] / f"frame_{frame_id}_depth_proxy.png", depth_vis)
        write_image(dirs["risk"] / f"frame_{frame_id}_risk_map.png", risk_vis)
        write_image(dirs["obstacles"] / f"frame_{frame_id}_obstacle_mask.png", obstacle_vis)
        montage = build_montage(
            [
                ("Left RGB input", left),
                ("MiDaS disparity", mono_disparity_vis),
                ("Stereo risk baseline", stereo_risk_vis),
                ("Mono depth proxy", depth_vis),
                ("Mono risk + decision", annotated_risk),
                ("Mono obstacle mask", obstacle_vis),
            ]
        )
        write_image(dirs["montage"] / f"frame_{frame_id}_week2_montage.png", montage)
        rows.append(
            metric_row(
                frame_id,
                baseline,
                mono_config.model_type,
                mono_disparity,
                gt,
                depth,
                risk,
                obstacle,
                decision,
                clearances,
                latency_ms,
                scale_factor,
                depth_config,
                alignment,
            )
        )
    summary = summarize(rows, mono_config.model_type)
    write_csv(dirs["metrics"] / "week02_mono_metrics.csv", rows)
    write_csv(dirs["metrics"] / "week02_mono_summary.csv", [summary])
    video_encoder = render_video(dirs["montage"], output_path / "week02_monocular_depth_demo.mp4", fps)
    write_json(
        dirs["metrics"] / "week02_mono_config.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "data_root": str(data_root),
            "week1_stereo_results": str(week1_dir),
            "baseline": baseline,
            "frames": selected_ids,
            "monocular_input": "left_camera_only",
            "model": asdict(mono_config) | {"midas_repo": str(mono_config.midas_repo), "torch_home": str(mono_config.torch_home), "resolved_device": str(model.device)},
            "stereo_depth_interface": asdict(depth_config),
            "scale_factor": scale_factor,
            "video_encoder": video_encoder,
            "output_dir": str(output_path),
        },
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Paper 2 Week 2 monocular depth baseline.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--baseline", default="ZM-Baseline-25")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--week1-dir", type=Path, default=DEFAULT_WEEK1_RESULTS)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-frames", type=int, default=15)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--midas-repo", type=Path, default=DEFAULT_MIDAS_REPO)
    parser.add_argument("--midas-model", default="MiDaS_small")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--collision-threshold-m", type=float, default=2.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    depth_config = StereoDepthConfig(collision_threshold_m=args.collision_threshold_m)
    mono_config = MonocularDepthConfig(model_type=args.midas_model, device=args.device, midas_repo=args.midas_repo)
    output_path = generate_monocular_output(
        data_root=args.data_root,
        output_dir=args.output_dir,
        week1_dir=args.week1_dir,
        baseline=args.baseline,
        max_frames=args.max_frames,
        fps=args.fps,
        run_name=args.run_name,
        depth_config=depth_config,
        mono_config=mono_config,
    )
    print(f"Week 2 monocular depth baseline complete: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
