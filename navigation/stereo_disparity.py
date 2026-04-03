from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from read_pfm import read_pfm


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "uav_stereo"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "week01_stereo_depth"


@dataclass(frozen=True)
class StereoDepthConfig:
    min_disparity: int = 128
    num_disparities: int = 64
    block_size: int = 3
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    use_wls: bool = False
    wls_lambda: float = 8000.0
    wls_sigma_color: float = 1.5
    focal_length_px: float = 400.0
    baseline_m: float = 0.15
    collision_threshold_m: float = 2.5


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "arrays": output_dir / "arrays",
        "depth": output_dir / "depth",
        "disparity": output_dir / "disparity",
        "frames": output_dir / "frames",
        "metrics": output_dir / "metrics",
        "montage": output_dir / "montage",
        "obstacles": output_dir / "obstacles",
        "risk": output_dir / "risk",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def frame_id_from_left(left_path: Path) -> str:
    stem = left_path.stem
    if not stem.endswith("_L"):
        raise ValueError(f"Unexpected UAVStereo left image name: {left_path.name}")
    return stem[:-2]


def find_frame_pairs(data_root: Path, baseline: str) -> list[dict[str, Path | str]]:
    baseline_dir = data_root / baseline
    left_dir = baseline_dir / "ImageLeft"
    right_dir = baseline_dir / "ImageRight"
    gt_dir = baseline_dir / "DispLeft"
    if not left_dir.exists():
        raise FileNotFoundError(f"UAVStereo left-image directory not found: {left_dir}")

    frames: list[dict[str, Path | str]] = []
    for left_path in sorted(left_dir.glob("*_L.png")):
        frame_id = frame_id_from_left(left_path)
        right_path = right_dir / f"{frame_id}_R.png"
        gt_path = gt_dir / f"dis{frame_id}_L.pfm"
        if right_path.exists() and gt_path.exists():
            frames.append({"id": frame_id, "left": left_path, "right": right_path, "gt": gt_path})
    if not frames:
        raise FileNotFoundError(f"No complete left/right/ground-truth frames found in {baseline_dir}")
    return frames


def select_evenly(items: list[dict[str, Path | str]], max_items: int) -> list[dict[str, Path | str]]:
    if max_items <= 0 or max_items >= len(items):
        return items
    indices = np.linspace(0, len(items) - 1, max_items)
    return [items[int(round(index))] for index in indices]


def valid_gt_mask(disparity: np.ndarray) -> np.ndarray:
    return np.isfinite(disparity) & (disparity > 0)


def valid_estimate_mask(disparity: np.ndarray, config: StereoDepthConfig) -> np.ndarray:
    minimum = max(0.0, float(config.min_disparity) + 0.5)
    return np.isfinite(disparity) & (disparity > minimum)


def compute_disparity_filtered(
    img_left_gray: np.ndarray,
    img_right_gray: np.ndarray,
    config: StereoDepthConfig | None = None,
) -> tuple[np.ndarray, float]:
    config = config or StereoDepthConfig()
    num_disparities = int(np.ceil(config.num_disparities / 16) * 16)
    block_size = config.block_size if config.block_size % 2 == 1 else config.block_size + 1

    matcher = cv2.StereoSGBM_create(
        minDisparity=config.min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=config.disp12_max_diff,
        uniquenessRatio=config.uniqueness_ratio,
        speckleWindowSize=config.speckle_window_size,
        speckleRange=config.speckle_range,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    start = time.perf_counter()
    if config.use_wls and hasattr(cv2, "ximgproc"):
        try:
            right_matcher = cv2.ximgproc.createRightMatcher(matcher)
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher)
            wls_filter.setLambda(config.wls_lambda)
            wls_filter.setSigmaColor(config.wls_sigma_color)
            disp_left = matcher.compute(img_left_gray, img_right_gray)
            disp_right = right_matcher.compute(img_right_gray, img_left_gray)
            disparity = wls_filter.filter(disp_left, img_left_gray, None, disp_right)
        except cv2.error:
            disparity = matcher.compute(img_left_gray, img_right_gray)
    else:
        disparity = matcher.compute(img_left_gray, img_right_gray)

    latency_ms = (time.perf_counter() - start) * 1000.0
    return (disparity.astype(np.float32) / 16.0), latency_ms


def disparity_to_depth(disparity: np.ndarray, config: StereoDepthConfig) -> np.ndarray:
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    mask = valid_estimate_mask(disparity, config)
    depth[mask] = (config.focal_length_px * config.baseline_m) / np.maximum(disparity[mask], 1e-6)
    return depth


def convert_disparity_to_risk(
    disparity: np.ndarray,
    config: StereoDepthConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = config or StereoDepthConfig()
    depth = disparity_to_depth(disparity, config)
    risk = np.zeros(disparity.shape, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    risk[valid] = np.clip(config.collision_threshold_m / depth[valid], 0.0, 1.0)
    obstacle = (valid & (depth < config.collision_threshold_m)).astype(np.uint8)
    return depth, risk, obstacle


def normalize_u8(values: np.ndarray, mask: np.ndarray | None = None, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    source = values.astype(np.float32, copy=True)
    valid = np.isfinite(source)
    if mask is not None:
        valid &= mask
    data = source[valid]
    if data.size == 0:
        return np.zeros(source.shape, dtype=np.uint8)
    low = float(np.percentile(data, p_low))
    high = float(np.percentile(data, p_high))
    if high <= low:
        high = low + 1.0
    normalized = np.clip((source - low) / (high - low), 0.0, 1.0)
    normalized[~np.isfinite(normalized)] = 0.0
    return (normalized * 255).astype(np.uint8)


def colorize(values: np.ndarray, colormap: int = cv2.COLORMAP_VIRIDIS, mask: np.ndarray | None = None) -> np.ndarray:
    return cv2.applyColorMap(normalize_u8(values, mask), colormap)


def colorize_risk(risk: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap((np.clip(risk, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_HOT)


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = image.copy()
    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(panel, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def resize_panel(image: np.ndarray, size: tuple[int, int] = (360, 240)) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def build_montage(panels: list[tuple[str, np.ndarray]]) -> np.ndarray:
    rendered = [resize_panel(label_panel(image, label)) for label, image in panels]
    while len(rendered) < 6:
        rendered.append(np.zeros_like(rendered[0]))
    top = np.hstack(rendered[:3])
    bottom = np.hstack(rendered[3:6])
    return np.vstack([top, bottom])


def sector_clearance(depth: np.ndarray) -> dict[str, float]:
    height, width = depth.shape
    lower_scene = depth[height // 4 :, :]
    sectors = {
        "left": lower_scene[:, : width // 3],
        "center": lower_scene[:, width // 3 : 2 * width // 3],
        "right": lower_scene[:, 2 * width // 3 :],
    }
    clearances: dict[str, float] = {}
    for name, sector in sectors.items():
        values = sector[np.isfinite(sector) & (sector > 0)]
        clearances[name] = float(np.percentile(values, 10)) if values.size else float("nan")
    return clearances


def collision_decision(depth: np.ndarray, risk: np.ndarray, config: StereoDepthConfig) -> tuple[str, dict[str, float]]:
    clearances = sector_clearance(depth)
    center = clearances["center"]
    center_risk = risk[:, risk.shape[1] // 3 : 2 * risk.shape[1] // 3]
    clearances["center_alert_fraction"] = float(np.mean(center_risk >= 0.95)) if center_risk.size else 0.0

    if np.isfinite(center) and center >= config.collision_threshold_m:
        return "FORWARD", clearances
    left = clearances["left"]
    right = clearances["right"]
    if not np.isfinite(left) and not np.isfinite(right):
        return "SLOW_OR_HOLD", clearances
    if not np.isfinite(right) or left >= right:
        return "TURN_LEFT", clearances
    return "TURN_RIGHT", clearances


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def metric_row(
    frame_id: str,
    baseline: str,
    estimate: np.ndarray,
    gt: np.ndarray,
    depth: np.ndarray,
    risk: np.ndarray,
    obstacle: np.ndarray,
    decision: str,
    clearances: dict[str, float],
    latency_ms: float,
    config: StereoDepthConfig,
) -> dict[str, object]:
    mask = valid_gt_mask(gt) & valid_estimate_mask(estimate, config)
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
        "method": "OpenCV StereoSGBM",
        "mae_disparity_px": mae,
        "rmse_disparity_px": rmse,
        "bad_pixel_3px_rate": bad_3px,
        "bad_pixel_5pct_rate": bad_5pct,
        "valid_estimate_fraction": float(np.mean(valid_estimate_mask(estimate, config))),
        "robust_min_depth_m": robust_min_depth,
        "mean_depth_m": mean_depth,
        "obstacle_fraction": float(np.mean(obstacle > 0)),
        "max_risk_fraction": float(np.mean(risk >= 0.95)),
        "decision": decision,
        "left_clearance_m": clearances["left"],
        "center_clearance_m": clearances["center"],
        "right_clearance_m": clearances["right"],
        "center_alert_fraction": clearances["center_alert_fraction"],
        "inference_latency_ms": latency_ms,
        "collision_threshold_m": config.collision_threshold_m,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
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
    ]
    summary: dict[str, object] = {
        "method": "OpenCV StereoSGBM",
        "frames": len(rows),
        "forward_count": sum(row["decision"] == "FORWARD" for row in rows),
        "turn_left_count": sum(row["decision"] == "TURN_LEFT" for row in rows),
        "turn_right_count": sum(row["decision"] == "TURN_RIGHT" for row in rows),
        "slow_or_hold_count": sum(row["decision"] == "SLOW_OR_HOLD" for row in rows),
    }
    for field in numeric_fields:
        values = np.array([float(row[field]) for row in rows], dtype=np.float64)
        summary[f"avg_{field}"] = float(np.nanmean(values))
    return summary


def render_video(montage_dir: Path, output_path: Path, fps: float) -> str:
    frames = sorted(montage_dir.glob("frame_*_week1_montage.png"))
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
                str(montage_dir / "frame_*_week1_montage.png"),
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


def generate_stereo_output(
    data_root: Path | str = DEFAULT_DATA_ROOT,
    output_dir: Path | str = DEFAULT_RESULTS_ROOT,
    baseline: str = "ZM-Baseline-25",
    max_frames: int = 8,
    fps: float = 1.0,
    run_name: str | None = None,
    config: StereoDepthConfig | None = None,
) -> Path:
    config = config or StereoDepthConfig()
    data_root = Path(data_root)
    output_root = Path(output_dir)
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_root / run_name
    dirs = ensure_dirs(output_path)

    frames = select_evenly(find_frame_pairs(data_root, baseline), max_frames)
    rows: list[dict[str, object]] = []
    selected_ids: list[str] = []

    for item in frames:
        frame_id = str(item["id"])
        selected_ids.append(frame_id)
        left = cv2.imread(str(item["left"]))
        right = cv2.imread(str(item["right"]))
        if left is None or right is None:
            raise FileNotFoundError(f"Could not load stereo frame pair {item['left']} / {item['right']}")
        gt, _ = read_pfm(item["gt"])

        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        disparity, latency_ms = compute_disparity_filtered(left_gray, right_gray, config)
        depth, risk, obstacle = convert_disparity_to_risk(disparity, config)
        decision, clearances = collision_decision(depth, risk, config)
        disparity_metric = disparity.copy()
        gt_metric = gt.copy()
        depth_metric = depth.copy()
        risk_metric = risk.copy()
        obstacle_metric = obstacle.copy()

        np.save(dirs["arrays"] / f"frame_{frame_id}_stereo_disparity.npy", disparity_metric)
        np.save(dirs["arrays"] / f"frame_{frame_id}_stereo_depth.npy", depth_metric)
        np.save(dirs["arrays"] / f"frame_{frame_id}_stereo_risk.npy", risk_metric)
        np.save(dirs["arrays"] / f"frame_{frame_id}_obstacle_mask.npy", obstacle_metric)

        gt_mask = valid_gt_mask(gt)
        disparity_vis = colorize(disparity, cv2.COLORMAP_VIRIDIS, gt_mask)
        depth_vis = colorize(depth, cv2.COLORMAP_TURBO)
        risk_vis = colorize_risk(risk)
        obstacle_vis = cv2.cvtColor(obstacle * 255, cv2.COLOR_GRAY2BGR)
        gt_vis = colorize(gt, cv2.COLORMAP_VIRIDIS, gt_mask)

        write_image(dirs["frames"] / f"frame_{frame_id}_left.png", left)
        write_image(dirs["frames"] / f"frame_{frame_id}_right.png", right)
        write_image(dirs["disparity"] / f"frame_{frame_id}_stereo_disparity.png", disparity_vis)
        write_image(dirs["disparity"] / f"frame_{frame_id}_gt_disparity.png", gt_vis)
        write_image(dirs["depth"] / f"frame_{frame_id}_depth_proxy.png", depth_vis)
        write_image(dirs["risk"] / f"frame_{frame_id}_risk_map.png", risk_vis)
        write_image(dirs["obstacles"] / f"frame_{frame_id}_obstacle_mask.png", obstacle_vis)

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
            f"threshold {config.collision_threshold_m:.1f} m",
            (12, annotated_risk.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        montage = build_montage(
            [
                ("Left RGB input", left),
                ("Right RGB input", right),
                ("StereoSGBM disparity", disparity_vis),
                ("Depth proxy", depth_vis),
                ("Obstacle risk + decision", annotated_risk),
                ("Collision proxy mask", obstacle_vis),
            ]
        )
        write_image(dirs["montage"] / f"frame_{frame_id}_week1_montage.png", montage)

        rows.append(
            metric_row(
                frame_id,
                baseline,
                disparity_metric,
                gt_metric,
                depth_metric,
                risk_metric,
                obstacle_metric,
                decision,
                clearances,
                latency_ms,
                config,
            )
        )

    summary = summarize(rows)
    write_csv(dirs["metrics"] / "week01_stereo_metrics.csv", rows)
    write_csv(dirs["metrics"] / "week01_stereo_summary.csv", [summary])
    write_json(
        dirs["metrics"] / "week01_stereo_config.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "data_root": str(data_root),
            "baseline": baseline,
            "frames": selected_ids,
            "config": asdict(config),
            "output_dir": str(output_path),
        },
    )
    render_video(dirs["montage"], output_path / "week01_stereo_depth_demo.mp4", fps)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Paper 2 Week 1 stereo depth baseline.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--baseline", default="ZM-Baseline-25")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Process two frames and write to a smoke run folder.")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--sgbm-min-disparity", type=int, default=128)
    parser.add_argument("--sgbm-num-disparities", type=int, default=64)
    parser.add_argument("--sgbm-block-size", type=int, default=3)
    parser.add_argument("--sgbm-uniqueness-ratio", type=int, default=10)
    parser.add_argument("--use-wls", action="store_true")
    parser.add_argument("--collision-threshold-m", type=float, default=2.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = StereoDepthConfig(
        min_disparity=args.sgbm_min_disparity,
        num_disparities=args.sgbm_num_disparities,
        block_size=args.sgbm_block_size,
        uniqueness_ratio=args.sgbm_uniqueness_ratio,
        use_wls=args.use_wls,
        collision_threshold_m=args.collision_threshold_m,
    )
    max_frames = 2 if args.smoke else args.max_frames
    run_name = args.run_name
    if args.smoke and run_name is None:
        run_name = "smoke"
    output_path = generate_stereo_output(
        data_root=args.data_root,
        output_dir=args.output_dir,
        baseline=args.baseline,
        max_frames=max_frames,
        fps=args.fps,
        run_name=run_name,
        config=config,
    )
    print(f"Week 1 stereo depth baseline complete: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
