from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stereo_disparity import build_montage, colorize, colorize_risk, write_csv, write_image, write_json


DEFAULT_WEEK1_DIR = SCRIPT_DIR / "results" / "week01_stereo_depth" / "final"
DEFAULT_WEEK2_DIR = SCRIPT_DIR / "results" / "week02_monocular_depth" / "final"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "week03_sensor_alignment"
DEFAULT_SCENARIOS_PATH = SCRIPT_DIR / "scenarios.json"


@dataclass(frozen=True)
class AlignmentConfig:
    grid_size: int = 50
    collision_threshold_m: float = 2.5
    risk_threshold: float = 0.5
    max_frames: int = 15


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "heatmaps": output_dir / "heatmaps",
        "metrics": output_dir / "metrics",
        "montage": output_dir / "montage",
        "normalized": output_dir / "normalized",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def frame_ids_from_config(week1_dir: Path, week2_dir: Path, max_frames: int) -> list[str]:
    week1_config = load_json(week1_dir / "metrics" / "week01_stereo_config.json")
    week2_config = load_json(week2_dir / "metrics" / "week02_mono_config.json")
    week1_ids = [str(frame_id) for frame_id in week1_config["frames"]]
    week2_ids = [str(frame_id) for frame_id in week2_config["frames"]]
    if week1_ids != week2_ids:
        raise RuntimeError("Week 1 and Week 2 frame IDs do not match")
    return week1_ids[:max_frames]


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def normalize_risk(values: np.ndarray) -> np.ndarray:
    normalized = np.clip(values.astype(np.float32), 0.0, 1.0)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized


def normalize_obstacle(values: np.ndarray) -> np.ndarray:
    return (values > 0).astype(np.uint8)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1).astype(np.float64)
    y = b.reshape(-1).astype(np.float64)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    x = np.clip(a.astype(np.float64), 0.0, 1.0)
    y = np.clip(b.astype(np.float64), 0.0, 1.0)
    c1 = 0.01**2
    c2 = 0.03**2
    mux = float(np.mean(x))
    muy = float(np.mean(y))
    vx = float(np.var(x))
    vy = float(np.var(y))
    cov = float(np.mean((x - mux) * (y - muy)))
    numerator = (2.0 * mux * muy + c1) * (2.0 * cov + c2)
    denominator = (mux**2 + muy**2 + c1) * (vx + vy + c2)
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def depth_alignment(stereo_depth: np.ndarray, mono_depth: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(stereo_depth) & np.isfinite(mono_depth) & (stereo_depth > 0) & (mono_depth > 0)
    if not np.any(mask):
        return {"depth_scale": 0.0, "depth_mae_m": 0.0, "depth_rmse_m": 0.0, "depth_valid_overlap": 0.0}
    scale = float(np.median(stereo_depth[mask]) / max(float(np.median(mono_depth[mask])), 1e-6))
    aligned = mono_depth * scale
    error = aligned[mask] - stereo_depth[mask]
    return {
        "depth_scale": scale,
        "depth_mae_m": float(np.mean(np.abs(error))),
        "depth_rmse_m": float(np.sqrt(np.mean(error**2))),
        "depth_valid_overlap": float(np.mean(mask)),
    }


def alignment_row(
    frame_id: str,
    week1_dir: Path,
    week2_dir: Path,
    config: AlignmentConfig,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    stereo_risk = normalize_risk(load_array(week1_dir / "arrays" / f"frame_{frame_id}_stereo_risk.npy"))
    mono_risk = normalize_risk(load_array(week2_dir / "arrays" / f"frame_{frame_id}_mono_risk.npy"))
    stereo_obstacle = normalize_obstacle(load_array(week1_dir / "arrays" / f"frame_{frame_id}_obstacle_mask.npy"))
    mono_obstacle = normalize_obstacle(load_array(week2_dir / "arrays" / f"frame_{frame_id}_obstacle_mask.npy"))
    stereo_depth = load_array(week1_dir / "arrays" / f"frame_{frame_id}_stereo_depth.npy")
    mono_depth = load_array(week2_dir / "arrays" / f"frame_{frame_id}_mono_depth.npy")
    if stereo_risk.shape != mono_risk.shape or stereo_obstacle.shape != mono_obstacle.shape:
        raise RuntimeError(f"Mono/stereo array shape mismatch for frame {frame_id}")
    diff = np.abs(stereo_risk - mono_risk)
    stereo_binary = stereo_risk >= config.risk_threshold
    mono_binary = mono_risk >= config.risk_threshold
    obstacle_union = np.logical_or(stereo_obstacle > 0, mono_obstacle > 0)
    obstacle_intersection = np.logical_and(stereo_obstacle > 0, mono_obstacle > 0)
    risk_error = stereo_risk - mono_risk
    depth_metrics = depth_alignment(stereo_depth, mono_depth)
    row = {
        "frame_id": frame_id,
        "risk_shape_match": True,
        "obstacle_shape_match": True,
        "risk_min_stereo": float(np.min(stereo_risk)),
        "risk_max_stereo": float(np.max(stereo_risk)),
        "risk_min_mono": float(np.min(mono_risk)),
        "risk_max_mono": float(np.max(mono_risk)),
        "risk_mae": float(np.mean(diff)),
        "risk_rmse": float(np.sqrt(np.mean(risk_error**2))),
        "risk_correlation": safe_corr(stereo_risk, mono_risk),
        "risk_ssim": ssim_global(stereo_risk, mono_risk),
        "binary_obstacle_agreement": float(np.mean(stereo_binary == mono_binary)),
        "obstacle_iou": float(np.sum(obstacle_intersection) / max(float(np.sum(obstacle_union)), 1.0)),
        "stereo_obstacle_fraction": float(np.mean(stereo_obstacle > 0)),
        "mono_obstacle_fraction": float(np.mean(mono_obstacle > 0)),
        "risk_threshold": config.risk_threshold,
        "collision_threshold_m": config.collision_threshold_m,
        **depth_metrics,
    }
    arrays = {
        "stereo_risk": stereo_risk,
        "mono_risk": mono_risk,
        "stereo_obstacle": stereo_obstacle,
        "mono_obstacle": mono_obstacle,
        "risk_diff": diff,
    }
    return row, arrays


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric_fields = [
        "risk_mae",
        "risk_rmse",
        "risk_correlation",
        "risk_ssim",
        "binary_obstacle_agreement",
        "obstacle_iou",
        "stereo_obstacle_fraction",
        "mono_obstacle_fraction",
        "depth_scale",
        "depth_mae_m",
        "depth_rmse_m",
        "depth_valid_overlap",
    ]
    summary: dict[str, object] = {"frames": len(rows)}
    for field in numeric_fields:
        values = np.array([float(row[field]) for row in rows], dtype=np.float64)
        summary[f"avg_{field}"] = float(np.mean(values))
        summary[f"min_{field}"] = float(np.min(values))
        summary[f"max_{field}"] = float(np.max(values))
    return summary


def scenario_frame(rows: list[dict[str, object]], key: str, reverse: bool = False, used: set[str] | None = None) -> str:
    if used is None:
        used = set()
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=reverse)
    for row in ordered:
        frame_id = str(row["frame_id"])
        if frame_id not in used:
            used.add(frame_id)
            return frame_id
    frame_id = str(ordered[0]["frame_id"])
    used.add(frame_id)
    return frame_id


def write_scenarios(rows: list[dict[str, object]], output_path: Path, config: AlignmentConfig) -> None:
    used: set[str] = set()
    selections = {
        "baseline_open": scenario_frame(rows, "stereo_obstacle_fraction", False, used),
        "dense_corridor": scenario_frame(rows, "stereo_obstacle_fraction", True, used),
        "obstacle_cluster": scenario_frame(rows, "binary_obstacle_agreement", False, used),
        "lateral_maneuver": scenario_frame(rows, "risk_correlation", False, used),
        "stress_density": scenario_frame(rows, "risk_rmse", True, used),
    }
    starts_goals = {
        "baseline_open": ([2, 2], [47, 47]),
        "dense_corridor": ([5, 5], [45, 45]),
        "obstacle_cluster": ([2, 47], [47, 2]),
        "lateral_maneuver": ([10, 5], [40, 45]),
        "stress_density": ([5, 45], [45, 5]),
    }
    names = {
        "baseline_open": "Baseline Open",
        "dense_corridor": "Dense Corridor",
        "obstacle_cluster": "Obstacle Cluster",
        "lateral_maneuver": "Lateral Maneuver",
        "stress_density": "Stress Density",
    }
    notes = {
        "baseline_open": "Lowest stereo obstacle fraction among the locked frame set.",
        "dense_corridor": "Highest stereo obstacle fraction among the locked frame set.",
        "obstacle_cluster": "Low mono-stereo obstacle agreement, useful for validation stress.",
        "lateral_maneuver": "Low mono-stereo risk correlation, useful for lateral planning checks.",
        "stress_density": "High mono-stereo risk RMSE, useful for worst-case comparison.",
    }
    scenarios: dict[str, object] = {
        "version": 1,
        "created_for": "paper2_week3_sensor_alignment",
        "grid_size": config.grid_size,
        "collision_threshold_m": config.collision_threshold_m,
        "scenarios": [],
    }
    for scenario_id, frame_id in selections.items():
        start, goal = starts_goals[scenario_id]
        scenarios["scenarios"].append(
            {
                "id": scenario_id,
                "name": names[scenario_id],
                "frame_id": frame_id,
                "grid_size": config.grid_size,
                "start_cell": start,
                "goal_cell": goal,
                "stereo_risk_map": f"navigation/results/week01_stereo_depth/final/arrays/frame_{frame_id}_stereo_risk.npy",
                "mono_risk_map": f"navigation/results/week02_monocular_depth/final/arrays/frame_{frame_id}_mono_risk.npy",
                "stereo_obstacle_map": f"navigation/results/week01_stereo_depth/final/arrays/frame_{frame_id}_obstacle_mask.npy",
                "mono_obstacle_map": f"navigation/results/week02_monocular_depth/final/arrays/frame_{frame_id}_obstacle_mask.npy",
                "collision_threshold_m": config.collision_threshold_m,
                "notes": notes[scenario_id],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scenarios, indent=2))


def render_artifacts(frame_id: str, arrays: dict[str, np.ndarray], dirs: dict[str, Path]) -> None:
    stereo_risk_vis = colorize_risk(arrays["stereo_risk"])
    mono_risk_vis = colorize_risk(arrays["mono_risk"])
    diff_vis = colorize(arrays["risk_diff"], cv2.COLORMAP_INFERNO)
    stereo_obstacle_vis = cv2.cvtColor(arrays["stereo_obstacle"] * 255, cv2.COLOR_GRAY2BGR)
    mono_obstacle_vis = cv2.cvtColor(arrays["mono_obstacle"] * 255, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(stereo_risk_vis, 0.5, mono_risk_vis, 0.5, 0)
    np.save(dirs["normalized"] / f"frame_{frame_id}_stereo_risk.npy", arrays["stereo_risk"])
    np.save(dirs["normalized"] / f"frame_{frame_id}_mono_risk.npy", arrays["mono_risk"])
    np.save(dirs["normalized"] / f"frame_{frame_id}_risk_diff.npy", arrays["risk_diff"])
    write_image(dirs["heatmaps"] / f"frame_{frame_id}_risk_difference.png", diff_vis)
    montage = build_montage(
        [
            ("Stereo risk", stereo_risk_vis),
            ("Mono risk", mono_risk_vis),
            ("Risk difference", diff_vis),
            ("Stereo obstacle", stereo_obstacle_vis),
            ("Mono obstacle", mono_obstacle_vis),
            ("Risk overlay", overlay),
        ]
    )
    write_image(dirs["montage"] / f"frame_{frame_id}_week3_alignment.png", montage)


def generate_alignment_output(
    week1_dir: Path | str = DEFAULT_WEEK1_DIR,
    week2_dir: Path | str = DEFAULT_WEEK2_DIR,
    output_dir: Path | str = DEFAULT_RESULTS_ROOT,
    run_name: str | None = None,
    scenarios_path: Path | str = DEFAULT_SCENARIOS_PATH,
    config: AlignmentConfig | None = None,
) -> Path:
    config = config or AlignmentConfig()
    week1_dir = Path(week1_dir)
    week2_dir = Path(week2_dir)
    output_root = Path(output_dir)
    scenarios_path = Path(scenarios_path)
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_root / run_name
    dirs = ensure_dirs(output_path)
    frame_ids = frame_ids_from_config(week1_dir, week2_dir, config.max_frames)
    rows: list[dict[str, object]] = []
    for frame_id in frame_ids:
        row, arrays = alignment_row(frame_id, week1_dir, week2_dir, config)
        render_artifacts(frame_id, arrays, dirs)
        rows.append(row)
    summary = summarize(rows)
    write_csv(dirs["metrics"] / "week03_alignment_metrics.csv", rows)
    write_csv(dirs["metrics"] / "week03_alignment_summary.csv", [summary])
    write_scenarios(rows, scenarios_path, config)
    write_json(
        dirs["metrics"] / "week03_alignment_config.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "week1_stereo_results": str(week1_dir),
            "week2_monocular_results": str(week2_dir),
            "scenarios_path": str(scenarios_path),
            "frames": frame_ids,
            "config": asdict(config),
            "output_dir": str(output_path),
        },
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Paper 2 Week 3 sensor alignment validation.")
    parser.add_argument("--week1-dir", type=Path, default=DEFAULT_WEEK1_DIR)
    parser.add_argument("--week2-dir", type=Path, default=DEFAULT_WEEK2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--scenarios-path", type=Path, default=DEFAULT_SCENARIOS_PATH)
    parser.add_argument("--grid-size", type=int, default=50)
    parser.add_argument("--collision-threshold-m", type=float, default=2.5)
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AlignmentConfig(
        grid_size=args.grid_size,
        collision_threshold_m=args.collision_threshold_m,
        risk_threshold=args.risk_threshold,
        max_frames=args.max_frames,
    )
    output_path = generate_alignment_output(
        week1_dir=args.week1_dir,
        week2_dir=args.week2_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        scenarios_path=args.scenarios_path,
        config=config,
    )
    print(f"Week 3 sensor alignment validation complete: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
