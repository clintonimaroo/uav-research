from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collision_avoidance_validation import hazard_overlay, safe_float
from navigation_integration import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SCENARIOS_PATH,
    PPONavigationAgent,
    NavigationIntegrationConfig,
    episode_plan,
    load_risk_map,
    write_csv,
    write_json,
)
from performance_evaluation import (
    DEFAULT_MONO_METRICS,
    DEFAULT_STEREO_METRICS,
    PerformanceEvaluationConfig,
    perception_latency,
    read_latency_metrics,
    run_episode,
)


DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "week07_robustness"
DEFAULT_WEEK6_SUMMARY = SCRIPT_DIR / "results" / "week06_performance_evaluation" / "final" / "metrics" / "week06_summary.csv"
PERTURBATION_TYPES = ["depth_noise", "image_blur", "reduced_resolution", "partial_occlusion"]
SEVERITIES = ["low", "medium", "high", "extreme"]


@dataclass(frozen=True)
class RobustnessConfig:
    grid_size: int = 50
    max_steps: int = 200
    episodes_per_condition: int = 500
    collision_risk_threshold: float = 0.95
    shield_caution_threshold: float = 0.50
    hazard_threshold: float = 0.70
    goal_threshold: float = 1.0
    start_goal_risk_threshold: float = 0.5
    policy_risk_scale: float = 1.0
    lookahead_steps: int = 3
    risk_penalty_weight: float = 20.0
    progress_weight: float = 5.0
    loop_penalty_weight: float = 3.0


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "metrics": output_dir / "metrics",
        "visuals": output_dir / "visuals",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def seed_for(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % (2**32 - 1)


def severity_value(perturbation_type: str, severity: str) -> float:
    values = {
        "depth_noise": {"low": 0.04, "medium": 0.08, "high": 0.14, "extreme": 0.22},
        "image_blur": {"low": 3.0, "medium": 5.0, "high": 9.0, "extreme": 13.0},
        "reduced_resolution": {"low": 0.75, "medium": 0.50, "high": 0.33, "extreme": 0.25},
        "partial_occlusion": {"low": 0.08, "medium": 0.16, "high": 0.25, "extreme": 0.35},
    }
    return values[perturbation_type][severity]


def perturb_risk_map(risk: np.ndarray, perturbation_type: str, severity: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value = severity_value(perturbation_type, severity)
    if perturbation_type == "depth_noise":
        output = risk + rng.normal(0.0, value, risk.shape).astype(np.float32)
    elif perturbation_type == "image_blur":
        kernel = int(value)
        output = cv2.GaussianBlur(risk, (kernel, kernel), 0)
    elif perturbation_type == "reduced_resolution":
        height, width = risk.shape
        reduced_width = max(4, int(round(width * value)))
        reduced_height = max(4, int(round(height * value)))
        small = cv2.resize(risk, (reduced_width, reduced_height), interpolation=cv2.INTER_AREA)
        output = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    elif perturbation_type == "partial_occlusion":
        output = risk.copy()
        height, width = risk.shape
        target_area = max(1, int(round(height * width * value)))
        covered = 0
        while covered < target_area:
            block_height = int(rng.integers(max(3, height // 10), max(4, height // 4)))
            block_width = int(rng.integers(max(3, width // 10), max(4, width // 4)))
            row = int(rng.integers(0, max(1, height - block_height + 1)))
            col = int(rng.integers(0, max(1, width - block_width + 1)))
            output[row : row + block_height, col : col + block_width] = 0.0
            covered += block_height * block_width
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")
    output = np.clip(output.astype(np.float32), 0.0, 1.0)
    output[~np.isfinite(output)] = 0.0
    return output


def read_week6_baseline(path: Path) -> dict[str, dict[str, float]]:
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["controller"] == "ppo_shield":
                rows[row["mode"]] = {key: float(value) for key, value in row.items() if key not in {"controller", "mode"}}
    return rows


def grouped_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    keys = sorted({(row["mode"], row["perturbation_type"], row["severity"]) for row in rows})
    for mode, perturbation_type, severity in keys:
        selected = [row for row in rows if row["mode"] == mode and row["perturbation_type"] == perturbation_type and row["severity"] == severity]
        runs = len(selected)
        successes = sum(int(row["success"]) for row in selected)
        collisions = sum(int(row["collided"]) for row in selected)
        timeouts = sum(1 for row in selected if row["done_reason"] == "max_steps")
        output.append(
            {
                "controller": "ppo_shield",
                "mode": mode,
                "perturbation_type": perturbation_type,
                "severity": severity,
                "severity_value": selected[0]["severity_value"] if selected else 0.0,
                "runs": runs,
                "successes": successes,
                "success_rate": successes / runs if runs else 0.0,
                "collisions": collisions,
                "collision_rate": collisions / runs if runs else 0.0,
                "timeouts": timeouts,
                "timeout_rate": timeouts / runs if runs else 0.0,
                "avg_steps": safe_float(float(np.mean([float(row["steps"]) for row in selected]))),
                "avg_final_distance": safe_float(float(np.mean([float(row["final_distance"]) for row in selected]))),
                "avg_path_length": safe_float(float(np.mean([float(row["path_length"]) for row in selected]))),
                "avg_path_efficiency": safe_float(float(np.mean([float(row["path_efficiency"]) for row in selected]))),
                "avg_progress_efficiency": safe_float(float(np.mean([float(row["progress_efficiency"]) for row in selected]))),
                "avg_min_obstacle_distance_cells": safe_float(float(np.mean([float(row["min_obstacle_distance_cells"]) for row in selected]))),
                "avg_hazard_exposure_steps": safe_float(float(np.mean([float(row["hazard_exposure_steps"]) for row in selected]))),
                "avg_shield_interventions": safe_float(float(np.mean([float(row["shield_interventions"]) for row in selected]))),
                "avg_total_inference_latency_ms": safe_float(float(np.mean([float(row["avg_total_inference_latency_ms"]) for row in selected]))),
                "avg_perturbation_latency_ms": safe_float(float(np.mean([float(row["perturbation_latency_ms"]) for row in selected]))),
            }
        )
    return output


def degradation_rows(summary_rows: list[dict[str, object]], baseline: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    rows = []
    for row in summary_rows:
        base = baseline[str(row["mode"])]
        rows.append(
            {
                "mode": row["mode"],
                "perturbation_type": row["perturbation_type"],
                "severity": row["severity"],
                "clean_success_rate": base["success_rate"],
                "perturbed_success_rate": row["success_rate"],
                "success_rate_drop": base["success_rate"] - float(row["success_rate"]),
                "clean_collision_rate": base["collision_rate"],
                "perturbed_collision_rate": row["collision_rate"],
                "collision_rate_increase": float(row["collision_rate"]) - base["collision_rate"],
                "clean_timeout_rate": base["timeout_rate"],
                "perturbed_timeout_rate": row["timeout_rate"],
                "timeout_rate_increase": float(row["timeout_rate"]) - base["timeout_rate"],
                "clean_path_efficiency": base["avg_path_efficiency"],
                "perturbed_path_efficiency": row["avg_path_efficiency"],
                "path_efficiency_reduction": base["avg_path_efficiency"] - float(row["avg_path_efficiency"]),
                "clean_min_obstacle_distance_cells": base["avg_min_obstacle_distance_cells"],
                "perturbed_min_obstacle_distance_cells": row["avg_min_obstacle_distance_cells"],
                "min_obstacle_distance_reduction": base["avg_min_obstacle_distance_cells"] - float(row["avg_min_obstacle_distance_cells"]),
                "clean_total_inference_latency_ms": base["avg_total_inference_latency_ms"],
                "perturbed_total_inference_latency_ms": row["avg_total_inference_latency_ms"],
                "inference_latency_change_ms": float(row["avg_total_inference_latency_ms"]) - base["avg_total_inference_latency_ms"],
            }
        )
    return rows


def failure_mode_rows(rows: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for summary in summary_rows:
        selected = [row for row in rows if row["mode"] == summary["mode"] and row["perturbation_type"] == summary["perturbation_type"] and row["severity"] == summary["severity"]]
        runs = len(selected)
        low_clearance = sum(1 for row in selected if float(row["min_obstacle_distance_cells"]) < 1.0)
        high_exposure = sum(1 for row in selected if float(row["hazard_exposure_steps"]) > 0)
        intervention_values = [float(row["shield_interventions"]) for row in selected]
        threshold = float(np.percentile(intervention_values, 75)) if intervention_values else 0.0
        intervention_spikes = sum(1 for value in intervention_values if value > threshold)
        inefficient = sum(1 for row in selected if int(row["success"]) == 1 and float(row["path_efficiency"]) < 0.5)
        output.append(
            {
                "mode": summary["mode"],
                "perturbation_type": summary["perturbation_type"],
                "severity": summary["severity"],
                "runs": runs,
                "collisions": summary["collisions"],
                "timeouts": summary["timeouts"],
                "low_clearance_runs": low_clearance,
                "low_clearance_rate": low_clearance / runs if runs else 0.0,
                "hazard_exposure_runs": high_exposure,
                "hazard_exposure_rate": high_exposure / runs if runs else 0.0,
                "shield_intervention_spike_runs": intervention_spikes,
                "shield_intervention_spike_rate": intervention_spikes / runs if runs else 0.0,
                "inefficient_success_runs": inefficient,
                "inefficient_success_rate": inefficient / runs if runs else 0.0,
            }
        )
    return output


def write_degradation_plots(output_dir: Path, rows: list[dict[str, object]]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    severity_order = {severity: idx for idx, severity in enumerate(SEVERITIES)}
    metrics = [
        ("perturbed_success_rate", "Success Rate", "week07_degradation_success_rate.png"),
        ("perturbed_collision_rate", "Collision Rate", "week07_degradation_collision_rate.png"),
        ("perturbed_path_efficiency", "Path Efficiency", "week07_degradation_path_efficiency.png"),
        ("perturbed_min_obstacle_distance_cells", "Minimum Obstacle Distance", "week07_degradation_min_obstacle_distance.png"),
    ]
    outputs = []
    for field, title, filename in metrics:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        for ax, perturbation_type in zip(axes.ravel(), PERTURBATION_TYPES):
            for mode, color, marker in [("mono", "#1f77b4", "o"), ("stereo", "#d62728", "s")]:
                selected = [
                    row for row in rows
                    if row["mode"] == mode and row["perturbation_type"] == perturbation_type
                ]
                selected = sorted(selected, key=lambda row: severity_order[str(row["severity"])])
                x_values = [severity_order[str(row["severity"])] for row in selected]
                y_values = [float(row[field]) for row in selected]
                ax.plot(x_values, y_values, marker=marker, linewidth=2, color=color, label=mode)
            ax.set_title(perturbation_type.replace("_", " ").title())
            ax.set_xticks(list(severity_order.values()), SEVERITIES)
            ax.grid(True, alpha=0.3)
        axes[0, 0].set_ylabel(title)
        axes[1, 0].set_ylabel(title)
        axes[1, 0].set_xlabel("Severity")
        axes[1, 1].set_xlabel("Severity")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.suptitle(f"Week 7 Robustness: {title}", y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = output_dir / filename
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        outputs.append(str(path))
    return outputs


def selected_visual_condition(mode: str, perturbation_type: str, severity: str, scenario_id: str, variant: int) -> bool:
    return mode in {"stereo", "mono"} and perturbation_type in {"depth_noise", "partial_occlusion"} and severity == "extreme" and scenario_id == "baseline_open" and variant == 0


def annotate_rows(rows: list[dict[str, object]], values: dict[str, object]) -> list[dict[str, object]]:
    return [{**values, **row} for row in rows]


def run_week7(
    scenarios_path: Path,
    model_path: Path,
    output_dir: Path,
    run_name: str,
    selected_scenarios: list[str] | None,
    selected_perturbations: list[str] | None,
    selected_severities: list[str] | None,
    config: RobustnessConfig,
    stereo_metrics_path: Path,
    mono_metrics_path: Path,
    week6_summary_path: Path,
) -> Path:
    output_path = output_dir / run_name
    ensure_dirs(output_path)
    scenarios = json.loads(scenarios_path.read_text())["scenarios"]
    if selected_scenarios:
        selected = set(selected_scenarios)
        scenarios = [scenario for scenario in scenarios if str(scenario["id"]) in selected]
    perturbations = selected_perturbations if selected_perturbations else PERTURBATION_TYPES
    severities = selected_severities if selected_severities else SEVERITIES
    plan_config = NavigationIntegrationConfig(
        grid_size=config.grid_size,
        max_steps=config.max_steps,
        collision_risk_threshold=config.collision_risk_threshold,
        goal_threshold=config.goal_threshold,
        episodes=config.episodes_per_condition,
        start_goal_risk_threshold=config.start_goal_risk_threshold,
        policy_risk_scale=config.policy_risk_scale,
    )
    run_config = PerformanceEvaluationConfig(
        grid_size=config.grid_size,
        max_steps=config.max_steps,
        episodes=config.episodes_per_condition,
        collision_risk_threshold=config.collision_risk_threshold,
        shield_caution_threshold=config.shield_caution_threshold,
        hazard_threshold=config.hazard_threshold,
        goal_threshold=config.goal_threshold,
        start_goal_risk_threshold=config.start_goal_risk_threshold,
        policy_risk_scale=config.policy_risk_scale,
        lookahead_steps=config.lookahead_steps,
        risk_penalty_weight=config.risk_penalty_weight,
        progress_weight=config.progress_weight,
        loop_penalty_weight=config.loop_penalty_weight,
    )
    planned = episode_plan(scenarios, plan_config)
    agent = PPONavigationAgent((config.grid_size, config.grid_size, 4), 8)
    if not agent.load_model(str(model_path)):
        raise RuntimeError(f"PPO checkpoint could not be loaded: {model_path}")
    stereo_latencies = read_latency_metrics(stereo_metrics_path)
    mono_latencies = read_latency_metrics(mono_metrics_path)
    baseline = read_week6_baseline(week6_summary_path)
    episode_rows = []
    step_rows = []
    intervention_rows = []
    event_rows = []
    selected_visuals = []
    episode_index = 1
    for perturbation_type in perturbations:
        for severity in severities:
            for scenario, mode, variant in planned:
                key = "stereo_risk_map" if mode == "stereo" else "mono_risk_map"
                clean_obstacle_risk = load_risk_map(scenario[key], config.grid_size)
                seed = seed_for(perturbation_type, severity, mode, scenario["id"], scenario["frame_id"], variant)
                perturb_start = time.perf_counter_ns()
                perceived_obstacle_risk = perturb_risk_map(clean_obstacle_risk, perturbation_type, severity, seed)
                perturb_latency_ms = (time.perf_counter_ns() - perturb_start) / 1_000_000.0
                hazard_risk = hazard_overlay(clean_obstacle_risk, str(scenario["id"]), int(variant), run_config)
                base_latency_ms = perception_latency(str(scenario["frame_id"]), mode, stereo_latencies, mono_latencies)
                metadata = {
                    "perturbation_type": perturbation_type,
                    "severity": severity,
                    "severity_value": severity_value(perturbation_type, severity),
                    "perturbation_seed": seed,
                    "perturbation_latency_ms": perturb_latency_ms,
                    "base_perception_latency_ms": base_latency_ms,
                }
                render_visual = selected_visual_condition(mode, perturbation_type, severity, str(scenario["id"]), int(variant))
                visual_label = f"week07_selected_{mode}_{perturbation_type}_{severity}_trajectory" if render_visual else None
                summary, steps, interventions, events, visual = run_episode(
                    scenario,
                    mode,
                    "ppo_shield",
                    perceived_obstacle_risk,
                    hazard_risk,
                    agent,
                    run_config,
                    output_path,
                    episode_index,
                    int(variant),
                    base_latency_ms + perturb_latency_ms,
                    render_visual=render_visual,
                    visual_label=visual_label,
                    physical_obstacle_risk=clean_obstacle_risk,
                )
                episode_rows.append({**metadata, **summary})
                step_rows.extend(annotate_rows(steps, metadata))
                intervention_rows.extend(annotate_rows(interventions, metadata))
                event_rows.extend(annotate_rows(events, metadata))
                if visual:
                    selected_visuals.append(visual)
                episode_index += 1
    summary = grouped_summary(episode_rows)
    degradation = degradation_rows(summary, baseline)
    failures = failure_mode_rows(episode_rows, summary)
    curve_visuals = write_degradation_plots(output_path / "visuals", degradation)
    selected_visuals.extend(curve_visuals)
    write_csv(output_path / "metrics" / "week07_episode_results.csv", episode_rows)
    write_csv(output_path / "metrics" / "week07_step_results.csv", step_rows)
    write_csv(output_path / "metrics" / "week07_interventions.csv", intervention_rows)
    write_csv(output_path / "metrics" / "week07_events.csv", event_rows)
    write_csv(output_path / "metrics" / "week07_summary.csv", summary)
    write_csv(output_path / "metrics" / "week07_degradation_curves.csv", degradation)
    write_csv(output_path / "metrics" / "week07_failure_modes.csv", failures)
    write_json(
        output_path / "metrics" / "week07_config.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "scenarios_path": str(scenarios_path),
            "model_path": str(model_path),
            "stereo_metrics_path": str(stereo_metrics_path),
            "mono_metrics_path": str(mono_metrics_path),
            "week6_summary_path": str(week6_summary_path),
            "input_shape": [config.grid_size, config.grid_size, 4],
            "action_dim": 8,
            "training_updates": 0,
            "controller": "ppo_shield",
            "perturbation_types": perturbations,
            "severities": severities,
            "config": asdict(config),
            "selected_visuals": selected_visuals,
            "output_dir": str(output_path),
        },
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Paper 2 Week 7 robustness evaluation.")
    parser.add_argument("--scenarios-path", type=Path, default=DEFAULT_SCENARIOS_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default="final")
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--perturbation", action="append", choices=PERTURBATION_TYPES, default=None)
    parser.add_argument("--severity", action="append", choices=SEVERITIES, default=None)
    parser.add_argument("--stereo-metrics", type=Path, default=DEFAULT_STEREO_METRICS)
    parser.add_argument("--mono-metrics", type=Path, default=DEFAULT_MONO_METRICS)
    parser.add_argument("--week6-summary", type=Path, default=DEFAULT_WEEK6_SUMMARY)
    parser.add_argument("--episodes-per-condition", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--collision-risk-threshold", type=float, default=0.95)
    parser.add_argument("--shield-caution-threshold", type=float, default=0.50)
    parser.add_argument("--hazard-threshold", type=float, default=0.70)
    parser.add_argument("--goal-threshold", type=float, default=1.0)
    parser.add_argument("--start-goal-risk-threshold", type=float, default=0.5)
    parser.add_argument("--policy-risk-scale", type=float, default=1.0)
    parser.add_argument("--lookahead-steps", type=int, default=3)
    parser.add_argument("--risk-penalty-weight", type=float, default=20.0)
    parser.add_argument("--progress-weight", type=float, default=5.0)
    parser.add_argument("--loop-penalty-weight", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = RobustnessConfig(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        episodes_per_condition=args.episodes_per_condition,
        collision_risk_threshold=args.collision_risk_threshold,
        shield_caution_threshold=args.shield_caution_threshold,
        hazard_threshold=args.hazard_threshold,
        goal_threshold=args.goal_threshold,
        start_goal_risk_threshold=args.start_goal_risk_threshold,
        policy_risk_scale=args.policy_risk_scale,
        lookahead_steps=args.lookahead_steps,
        risk_penalty_weight=args.risk_penalty_weight,
        progress_weight=args.progress_weight,
        loop_penalty_weight=args.loop_penalty_weight,
    )
    output_path = run_week7(
        scenarios_path=args.scenarios_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        selected_scenarios=args.scenario,
        selected_perturbations=args.perturbation,
        selected_severities=args.severity,
        config=config,
        stereo_metrics_path=args.stereo_metrics,
        mono_metrics_path=args.mono_metrics,
        week6_summary_path=args.week6_summary,
    )
    print(f"Week 7 robustness evaluation complete: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
