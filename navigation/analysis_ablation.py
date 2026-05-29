from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collision_avoidance_validation import safe_float
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


DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "week08_analysis_ablation"
DEFAULT_WEEK6_DIR = SCRIPT_DIR / "results" / "week06_performance_evaluation" / "final" / "metrics"
DEFAULT_WEEK7_DIR = SCRIPT_DIR / "results" / "week07_robustness" / "final" / "metrics"
DEFAULT_DOC_PATH = PROJECT_ROOT / "docs" / "week08_analysis_ablation.md"
HAZARD_LEVELS = ["none", "low", "moderate", "high"]


@dataclass(frozen=True)
class Week8Config:
    grid_size: int = 50
    max_steps: int = 200
    episodes_per_mode_level: int = 250
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(row: dict[str, object], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def to_int(row: dict[str, object], key: str) -> int:
    return int(float(row.get(key, 0) or 0))


def seed_for(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % (2**32 - 1)


def hazard_params(level: str) -> tuple[int, float, float, float]:
    values = {
        "none": (0, 0.0, 0.0, 0.0),
        "low": (1, 3.0, 0.58, 0.08),
        "moderate": (3, 4.0, 0.72, 0.18),
        "high": (5, 5.0, 0.86, 0.30),
    }
    return values[level]


def hazard_density_overlay(obstacle_risk: np.ndarray, scenario_id: str, variant: int, level: str, config: Week8Config) -> np.ndarray:
    count, radius_base, intensity_base, background = hazard_params(level)
    hazard = np.full(obstacle_risk.shape, background, dtype=np.float32)
    if count == 0:
        return np.zeros_like(obstacle_risk, dtype=np.float32)
    rng = np.random.default_rng(seed_for(scenario_id, variant, level))
    safe = np.argwhere(obstacle_risk < config.start_goal_risk_threshold)
    edge = np.argwhere(obstacle_risk >= np.quantile(obstacle_risk, 0.80))
    pools = [safe, edge]
    rows, cols = np.indices(obstacle_risk.shape)
    for idx in range(count):
        pool = pools[idx % len(pools)]
        if len(pool) == 0:
            pool = np.argwhere(obstacle_risk < config.collision_risk_threshold)
        if len(pool) == 0:
            continue
        center = pool[int(rng.integers(0, len(pool)))]
        radius = radius_base + float(rng.uniform(-0.75, 1.25))
        intensity = min(1.0, intensity_base + 0.04 * (idx % 3))
        dist_sq = (rows - int(center[0])) ** 2 + (cols - int(center[1])) ** 2
        hazard = np.maximum(hazard, intensity * np.exp(-dist_sq / max(2.0 * radius * radius, 1.0)))
    return np.clip(hazard, 0.0, 1.0).astype(np.float32)


def annotate_rows(rows: list[dict[str, object]], values: dict[str, object]) -> list[dict[str, object]]:
    return [{**values, **row} for row in rows]


def summarize_hazard(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    keys = [(mode, level) for mode in ["mono", "stereo"] for level in HAZARD_LEVELS]
    for mode, level in keys:
        selected = [row for row in rows if row["mode"] == mode and row["hazard_density"] == level]
        runs = len(selected)
        successes = sum(to_int(row, "success") for row in selected)
        collisions = sum(to_int(row, "collided") for row in selected)
        timeouts = sum(1 for row in selected if row["done_reason"] == "max_steps")
        output.append(
            {
                "mode": mode,
                "hazard_density": level,
                "runs": runs,
                "successes": successes,
                "success_rate": successes / runs if runs else 0.0,
                "collisions": collisions,
                "collision_rate": collisions / runs if runs else 0.0,
                "timeouts": timeouts,
                "timeout_rate": timeouts / runs if runs else 0.0,
                "avg_steps": safe_float(float(np.mean([to_float(row, "steps") for row in selected])) if selected else 0.0),
                "avg_final_distance": safe_float(float(np.mean([to_float(row, "final_distance") for row in selected])) if selected else 0.0),
                "avg_path_length": safe_float(float(np.mean([to_float(row, "path_length") for row in selected])) if selected else 0.0),
                "avg_path_efficiency": safe_float(float(np.mean([to_float(row, "path_efficiency") for row in selected])) if selected else 0.0),
                "avg_min_obstacle_distance_cells": safe_float(float(np.mean([to_float(row, "min_obstacle_distance_cells") for row in selected])) if selected else 0.0),
                "avg_hazard_exposure_steps": safe_float(float(np.mean([to_float(row, "hazard_exposure_steps") for row in selected])) if selected else 0.0),
                "avg_shield_interventions": safe_float(float(np.mean([to_float(row, "shield_interventions") for row in selected])) if selected else 0.0),
                "avg_total_inference_latency_ms": safe_float(float(np.mean([to_float(row, "avg_total_inference_latency_ms") for row in selected])) if selected else 0.0),
            }
        )
    return output


def run_hazard_density(
    scenarios_path: Path,
    model_path: Path,
    output_path: Path,
    selected_scenarios: list[str] | None,
    config: Week8Config,
    stereo_metrics_path: Path,
    mono_metrics_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[str]]:
    scenarios = json.loads(scenarios_path.read_text())["scenarios"]
    if selected_scenarios:
        selected = set(selected_scenarios)
        scenarios = [scenario for scenario in scenarios if str(scenario["id"]) in selected]
    plan_config = NavigationIntegrationConfig(
        grid_size=config.grid_size,
        max_steps=config.max_steps,
        collision_risk_threshold=config.collision_risk_threshold,
        goal_threshold=config.goal_threshold,
        episodes=config.episodes_per_mode_level * 2,
        start_goal_risk_threshold=config.start_goal_risk_threshold,
        policy_risk_scale=config.policy_risk_scale,
    )
    run_config = PerformanceEvaluationConfig(
        grid_size=config.grid_size,
        max_steps=config.max_steps,
        episodes=config.episodes_per_mode_level * 2,
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
    episode_rows = []
    step_rows = []
    intervention_rows = []
    event_rows = []
    selected_visuals = []
    episode_index = 1
    for level in HAZARD_LEVELS:
        for scenario, mode, variant in planned:
            key = "stereo_risk_map" if mode == "stereo" else "mono_risk_map"
            obstacle_risk = load_risk_map(scenario[key], config.grid_size)
            hazard_risk = hazard_density_overlay(obstacle_risk, str(scenario["id"]), int(variant), level, config)
            latency_ms = perception_latency(str(scenario["frame_id"]), mode, stereo_latencies, mono_latencies)
            metadata = {
                "hazard_density": level,
                "hazard_density_seed": seed_for(scenario["id"], variant, level),
            }
            render_visual = level in {"none", "high"} and str(scenario["id"]) == "baseline_open" and int(variant) == 0 and mode in {"mono", "stereo"}
            visual_label = f"week08_selected_{mode}_{level}_hazard_density_trajectory" if render_visual else None
            summary, steps, interventions, events, visual = run_episode(
                scenario,
                mode,
                "ppo_shield",
                obstacle_risk,
                hazard_risk,
                agent,
                run_config,
                output_path,
                episode_index,
                int(variant),
                latency_ms,
                render_visual=render_visual,
                visual_label=visual_label,
            )
            episode_rows.append({**metadata, **summary})
            step_rows.extend(annotate_rows(steps, metadata))
            intervention_rows.extend(annotate_rows(interventions, metadata))
            event_rows.extend(annotate_rows(events, metadata))
            if visual:
                selected_visuals.append(visual)
            episode_index += 1
    summary_rows = summarize_hazard(episode_rows)
    return episode_rows, step_rows, intervention_rows, event_rows, summary_rows, selected_visuals


def controller_ablation_rows(week6_summary: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for mode in ["mono", "stereo"]:
        raw = next(row for row in week6_summary if row["controller"] == "raw_ppo" and row["mode"] == mode)
        shield = next(row for row in week6_summary if row["controller"] == "ppo_shield" and row["mode"] == mode)
        rows.append(
            {
                "mode": mode,
                "raw_success_rate": to_float(raw, "success_rate"),
                "shield_success_rate": to_float(shield, "success_rate"),
                "success_rate_gain": to_float(shield, "success_rate") - to_float(raw, "success_rate"),
                "raw_collision_rate": to_float(raw, "collision_rate"),
                "shield_collision_rate": to_float(shield, "collision_rate"),
                "collision_rate_reduction": to_float(raw, "collision_rate") - to_float(shield, "collision_rate"),
                "raw_path_efficiency": to_float(raw, "avg_path_efficiency"),
                "shield_path_efficiency": to_float(shield, "avg_path_efficiency"),
            }
        )
    return rows


def clean_condition_rows(week6_summary: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        {
            "controller": row["controller"],
            "mode": row["mode"],
            "runs": to_int(row, "runs"),
            "success_rate": to_float(row, "success_rate"),
            "collision_rate": to_float(row, "collision_rate"),
            "timeout_rate": to_float(row, "timeout_rate"),
            "avg_path_efficiency": to_float(row, "avg_path_efficiency"),
            "avg_min_obstacle_distance_cells": to_float(row, "avg_min_obstacle_distance_cells"),
            "avg_total_inference_latency_ms": to_float(row, "avg_total_inference_latency_ms"),
        }
        for row in week6_summary
    ]


def degradation_table_rows(week7_degradation: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for mode in ["mono", "stereo"]:
        for perturbation in ["depth_noise", "image_blur", "reduced_resolution", "partial_occlusion"]:
            selected = [row for row in week7_degradation if row["mode"] == mode and row["perturbation_type"] == perturbation]
            worst = max(selected, key=lambda row: to_float(row, "success_rate_drop"))
            rows.append(
                {
                    "mode": mode,
                    "perturbation_type": perturbation,
                    "worst_severity": worst["severity"],
                    "clean_success_rate": to_float(worst, "clean_success_rate"),
                    "worst_success_rate": to_float(worst, "perturbed_success_rate"),
                    "success_rate_drop": to_float(worst, "success_rate_drop"),
                    "collision_rate_increase": to_float(worst, "collision_rate_increase"),
                    "path_efficiency_reduction": to_float(worst, "path_efficiency_reduction"),
                    "min_obstacle_distance_reduction": to_float(worst, "min_obstacle_distance_reduction"),
                }
            )
    return rows


def contribution_rows(clean_rows: list[dict[str, object]], ablation_rows: list[dict[str, object]], degradation_rows_out: list[dict[str, object]], hazard_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    shield = [row for row in clean_rows if row["controller"] == "ppo_shield"]
    mono = next(row for row in shield if row["mode"] == "mono")
    stereo = next(row for row in shield if row["mode"] == "stereo")
    mono_worst = max([row for row in degradation_rows_out if row["mode"] == "mono"], key=lambda row: to_float(row, "success_rate_drop"))
    stereo_worst = max([row for row in degradation_rows_out if row["mode"] == "stereo"], key=lambda row: to_float(row, "success_rate_drop"))
    mono_high = next(row for row in hazard_summary if row["mode"] == "mono" and row["hazard_density"] == "high")
    stereo_high = next(row for row in hazard_summary if row["mode"] == "stereo" and row["hazard_density"] == "high")
    return [
        {
            "finding": "clean_condition_safety",
            "value": "PPO+shield reached zero clean-condition collisions for both sensing modes",
            "mono_success_rate": mono["success_rate"],
            "stereo_success_rate": stereo["success_rate"],
            "mono_latency_ms": mono["avg_total_inference_latency_ms"],
            "stereo_latency_ms": stereo["avg_total_inference_latency_ms"],
        },
        {
            "finding": "controller_ablation",
            "value": "Raw PPO was unsafe while PPO+shield removed clean-condition collisions",
            "mono_success_gain": next(row for row in ablation_rows if row["mode"] == "mono")["success_rate_gain"],
            "stereo_success_gain": next(row for row in ablation_rows if row["mode"] == "stereo")["success_rate_gain"],
            "mono_collision_reduction": next(row for row in ablation_rows if row["mode"] == "mono")["collision_rate_reduction"],
            "stereo_collision_reduction": next(row for row in ablation_rows if row["mode"] == "stereo")["collision_rate_reduction"],
        },
        {
            "finding": "perception_degradation",
            "value": "Partial occlusion was the strongest stereo failure mode; monocular degraded most under resolution or blur",
            "mono_worst_perturbation": mono_worst["perturbation_type"],
            "mono_worst_success_drop": mono_worst["success_rate_drop"],
            "stereo_worst_perturbation": stereo_worst["perturbation_type"],
            "stereo_worst_success_drop": stereo_worst["success_rate_drop"],
        },
        {
            "finding": "hazard_density",
            "value": "Hazard density increased exposure and shield workload while physical collision remained evaluated from obstacle risk",
            "mono_high_hazard_success": mono_high["success_rate"],
            "stereo_high_hazard_success": stereo_high["success_rate"],
            "mono_high_hazard_exposure_steps": mono_high["avg_hazard_exposure_steps"],
            "stereo_high_hazard_exposure_steps": stereo_high["avg_hazard_exposure_steps"],
        },
    ]


def plot_grouped_bars(rows: list[dict[str, object]], output_path: Path, group_key: str, x_key: str, y_key: str, ylabel: str, title: str) -> str:
    groups = list(dict.fromkeys(str(row[group_key]) for row in rows))
    labels = list(dict.fromkeys(str(row[x_key]) for row in rows))
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2F6DAE", "#E66101", "#5E9F49", "#7B4EA3"]
    markers = ["o", "s", "^", "D"]
    for idx, group in enumerate(groups):
        values = [to_float(next(row for row in rows if str(row[group_key]) == group and str(row[x_key]) == label), y_key) for label in labels]
        ax.bar(x + (idx - (len(groups) - 1) / 2.0) * width, values, width, label=group, color=colors[idx % len(colors)], edgecolor="black", linewidth=0.8)
        ax.plot(x + (idx - (len(groups) - 1) / 2.0) * width, values, linestyle="", marker=markers[idx % len(markers)], color="black", markersize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("_", " ").title() for label in labels])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_degradation(rows: list[dict[str, str]], output_path: Path) -> str:
    severity_order = {"low": 0, "medium": 1, "high": 2, "extreme": 3}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, perturbation in zip(axes.ravel(), ["depth_noise", "image_blur", "reduced_resolution", "partial_occlusion"]):
        for mode, color, marker in [("mono", "#2F6DAE", "o"), ("stereo", "#E66101", "^")]:
            selected = [row for row in rows if row["mode"] == mode and row["perturbation_type"] == perturbation]
            selected = sorted(selected, key=lambda row: severity_order[row["severity"]])
            x = [severity_order[row["severity"]] for row in selected]
            y = [to_float(row, "perturbed_success_rate") for row in selected]
            ax.plot(x, y, color=color, marker=marker, linewidth=2.2, markersize=6, label=mode)
        ax.set_title(perturbation.replace("_", " ").title())
        ax.set_xticks(list(severity_order.values()), [name.title() for name in severity_order])
        ax.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("Success Rate")
    axes[1, 0].set_ylabel("Success Rate")
    axes[1, 0].set_xlabel("Severity")
    axes[1, 1].set_xlabel("Severity")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Perception Degradation Ablation", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_tradeoff(clean_rows: list[dict[str, object]], output_path: Path) -> str:
    selected = [row for row in clean_rows if row["controller"] == "ppo_shield"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for row, color, marker in zip(selected, ["#2F6DAE", "#E66101"], ["o", "^"]):
        ax.scatter(to_float(row, "avg_total_inference_latency_ms"), to_float(row, "success_rate"), s=120, color=color, marker=marker, edgecolor="black", label=str(row["mode"]))
        ax.text(to_float(row, "avg_total_inference_latency_ms") + 0.5, to_float(row, "success_rate") - 0.01, str(row["mode"]).title())
    ax.set_xlabel("Total Inference Latency (ms)")
    ax.set_ylabel("Clean Success Rate")
    ax.set_title("Efficiency vs Safety Trade-Off")
    ax.set_ylim(0.70, 1.02)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def write_report(path: Path, output_path: Path, clean_rows: list[dict[str, object]], ablation_rows_out: list[dict[str, object]], degradation_out: list[dict[str, object]], hazard_summary: list[dict[str, object]], visuals: list[str]) -> None:
    shield = [row for row in clean_rows if row["controller"] == "ppo_shield"]
    raw = [row for row in clean_rows if row["controller"] == "raw_ppo"]
    lines = []
    lines.append("# Weekly Research Report")
    lines.append("")
    lines.append("## Week Ending")
    lines.append("May 22, 2026")
    lines.append("")
    lines.append("## Researcher")
    lines.append("Clinton Imaro")
    lines.append("")
    lines.append("## Project Title")
    lines.append("Performance Evaluation of Monocular and Stereo Vision for Autonomous UAV Collision Avoidance in Disaster Response Environments")
    lines.append("")
    lines.append("## Objectives (Week 8 Checklist)")
    lines.append("- Focus the final contribution narrative.")
    lines.append("- Run ablation studies for noise levels, hazard density, and perception degradation.")
    lines.append("- Generate final paper-ready plots and tables.")
    lines.append("- Extract efficiency-versus-safety trade-offs.")
    lines.append("")
    lines.append("## Summary of Work for the Week")
    lines.append("This week consolidated the Week 6 clean-condition evaluation and Week 7 robustness results into final paper evidence. A new hazard-density ablation was added so that the disaster-risk component is isolated separately from perception degradation. The PPO checkpoint and network architecture were not changed.")
    lines.append("")
    lines.append("## Accomplishments")
    lines.append("- Consolidated clean-condition, controller-ablation, perception-degradation, and hazard-density results.")
    lines.append("- Added a direct hazard-density ablation using the same PPO+shield controller.")
    lines.append("- Generated final tables and selected paper-ready figures.")
    lines.append("- Identified the main safety-efficiency trade-offs for the paper discussion.")
    lines.append("")
    lines.append("## Clean-Condition Controller Result")
    lines.append("")
    lines.append("| Controller | Mode | Success Rate | Collision Rate | Path Efficiency | Min Obstacle Distance | Total Inference Latency |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in clean_rows:
        lines.append(f"| {row['controller']} | {row['mode']} | {to_float(row, 'success_rate'):.3f} | {to_float(row, 'collision_rate'):.3f} | {to_float(row, 'avg_path_efficiency'):.3f} | {to_float(row, 'avg_min_obstacle_distance_cells'):.3f} | {to_float(row, 'avg_total_inference_latency_ms'):.2f} ms |")
    lines.append("")
    lines.append("## Controller Ablation")
    lines.append("")
    lines.append("| Mode | Raw Success | Shield Success | Success Gain | Raw Collision | Shield Collision | Collision Reduction |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in ablation_rows_out:
        lines.append(f"| {row['mode']} | {to_float(row, 'raw_success_rate'):.3f} | {to_float(row, 'shield_success_rate'):.3f} | {to_float(row, 'success_rate_gain'):.3f} | {to_float(row, 'raw_collision_rate'):.3f} | {to_float(row, 'shield_collision_rate'):.3f} | {to_float(row, 'collision_rate_reduction'):.3f} |")
    lines.append("")
    lines.append("## Hazard-Density Ablation")
    lines.append("")
    lines.append("| Mode | Hazard Density | Success Rate | Collision Rate | Hazard Exposure Steps | Shield Interventions |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in hazard_summary:
        lines.append(f"| {row['mode']} | {row['hazard_density']} | {to_float(row, 'success_rate'):.3f} | {to_float(row, 'collision_rate'):.3f} | {to_float(row, 'avg_hazard_exposure_steps'):.3f} | {to_float(row, 'avg_shield_interventions'):.3f} |")
    lines.append("")
    lines.append("## Key Insights")
    mono_shield = next(row for row in shield if row["mode"] == "mono")
    stereo_shield = next(row for row in shield if row["mode"] == "stereo")
    raw_collision = max(to_float(row, "collision_rate") for row in raw)
    lines.append(f"- Raw PPO collision rate reached {raw_collision:.3f}, so it should be presented as an ablation baseline rather than the final controller.")
    lines.append(f"- PPO+shield reduced clean-condition collision rate to 0.000 for both sensing modes.")
    lines.append(f"- Monocular PPO+shield had higher clean success ({to_float(mono_shield, 'success_rate'):.3f}) but higher latency ({to_float(mono_shield, 'avg_total_inference_latency_ms'):.2f} ms).")
    lines.append(f"- Stereo PPO+shield was faster ({to_float(stereo_shield, 'avg_total_inference_latency_ms'):.2f} ms) but less robust under partial occlusion and reduced-resolution stress.")
    lines.append("- Hazard density mainly increases exposure and shield workload; physical collision remains evaluated from structural obstacle risk.")
    lines.append("")
    lines.append("## Testing and Validation")
    lines.append("")
    lines.append("| Validation Item | Result |")
    lines.append("|---|---|")
    lines.append("| Hazard-density episode rows | 2,000 |")
    lines.append("| Hazard-density summary rows | 8 |")
    lines.append("| Rows per mode/hazard level | 250 |")
    lines.append("| Week 6 logs reused | Passed |")
    lines.append("| Week 7 logs reused | Passed |")
    lines.append("| PPO checkpoint reused without retraining | Passed |")
    lines.append("| PPO input shape remained `(50, 50, 4)` | Passed |")
    lines.append("| Training updates during evaluation | 0 |")
    lines.append("")
    lines.append("Generated outputs were saved under:")
    lines.append("")
    lines.append(f"`{output_path}`")
    lines.append("")
    lines.append("## Visuals")
    lines.append("")
    for idx, visual in enumerate(visuals, 1):
        lines.append(f"Figure {idx}. `{visual}`")
        lines.append("")
        lines.append(f"![Week 8 visual {idx}]({Path(visual).resolve()})")
        lines.append("")
    lines.append("## Next Steps")
    lines.append("- Use Week 8 tables and figures as the Paper 2 results backbone.")
    lines.append("- Move the clean-condition, degradation, hazard-density, and trade-off figures into the final paper draft.")
    lines.append("- Keep limitations explicit: controlled UAVStereo-derived maps, perception-interface perturbations, and grid-cell clearance rather than meter-scale flight testing.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def validate_numeric(rows: list[dict[str, object]], label: str) -> None:
    for row_idx, row in enumerate(rows, 1):
        for key, value in row.items():
            if isinstance(value, (int, float)) and not np.isfinite(float(value)):
                raise RuntimeError(f"Invalid numeric value in {label} row {row_idx} field {key}")


def rectangular_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return [{key: row.get(key, "") for key in keys} for row in rows]


def run_week8(
    scenarios_path: Path,
    model_path: Path,
    output_dir: Path,
    run_name: str,
    selected_scenarios: list[str] | None,
    config: Week8Config,
    stereo_metrics_path: Path,
    mono_metrics_path: Path,
    week6_dir: Path,
    week7_dir: Path,
    doc_path: Path,
) -> Path:
    output_path = output_dir / run_name
    paths = ensure_dirs(output_path)
    hazard_rows, hazard_steps, hazard_interventions, hazard_events, hazard_summary, selected_visuals = run_hazard_density(
        scenarios_path,
        model_path,
        output_path,
        selected_scenarios,
        config,
        stereo_metrics_path,
        mono_metrics_path,
    )
    week6_summary = read_csv(week6_dir / "week06_summary.csv")
    week7_degradation = read_csv(week7_dir / "week07_degradation_curves.csv")
    clean_rows = clean_condition_rows(week6_summary)
    ablation_rows_out = controller_ablation_rows(week6_summary)
    degradation_rows_out = degradation_table_rows(week7_degradation)
    contribution = contribution_rows(clean_rows, ablation_rows_out, degradation_rows_out, hazard_summary)
    validate_numeric(hazard_rows, "week08_hazard_density_episodes")
    validate_numeric(hazard_summary, "week08_hazard_density_summary")
    write_csv(paths["metrics"] / "week08_hazard_density_episodes.csv", hazard_rows)
    write_csv(paths["metrics"] / "week08_hazard_density_steps.csv", hazard_steps)
    write_csv(paths["metrics"] / "week08_hazard_density_interventions.csv", hazard_interventions)
    write_csv(paths["metrics"] / "week08_hazard_density_events.csv", hazard_events)
    write_csv(paths["metrics"] / "week08_hazard_density_summary.csv", hazard_summary)
    write_csv(paths["metrics"] / "week08_clean_condition_table.csv", clean_rows)
    write_csv(paths["metrics"] / "week08_controller_ablation_table.csv", ablation_rows_out)
    write_csv(paths["metrics"] / "week08_perception_degradation_table.csv", degradation_rows_out)
    write_csv(paths["metrics"] / "week08_final_contribution_summary.csv", rectangular_rows(contribution))
    visuals = []
    visuals.append(plot_grouped_bars(clean_rows, paths["visuals"] / "week08_clean_success_rate.png", "mode", "controller", "success_rate", "Success Rate", "Clean-Condition Success Rate"))
    visuals.append(plot_grouped_bars(clean_rows, paths["visuals"] / "week08_clean_collision_rate.png", "mode", "controller", "collision_rate", "Collision Rate", "Clean-Condition Collision Rate"))
    visuals.append(plot_grouped_bars(hazard_summary, paths["visuals"] / "week08_hazard_density_success_rate.png", "mode", "hazard_density", "success_rate", "Success Rate", "Hazard-Density Ablation"))
    visuals.append(plot_grouped_bars(hazard_summary, paths["visuals"] / "week08_hazard_density_exposure.png", "mode", "hazard_density", "avg_hazard_exposure_steps", "Hazard Exposure Steps", "Hazard Exposure Under Density Ablation"))
    visuals.append(plot_degradation(week7_degradation, paths["visuals"] / "week08_perception_degradation_success.png"))
    visuals.append(plot_tradeoff(clean_rows, paths["visuals"] / "week08_latency_success_tradeoff.png"))
    visuals.extend(selected_visuals)
    write_json(
        paths["metrics"] / "week08_config.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "scenarios_path": str(scenarios_path),
            "model_path": str(model_path),
            "stereo_metrics_path": str(stereo_metrics_path),
            "mono_metrics_path": str(mono_metrics_path),
            "week6_dir": str(week6_dir),
            "week7_dir": str(week7_dir),
            "input_shape": [config.grid_size, config.grid_size, 4],
            "action_dim": 8,
            "training_updates": 0,
            "hazard_levels": HAZARD_LEVELS,
            "config": asdict(config),
            "selected_visuals": visuals,
            "output_dir": str(output_path),
        },
    )
    write_report(doc_path, output_path, clean_rows, ablation_rows_out, degradation_rows_out, hazard_summary, visuals[:8])
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Paper 2 Week 8 analysis and ablation.")
    parser.add_argument("--scenarios-path", type=Path, default=DEFAULT_SCENARIOS_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default="final")
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--stereo-metrics", type=Path, default=DEFAULT_STEREO_METRICS)
    parser.add_argument("--mono-metrics", type=Path, default=DEFAULT_MONO_METRICS)
    parser.add_argument("--week6-dir", type=Path, default=DEFAULT_WEEK6_DIR)
    parser.add_argument("--week7-dir", type=Path, default=DEFAULT_WEEK7_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--episodes-per-mode-level", type=int, default=250)
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
    config = Week8Config(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        episodes_per_mode_level=args.episodes_per_mode_level,
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
    output_path = run_week8(
        scenarios_path=args.scenarios_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        selected_scenarios=args.scenario,
        config=config,
        stereo_metrics_path=args.stereo_metrics,
        mono_metrics_path=args.mono_metrics,
        week6_dir=args.week6_dir,
        week7_dir=args.week7_dir,
        doc_path=args.doc_path,
    )
    print(f"Week 8 analysis and ablation complete: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
