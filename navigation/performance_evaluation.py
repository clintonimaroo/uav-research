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
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collision_avoidance_validation import (
    CollisionAvoidanceConfig,
    candidate_position,
    event_rows_for_step,
    hazard_overlay,
    safe_float,
    shield_action,
)
from navigation_integration import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SCENARIOS_PATH,
    PPONavigationAgent,
    NavigationIntegrationConfig,
    deterministic_action,
    episode_plan,
    load_risk_map,
    observation,
    path_length,
    policy_risk,
    render_trajectory,
    write_csv,
    write_json,
)


DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "week06_performance_evaluation"
DEFAULT_STEREO_METRICS = SCRIPT_DIR / "results" / "week01_stereo_depth" / "final" / "metrics" / "week01_stereo_metrics.csv"
DEFAULT_MONO_METRICS = SCRIPT_DIR / "results" / "week02_monocular_depth" / "final" / "metrics" / "week02_mono_metrics.csv"


@dataclass(frozen=True)
class PerformanceEvaluationConfig:
    grid_size: int = 50
    max_steps: int = 200
    episodes: int = 500
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


def read_latency_metrics(path: Path) -> dict[str, float]:
    values = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            values[str(row["frame_id"])] = float(row["inference_latency_ms"])
    values["__mean__"] = float(np.mean(list(values.values()))) if values else 0.0
    return values


def perception_latency(frame_id: str, mode: str, stereo_values: dict[str, float], mono_values: dict[str, float]) -> float:
    values = stereo_values if mode == "stereo" else mono_values
    return float(values.get(str(frame_id), values.get("__mean__", 0.0)))


def clearance_field(obstacle_risk: np.ndarray, threshold: float) -> np.ndarray:
    obstacle = obstacle_risk >= threshold
    if not np.any(obstacle):
        return np.full(obstacle_risk.shape, float(np.linalg.norm(obstacle_risk.shape)), dtype=np.float32)
    safe = np.logical_not(obstacle).astype(np.uint8)
    return cv2.distanceTransform(safe, cv2.DIST_L2, 5).astype(np.float32)


def should_render(controller: str, mode: str, scenario_id: str, variant: int) -> bool:
    return scenario_id == "baseline_open" and variant == 0 and controller in {"raw_ppo", "ppo_shield"} and mode in {"stereo", "mono"}


def run_episode(
    scenario: dict[str, object],
    mode: str,
    controller: str,
    obstacle_risk: np.ndarray,
    hazard_risk: np.ndarray,
    agent: PPONavigationAgent,
    config: PerformanceEvaluationConfig,
    output_dir: Path,
    episode_index: int,
    episode_variant: int,
    perception_latency_ms: float,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], str]:
    start = np.array(scenario["start_cell"], dtype=np.int32)
    goal = np.array(scenario["goal_cell"], dtype=np.int32)
    position = start.copy()
    path = [position.copy()]
    combined_risk = np.maximum(obstacle_risk, hazard_risk)
    policy_config = NavigationIntegrationConfig(
        grid_size=config.grid_size,
        max_steps=config.max_steps,
        collision_risk_threshold=config.collision_risk_threshold,
        goal_threshold=config.goal_threshold,
        episodes=config.episodes,
        start_goal_risk_threshold=config.start_goal_risk_threshold,
        policy_risk_scale=config.policy_risk_scale,
    )
    shield_config = CollisionAvoidanceConfig(
        grid_size=config.grid_size,
        max_steps=config.max_steps,
        episodes=config.episodes,
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
    input_risk = policy_risk(combined_risk, policy_config)
    clearance = clearance_field(obstacle_risk, config.collision_risk_threshold)
    start_distance = float(np.linalg.norm(goal - start))
    reason = "max_steps"
    success = False
    collided = False
    step_rows = []
    intervention_rows = []
    event_rows = []
    recent_positions = [tuple(position.tolist())]
    ppo_latencies = []
    shield_latencies = []
    start_time = time.perf_counter()
    for step in range(1, config.max_steps + 1):
        state = observation(input_risk, position, goal, policy_config)
        policy_start = time.perf_counter_ns()
        ppo_action, confidence = deterministic_action(agent, state)
        ppo_latency_ms = (time.perf_counter_ns() - policy_start) / 1_000_000.0
        selected_action = ppo_action
        overridden = False
        shield_reason = "not_applied"
        proposed_score = 0.0
        selected_score = 0.0
        shield_latency_ms = 0.0
        if controller == "ppo_shield":
            shield_start = time.perf_counter_ns()
            selected_action, overridden, shield_reason, proposed_score, selected_score = shield_action(
                ppo_action,
                position,
                goal,
                obstacle_risk,
                hazard_risk,
                recent_positions,
                shield_config,
            )
            shield_latency_ms = (time.perf_counter_ns() - shield_start) / 1_000_000.0
        ppo_latencies.append(ppo_latency_ms)
        shield_latencies.append(shield_latency_ms)
        previous_distance = float(np.linalg.norm(goal - position))
        position = candidate_position(position, selected_action, config.grid_size)
        recent_positions.append(tuple(position.tolist()))
        path.append(position.copy())
        obstacle_value = float(obstacle_risk[int(position[0]), int(position[1])])
        hazard_value = float(hazard_risk[int(position[0]), int(position[1])])
        combined_value = float(combined_risk[int(position[0]), int(position[1])])
        clearance_cells = float(clearance[int(position[0]), int(position[1])])
        final_distance = float(np.linalg.norm(goal - position))
        progress = previous_distance - final_distance
        if final_distance < config.goal_threshold:
            success = True
            reason = "goal_reached"
        elif obstacle_value >= config.collision_risk_threshold:
            collided = True
            reason = "collision_risk"
        total_latency_ms = perception_latency_ms + ppo_latency_ms + shield_latency_ms
        row_base = {
            "episode": episode_index,
            "episode_variant": episode_variant,
            "controller": controller,
            "scenario_id": scenario["id"],
            "scenario_name": scenario["name"],
            "mode": mode,
            "frame_id": scenario["frame_id"],
            "step": step,
            "row": int(position[0]),
            "col": int(position[1]),
        }
        step_rows.append(
            {
                **row_base,
                "ppo_action": ppo_action,
                "selected_action": selected_action,
                "action_confidence": confidence,
                "shield_overrode": int(overridden),
                "shield_reason": shield_reason,
                "obstacle_risk": obstacle_value,
                "hazard_risk": hazard_value,
                "combined_risk": combined_value,
                "obstacle_clearance_cells": clearance_cells,
                "distance_to_goal": final_distance,
                "progress": progress,
                "perception_latency_ms": perception_latency_ms,
                "ppo_decision_latency_ms": ppo_latency_ms,
                "shield_decision_latency_ms": shield_latency_ms,
                "total_inference_latency_ms": total_latency_ms,
                "done": int(success or collided),
                "done_reason": reason if success or collided else "",
            }
        )
        if overridden:
            intervention_rows.append(
                {
                    **row_base,
                    "ppo_action": ppo_action,
                    "selected_action": selected_action,
                    "reason": shield_reason,
                    "proposed_score": proposed_score,
                    "selected_score": selected_score,
                    "obstacle_risk": obstacle_value,
                    "hazard_risk": hazard_value,
                    "distance_to_goal": final_distance,
                    "shield_decision_latency_ms": shield_latency_ms,
                }
            )
        event_rows.extend(event_rows_for_step(row_base, obstacle_value, hazard_value, overridden, shield_reason, shield_config))
        if success or collided:
            break
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    length = path_length(path)
    final_distance = float(np.linalg.norm(goal - position))
    path_obstacle_values = [float(obstacle_risk[int(p[0]), int(p[1])]) for p in path]
    path_hazard_values = [float(hazard_risk[int(p[0]), int(p[1])]) for p in path]
    path_clearance_values = [float(clearance[int(p[0]), int(p[1])]) for p in path]
    interventions = sum(int(row["shield_overrode"]) for row in step_rows)
    progress_efficiency = max(0.0, start_distance - final_distance) / max(length, 1e-6)
    selected_visual = ""
    if should_render(controller, mode, str(scenario["id"]), episode_variant):
        selected_visual = str(output_dir / "visuals" / f"week06_selected_{controller}_{mode}_trajectory.png")
        render_trajectory(combined_risk, path, start, goal, Path(selected_visual))
    summary = {
        "episode": episode_index,
        "episode_variant": episode_variant,
        "controller": controller,
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "mode": mode,
        "frame_id": scenario["frame_id"],
        "start_row": int(start[0]),
        "start_col": int(start[1]),
        "goal_row": int(goal[0]),
        "goal_col": int(goal[1]),
        "start_obstacle_risk": float(obstacle_risk[int(start[0]), int(start[1])]),
        "goal_obstacle_risk": float(obstacle_risk[int(goal[0]), int(goal[1])]),
        "success": int(success),
        "collided": int(collided),
        "done_reason": reason,
        "steps": len(path) - 1,
        "path_length": length,
        "start_distance": start_distance,
        "final_distance": final_distance,
        "path_efficiency": float(start_distance / max(length, 1e-6)) if success else 0.0,
        "progress_efficiency": progress_efficiency,
        "min_obstacle_distance_cells": safe_float(float(np.min(path_clearance_values))),
        "mean_obstacle_risk": safe_float(float(np.mean(path_obstacle_values))),
        "max_obstacle_risk": safe_float(float(np.max(path_obstacle_values))),
        "mean_hazard_risk": safe_float(float(np.mean(path_hazard_values))),
        "max_hazard_risk": safe_float(float(np.max(path_hazard_values))),
        "hazard_exposure_steps": sum(1 for value in path_hazard_values if value >= config.hazard_threshold),
        "shield_interventions": interventions,
        "shield_intervention_rate": interventions / max(len(path) - 1, 1),
        "perception_latency_ms": perception_latency_ms,
        "avg_ppo_decision_latency_ms": safe_float(float(np.mean(ppo_latencies))),
        "p50_ppo_decision_latency_ms": safe_float(float(np.percentile(ppo_latencies, 50))),
        "p95_ppo_decision_latency_ms": safe_float(float(np.percentile(ppo_latencies, 95))),
        "avg_shield_decision_latency_ms": safe_float(float(np.mean(shield_latencies))),
        "p50_shield_decision_latency_ms": safe_float(float(np.percentile(shield_latencies, 50))),
        "p95_shield_decision_latency_ms": safe_float(float(np.percentile(shield_latencies, 95))),
        "avg_total_inference_latency_ms": perception_latency_ms + safe_float(float(np.mean(ppo_latencies))) + safe_float(float(np.mean(shield_latencies))),
        "elapsed_ms": elapsed_ms,
        "selected_visual": selected_visual,
    }
    return summary, step_rows, intervention_rows, event_rows, selected_visual


def grouped_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    keys = sorted({(row["controller"], row["mode"]) for row in rows})
    for controller, mode in keys:
        selected = [row for row in rows if row["controller"] == controller and row["mode"] == mode]
        runs = len(selected)
        successes = sum(int(row["success"]) for row in selected)
        collisions = sum(int(row["collided"]) for row in selected)
        timeouts = sum(1 for row in selected if row["done_reason"] == "max_steps")
        output.append(
            {
                "controller": controller,
                "mode": mode,
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
                "avg_perception_latency_ms": safe_float(float(np.mean([float(row["perception_latency_ms"]) for row in selected]))),
                "avg_ppo_decision_latency_ms": safe_float(float(np.mean([float(row["avg_ppo_decision_latency_ms"]) for row in selected]))),
                "avg_shield_decision_latency_ms": safe_float(float(np.mean([float(row["avg_shield_decision_latency_ms"]) for row in selected]))),
                "avg_total_inference_latency_ms": safe_float(float(np.mean([float(row["avg_total_inference_latency_ms"]) for row in selected]))),
            }
        )
    return output


def latency_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "controller": row["controller"],
            "mode": row["mode"],
            "perception_latency_ms": row["avg_perception_latency_ms"],
            "ppo_decision_latency_ms": row["avg_ppo_decision_latency_ms"],
            "shield_decision_latency_ms": row["avg_shield_decision_latency_ms"],
            "total_inference_latency_ms": row["avg_total_inference_latency_ms"],
        }
        for row in summary_rows
    ]


def run_week6(
    scenarios_path: Path,
    model_path: Path,
    output_dir: Path,
    run_name: str,
    selected_scenarios: list[str] | None,
    config: PerformanceEvaluationConfig,
    stereo_metrics_path: Path,
    mono_metrics_path: Path,
) -> Path:
    output_path = output_dir / run_name
    ensure_dirs(output_path)
    scenarios = json.loads(scenarios_path.read_text())["scenarios"]
    if selected_scenarios:
        selected = set(selected_scenarios)
        scenarios = [scenario for scenario in scenarios if str(scenario["id"]) in selected]
    plan_config = NavigationIntegrationConfig(
        grid_size=config.grid_size,
        max_steps=config.max_steps,
        collision_risk_threshold=config.collision_risk_threshold,
        goal_threshold=config.goal_threshold,
        episodes=config.episodes,
        start_goal_risk_threshold=config.start_goal_risk_threshold,
        policy_risk_scale=config.policy_risk_scale,
    )
    planned = episode_plan(scenarios, plan_config)
    agent = PPONavigationAgent((config.grid_size, config.grid_size, 4), 8)
    if not agent.load_model(str(model_path)):
        raise RuntimeError(f"PPO checkpoint could not be loaded: {model_path}")
    stereo_latencies = read_latency_metrics(stereo_metrics_path)
    mono_latencies = read_latency_metrics(mono_metrics_path)
    run_rows = []
    step_rows = []
    intervention_rows = []
    event_rows = []
    selected_visuals = []
    episode_index = 1
    for controller in ["raw_ppo", "ppo_shield"]:
        for scenario, mode, variant in planned:
            key = "stereo_risk_map" if mode == "stereo" else "mono_risk_map"
            obstacle_risk = load_risk_map(scenario[key], config.grid_size)
            hazard_risk = hazard_overlay(obstacle_risk, str(scenario["id"]), int(variant), config)
            latency_ms = perception_latency(str(scenario["frame_id"]), mode, stereo_latencies, mono_latencies)
            summary, steps, interventions, events, visual = run_episode(
                scenario,
                mode,
                controller,
                obstacle_risk,
                hazard_risk,
                agent,
                config,
                output_path,
                episode_index,
                int(variant),
                latency_ms,
            )
            run_rows.append(summary)
            step_rows.extend(steps)
            intervention_rows.extend(interventions)
            event_rows.extend(events)
            if visual:
                selected_visuals.append(visual)
            episode_index += 1
    summary_rows = grouped_summary(run_rows)
    latency = latency_rows(summary_rows)
    write_csv(output_path / "metrics" / "week06_episode_results.csv", run_rows)
    write_csv(output_path / "metrics" / "week06_step_results.csv", step_rows)
    write_csv(output_path / "metrics" / "week06_interventions.csv", intervention_rows)
    write_csv(output_path / "metrics" / "week06_events.csv", event_rows)
    write_csv(output_path / "metrics" / "week06_summary.csv", summary_rows)
    write_csv(output_path / "metrics" / "week06_mono_stereo_comparison.csv", summary_rows)
    write_csv(output_path / "metrics" / "week06_latency_comparison.csv", latency)
    write_json(
        output_path / "metrics" / "week06_config.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "scenarios_path": str(scenarios_path),
            "model_path": str(model_path),
            "stereo_metrics_path": str(stereo_metrics_path),
            "mono_metrics_path": str(mono_metrics_path),
            "input_shape": [config.grid_size, config.grid_size, 4],
            "action_dim": 8,
            "training_updates": 0,
            "controllers": ["raw_ppo", "ppo_shield"],
            "config": asdict(config),
            "selected_visuals": selected_visuals,
            "output_dir": str(output_path),
        },
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Paper 2 Week 6 performance evaluation.")
    parser.add_argument("--scenarios-path", type=Path, default=DEFAULT_SCENARIOS_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default="final")
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--stereo-metrics", type=Path, default=DEFAULT_STEREO_METRICS)
    parser.add_argument("--mono-metrics", type=Path, default=DEFAULT_MONO_METRICS)
    parser.add_argument("--episodes", type=int, default=500)
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
    config = PerformanceEvaluationConfig(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        episodes=args.episodes,
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
    output_path = run_week6(
        scenarios_path=args.scenarios_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        selected_scenarios=args.scenario,
        config=config,
        stereo_metrics_path=args.stereo_metrics,
        mono_metrics_path=args.mono_metrics,
    )
    print(f"Week 6 performance evaluation complete: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
