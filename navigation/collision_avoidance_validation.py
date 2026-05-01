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

from navigation_integration import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SCENARIOS_PATH,
    PPONavigationAgent,
    NavigationIntegrationConfig,
    action_delta,
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


DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "week05_collision_avoidance"


@dataclass(frozen=True)
class CollisionAvoidanceConfig:
    grid_size: int = 50
    max_steps: int = 200
    episodes: int = 50
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
        "trajectories": output_dir / "trajectories",
        "visuals": output_dir / "visuals",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def safe_float(value: float) -> float:
    if np.isfinite(value):
        return float(value)
    return 0.0


def hazard_overlay(obstacle_risk: np.ndarray, scenario_id: str, variant: int, config: CollisionAvoidanceConfig) -> np.ndarray:
    seed = sum(ord(ch) for ch in scenario_id) + 7919 * (variant + 1)
    rng = np.random.default_rng(seed)
    hazard = np.zeros_like(obstacle_risk, dtype=np.float32)
    high_risk = np.argwhere(obstacle_risk >= np.quantile(obstacle_risk, 0.88))
    low_risk = np.argwhere(obstacle_risk < config.start_goal_risk_threshold)
    centers = []
    for pool in [high_risk, low_risk]:
        if len(pool) == 0:
            continue
        count = 2 if pool is high_risk else 1
        for _ in range(count):
            idx = int(rng.integers(0, len(pool)))
            centers.append(pool[idx])
    rows, cols = np.indices(obstacle_risk.shape)
    for idx, center in enumerate(centers):
        radius = 3.0 + float(rng.uniform(0.0, 2.0))
        intensity = 0.65 + 0.1 * (idx % 3)
        dist_sq = (rows - int(center[0])) ** 2 + (cols - int(center[1])) ** 2
        hazard = np.maximum(hazard, intensity * np.exp(-dist_sq / max(2.0 * radius * radius, 1.0)))
    return np.clip(hazard, 0.0, 1.0).astype(np.float32)


def candidate_position(position: np.ndarray, action: int, grid_size: int) -> np.ndarray:
    return np.clip(position + action_delta(action), 0, grid_size - 1).astype(np.int32)


def immediate_cost(
    position: np.ndarray,
    next_position: np.ndarray,
    goal: np.ndarray,
    obstacle_risk: np.ndarray,
    hazard_risk: np.ndarray,
    recent_positions: list[tuple[int, int]],
    config: CollisionAvoidanceConfig,
) -> float:
    obstacle_value = float(obstacle_risk[int(next_position[0]), int(next_position[1])])
    hazard_value = float(hazard_risk[int(next_position[0]), int(next_position[1])])
    current_distance = float(np.linalg.norm(goal - position))
    next_distance = float(np.linalg.norm(goal - next_position))
    progress = current_distance - next_distance
    collision_cost = 10000.0 if obstacle_value >= config.collision_risk_threshold else 0.0
    caution_cost = 200.0 if max(obstacle_value, hazard_value) >= config.shield_caution_threshold else 0.0
    loop_cost = config.loop_penalty_weight if tuple(next_position.tolist()) in recent_positions[-8:] else 0.0
    distance_cost = 0.05 * next_distance
    risk_cost = config.risk_penalty_weight * (obstacle_value + 0.5 * hazard_value)
    return collision_cost + caution_cost + loop_cost + distance_cost + risk_cost - config.progress_weight * progress


def rollout_cost(
    first_action: int,
    position: np.ndarray,
    goal: np.ndarray,
    obstacle_risk: np.ndarray,
    hazard_risk: np.ndarray,
    recent_positions: list[tuple[int, int]],
    config: CollisionAvoidanceConfig,
) -> float:
    simulated = position.copy()
    total = 0.0
    for depth in range(config.lookahead_steps):
        if depth == 0:
            action = first_action
            next_position = candidate_position(simulated, action, config.grid_size)
            total += immediate_cost(simulated, next_position, goal, obstacle_risk, hazard_risk, recent_positions, config)
            simulated = next_position
            continue
        scores = []
        for action in range(8):
            next_position = candidate_position(simulated, action, config.grid_size)
            scores.append(immediate_cost(simulated, next_position, goal, obstacle_risk, hazard_risk, recent_positions, config))
        action = int(np.argmin(scores))
        next_position = candidate_position(simulated, action, config.grid_size)
        total += float(np.min(scores)) / float(depth + 1)
        simulated = next_position
    return total


def shield_action(
    ppo_action: int,
    position: np.ndarray,
    goal: np.ndarray,
    obstacle_risk: np.ndarray,
    hazard_risk: np.ndarray,
    recent_positions: list[tuple[int, int]],
    config: CollisionAvoidanceConfig,
) -> tuple[int, bool, str, float, float]:
    proposed = candidate_position(position, ppo_action, config.grid_size)
    proposed_obstacle = float(obstacle_risk[int(proposed[0]), int(proposed[1])])
    proposed_hazard = float(hazard_risk[int(proposed[0]), int(proposed[1])])
    previous_distance = float(np.linalg.norm(goal - position))
    proposed_distance = float(np.linalg.norm(goal - proposed))
    proposed_progress = previous_distance - proposed_distance
    proposed_recent = tuple(proposed.tolist()) in recent_positions[-8:]
    proposed_safe = (
        proposed_obstacle < config.shield_caution_threshold
        and proposed_hazard < config.hazard_threshold
        and proposed_progress >= -0.25
        and not proposed_recent
    )
    scores = [rollout_cost(action, position, goal, obstacle_risk, hazard_risk, recent_positions, config) for action in range(8)]
    best_action = int(np.argmin(scores))
    best_score = float(scores[best_action])
    proposed_score = float(scores[ppo_action])
    if proposed_safe and proposed_score <= best_score + 5.0:
        return ppo_action, False, "accepted", proposed_score, proposed_score
    reason_parts = []
    if proposed_obstacle >= config.shield_caution_threshold:
        reason_parts.append("obstacle_risk")
    if proposed_hazard >= config.hazard_threshold:
        reason_parts.append("hazard_risk")
    if proposed_progress < -0.25:
        reason_parts.append("negative_progress")
    if proposed_recent:
        reason_parts.append("loop_prevention")
    if not reason_parts:
        reason_parts.append("lookahead_safer")
    return best_action, best_action != ppo_action, "+".join(reason_parts), proposed_score, best_score


def event_rows_for_step(
    base: dict[str, object],
    obstacle_value: float,
    hazard_value: float,
    action_overridden: bool,
    reason: str,
    config: CollisionAvoidanceConfig,
) -> list[dict[str, object]]:
    rows = []
    if obstacle_value >= config.collision_risk_threshold:
        event = dict(base)
        event["event_type"] = "collision"
        event["event_value"] = obstacle_value
        rows.append(event)
    if hazard_value >= config.hazard_threshold:
        event = dict(base)
        event["event_type"] = "hazard_exposure"
        event["event_value"] = hazard_value
        rows.append(event)
    if obstacle_value >= config.collision_risk_threshold and hazard_value >= config.hazard_threshold:
        event = dict(base)
        event["event_type"] = "joint_hazard_obstacle"
        event["event_value"] = max(obstacle_value, hazard_value)
        rows.append(event)
    if action_overridden:
        event = dict(base)
        event["event_type"] = "shield_intervention"
        event["event_value"] = reason
        rows.append(event)
    return rows


def run_episode(
    scenario: dict[str, object],
    mode: str,
    controller: str,
    obstacle_risk: np.ndarray,
    hazard_risk: np.ndarray,
    agent: PPONavigationAgent,
    config: CollisionAvoidanceConfig,
    output_dir: Path,
    episode_index: int,
    episode_variant: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
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
    input_risk = policy_risk(combined_risk, policy_config)
    start_distance = float(np.linalg.norm(goal - start))
    reason = "max_steps"
    success = False
    collided = False
    step_rows = []
    intervention_rows = []
    event_rows = []
    recent_positions = [tuple(position.tolist())]
    start_time = time.perf_counter()
    for step in range(1, config.max_steps + 1):
        state = observation(input_risk, position, goal, policy_config)
        ppo_action, confidence = deterministic_action(agent, state)
        selected_action = ppo_action
        overridden = False
        shield_reason = "not_applied"
        proposed_score = 0.0
        selected_score = 0.0
        if controller == "ppo_shield":
            selected_action, overridden, shield_reason, proposed_score, selected_score = shield_action(
                ppo_action,
                position,
                goal,
                obstacle_risk,
                hazard_risk,
                recent_positions,
                config,
            )
        previous_distance = float(np.linalg.norm(goal - position))
        position = candidate_position(position, selected_action, config.grid_size)
        recent_positions.append(tuple(position.tolist()))
        path.append(position.copy())
        obstacle_value = float(obstacle_risk[int(position[0]), int(position[1])])
        hazard_value = float(hazard_risk[int(position[0]), int(position[1])])
        combined_value = float(combined_risk[int(position[0]), int(position[1])])
        final_distance = float(np.linalg.norm(goal - position))
        progress = previous_distance - final_distance
        if final_distance < config.goal_threshold:
            success = True
            reason = "goal_reached"
        elif obstacle_value >= config.collision_risk_threshold:
            collided = True
            reason = "collision_risk"
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
                "distance_to_goal": final_distance,
                "progress": progress,
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
                }
            )
        event_rows.extend(event_rows_for_step(row_base, obstacle_value, hazard_value, overridden, shield_reason, config))
        if success or collided:
            break
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    trajectory_path = output_dir / "trajectories" / f"episode_{episode_index:03d}_{controller}_{scenario['id']}_{mode}_trajectory.png"
    render_trajectory(combined_risk, path, start, goal, trajectory_path)
    length = path_length(path)
    final_distance = float(np.linalg.norm(goal - position))
    path_obstacle_values = [float(obstacle_risk[int(p[0]), int(p[1])]) for p in path]
    path_hazard_values = [float(hazard_risk[int(p[0]), int(p[1])]) for p in path]
    interventions = sum(int(row["shield_overrode"]) for row in step_rows)
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
        "mean_obstacle_risk": safe_float(float(np.mean(path_obstacle_values))),
        "max_obstacle_risk": safe_float(float(np.max(path_obstacle_values))),
        "min_obstacle_margin": config.collision_risk_threshold - safe_float(float(np.max(path_obstacle_values))),
        "mean_hazard_risk": safe_float(float(np.mean(path_hazard_values))),
        "max_hazard_risk": safe_float(float(np.max(path_hazard_values))),
        "hazard_exposure_steps": sum(1 for value in path_hazard_values if value >= config.hazard_threshold),
        "joint_hazard_obstacle_steps": sum(
            1
            for obstacle_value, hazard_value in zip(path_obstacle_values, path_hazard_values)
            if obstacle_value >= config.collision_risk_threshold and hazard_value >= config.hazard_threshold
        ),
        "shield_interventions": interventions,
        "shield_intervention_rate": interventions / max(len(path) - 1, 1),
        "elapsed_ms": elapsed_ms,
        "trajectory_png": str(trajectory_path),
    }
    return summary, step_rows, intervention_rows, event_rows


def mode_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
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
                "avg_min_obstacle_margin": safe_float(float(np.mean([float(row["min_obstacle_margin"]) for row in selected]))),
                "avg_hazard_exposure_steps": safe_float(float(np.mean([float(row["hazard_exposure_steps"]) for row in selected]))),
                "avg_joint_hazard_obstacle_steps": safe_float(float(np.mean([float(row["joint_hazard_obstacle_steps"]) for row in selected]))),
                "avg_shield_interventions": safe_float(float(np.mean([float(row["shield_interventions"]) for row in selected]))),
                "avg_shield_intervention_rate": safe_float(float(np.mean([float(row["shield_intervention_rate"]) for row in selected]))),
            }
        )
    return output


def selected_visuals(output_dir: Path, run_rows: list[dict[str, object]]) -> list[str]:
    visuals_dir = output_dir / "visuals"
    selected = []
    for mode in ["stereo", "mono"]:
        candidates = [
            row for row in run_rows
            if row["controller"] == "ppo_shield" and row["mode"] == mode and row["scenario_id"] == "baseline_open"
        ]
        if not candidates:
            continue
        path = Path(str(candidates[0]["trajectory_png"]))
        image = cv2.imread(str(path))
        if image is None:
            continue
        output = visuals_dir / f"week05_selected_{mode}_shield_trajectory.png"
        cv2.imwrite(str(output), image)
        selected.append(str(output))
    return selected


def run_week5(
    scenarios_path: Path,
    model_path: Path,
    output_dir: Path,
    run_name: str,
    controllers: list[str],
    selected_scenarios: list[str] | None,
    config: CollisionAvoidanceConfig,
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
    run_rows = []
    step_rows = []
    intervention_rows = []
    event_rows = []
    episode_index = 1
    for controller in controllers:
        for scenario, mode, variant in planned:
            key = "stereo_risk_map" if mode == "stereo" else "mono_risk_map"
            obstacle_risk = load_risk_map(scenario[key], config.grid_size)
            hazard_risk = hazard_overlay(obstacle_risk, str(scenario["id"]), int(variant), config)
            summary, steps, interventions, events = run_episode(
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
            )
            run_rows.append(summary)
            step_rows.extend(steps)
            intervention_rows.extend(interventions)
            event_rows.extend(events)
            episode_index += 1
    summary_rows = mode_summary(run_rows)
    write_csv(output_path / "metrics" / "week05_collision_runs.csv", run_rows)
    write_csv(output_path / "metrics" / "week05_collision_steps.csv", step_rows)
    write_csv(output_path / "metrics" / "week05_collision_interventions.csv", intervention_rows)
    write_csv(output_path / "metrics" / "week05_collision_events.csv", event_rows)
    write_csv(output_path / "metrics" / "week05_collision_summary.csv", summary_rows)
    visuals = selected_visuals(output_path, run_rows)
    write_json(
        output_path / "metrics" / "week05_collision_config.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "scenarios_path": str(scenarios_path),
            "model_path": str(model_path),
            "input_shape": [config.grid_size, config.grid_size, 4],
            "action_dim": 8,
            "training_updates": 0,
            "controllers": controllers,
            "config": asdict(config),
            "selected_visuals": visuals,
            "output_dir": str(output_path),
        },
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Paper 2 Week 5 collision avoidance validation.")
    parser.add_argument("--scenarios-path", type=Path, default=DEFAULT_SCENARIOS_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default="final")
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--controller", action="append", choices=["raw_ppo", "ppo_shield"], default=None)
    parser.add_argument("--episodes", type=int, default=50)
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
    controllers = args.controller or ["raw_ppo", "ppo_shield"]
    config = CollisionAvoidanceConfig(
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
    output_path = run_week5(
        scenarios_path=args.scenarios_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        controllers=controllers,
        selected_scenarios=args.scenario,
        config=config,
    )
    print(f"Week 5 collision avoidance validation complete: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
