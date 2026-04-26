from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ppo_navigation import PPONavigationAgent


DEFAULT_SCENARIOS_PATH = SCRIPT_DIR / "scenarios.json"
DEFAULT_MODEL_PATH = SCRIPT_DIR / "navigation_models" / "best_navigation_model.pth"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "week04_navigation_integration"


@dataclass(frozen=True)
class NavigationIntegrationConfig:
    grid_size: int = 50
    max_steps: int = 200
    collision_risk_threshold: float = 0.95
    goal_threshold: float = 1.0
    episodes: int | None = None
    start_goal_risk_threshold: float = 0.5
    policy_risk_scale: float = 1.0


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "metrics": output_dir / "metrics",
        "trajectories": output_dir / "trajectories",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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


def load_scenarios(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    scenarios = payload.get("scenarios", [])
    if not scenarios:
        raise RuntimeError(f"No scenarios found in {path}")
    return scenarios


def load_risk_map(path_value: str | Path, grid_size: int) -> np.ndarray:
    risk = np.load(resolve_path(path_value)).astype(np.float32)
    risk = np.clip(risk, 0.0, 1.0)
    if risk.shape != (grid_size, grid_size):
        risk = cv2.resize(risk, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    risk = np.clip(risk.astype(np.float32), 0.0, 1.0)
    risk[~np.isfinite(risk)] = 0.0
    return risk


def coord_grid(grid_size: int) -> np.ndarray:
    rows = np.arange(grid_size, dtype=np.float32)
    return np.stack(np.meshgrid(rows, rows, indexing="ij"))


def observation(risk: np.ndarray, position: np.ndarray, goal: np.ndarray, config: NavigationIntegrationConfig) -> np.ndarray:
    obs = np.zeros((config.grid_size, config.grid_size, 4), dtype=np.float32)
    obs[:, :, 0] = risk
    obs[:, :, 1] = 1.0
    coords = coord_grid(config.grid_size)
    max_distance = float(np.linalg.norm(np.array([0, 0]) - np.array([config.grid_size - 1, config.grid_size - 1])))
    dy = coords[0] - float(position[0])
    dx = coords[1] - float(position[1])
    obs[:, :, 2] = 1.0 - np.clip(np.sqrt(dy * dy + dx * dx) / max_distance, 0.0, 1.0)
    gy = coords[0] - float(goal[0])
    gx = coords[1] - float(goal[1])
    obs[:, :, 3] = 1.0 - np.clip(np.sqrt(gy * gy + gx * gx) / max_distance, 0.0, 1.0)
    return obs


def policy_risk(risk: np.ndarray, config: NavigationIntegrationConfig) -> np.ndarray:
    return np.clip(risk * config.policy_risk_scale, 0.0, 1.0).astype(np.float32)


def deterministic_action(agent: PPONavigationAgent, state: np.ndarray) -> tuple[int, float]:
    state_tensor = torch.FloatTensor(state).to(agent.device)
    with torch.no_grad():
        logits, _ = agent.old_policy_net(state_tensor)
        probs = F.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1)
    return int(action.item()), float(probs[0, action].item())


def action_delta(action: int) -> np.ndarray:
    actions = {
        0: [-1, 0],
        1: [-1, 1],
        2: [0, 1],
        3: [1, 1],
        4: [1, 0],
        5: [1, -1],
        6: [0, -1],
        7: [-1, -1],
    }
    return np.array(actions[int(action)], dtype=np.int32)


def path_length(path: list[np.ndarray]) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(np.linalg.norm(path[i] - path[i - 1]) for i in range(1, len(path))))


def render_trajectory(risk: np.ndarray, path: list[np.ndarray], start: np.ndarray, goal: np.ndarray, output_path: Path) -> None:
    base = cv2.applyColorMap((np.clip(risk, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    scale = 12
    image = cv2.resize(base, (risk.shape[1] * scale, risk.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    points = [(int(p[1] * scale + scale / 2), int(p[0] * scale + scale / 2)) for p in path]
    for idx in range(1, len(points)):
        cv2.line(image, points[idx - 1], points[idx], (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(image, (int(start[1] * scale + scale / 2), int(start[0] * scale + scale / 2)), 7, (0, 255, 0), -1)
    cv2.circle(image, (int(goal[1] * scale + scale / 2), int(goal[0] * scale + scale / 2)), 8, (255, 255, 0), -1)
    cv2.circle(image, points[-1], 7, (255, 0, 255), -1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write trajectory image: {output_path}")


def shifted_cell(cell: list[int] | tuple[int, int], variant: int, grid_size: int, role: str) -> list[int]:
    offsets = {
        "start": [(0, 0), (0, 2), (2, 0), (0, -2), (-2, 0)],
        "goal": [(0, 0), (0, -2), (-2, 0), (0, 2), (2, 0)],
    }[role]
    row_offset, col_offset = offsets[variant % len(offsets)]
    return [
        int(np.clip(int(cell[0]) + row_offset, 1, grid_size - 2)),
        int(np.clip(int(cell[1]) + col_offset, 1, grid_size - 2)),
    ]


def nearest_safe_cell(risk: np.ndarray, target: list[int], config: NavigationIntegrationConfig) -> list[int]:
    target_array = np.array(target, dtype=np.int32)
    safe = np.argwhere(risk < config.start_goal_risk_threshold)
    if safe.size == 0:
        safe = np.argwhere(risk < config.collision_risk_threshold)
    if safe.size == 0:
        return target
    distances = np.sum((safe - target_array) ** 2, axis=1)
    chosen = safe[int(np.argmin(distances))]
    return [int(chosen[0]), int(chosen[1])]


def episode_plan(
    scenarios: list[dict[str, object]],
    config: NavigationIntegrationConfig,
) -> list[tuple[dict[str, object], str, int]]:
    if config.episodes is None:
        return [(scenario, mode, 0) for scenario in scenarios for mode in ["stereo", "mono"]]
    modes = ["stereo", "mono"]
    plan: list[tuple[dict[str, object], str, int]] = []
    counts: dict[tuple[str, str], int] = {}
    for idx in range(config.episodes):
        scenario = scenarios[(idx // len(modes)) % len(scenarios)]
        mode = modes[idx % len(modes)]
        key = (str(scenario["id"]), mode)
        variant = counts.get(key, 0)
        counts[key] = variant + 1
        scenario_copy = dict(scenario)
        stereo_risk = load_risk_map(scenario["stereo_risk_map"], config.grid_size)
        mono_risk = load_risk_map(scenario["mono_risk_map"], config.grid_size)
        shared_risk = np.maximum(stereo_risk, mono_risk)
        start_target = shifted_cell(scenario["start_cell"], variant, config.grid_size, "start")
        goal_target = shifted_cell(scenario["goal_cell"], variant, config.grid_size, "goal")
        scenario_copy["start_cell"] = nearest_safe_cell(shared_risk, start_target, config)
        scenario_copy["goal_cell"] = nearest_safe_cell(shared_risk, goal_target, config)
        plan.append((scenario_copy, mode, variant))
    return plan


def run_navigation(
    scenario: dict[str, object],
    mode: str,
    risk: np.ndarray,
    agent: PPONavigationAgent,
    config: NavigationIntegrationConfig,
    output_dir: Path,
    episode_index: int,
    episode_variant: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    start = np.array(scenario["start_cell"], dtype=np.int32)
    goal = np.array(scenario["goal_cell"], dtype=np.int32)
    position = start.copy()
    path = [position.copy()]
    rows: list[dict[str, object]] = []
    start_distance = float(np.linalg.norm(goal - start))
    reason = "max_steps"
    success = False
    collided = False
    start_time = time.perf_counter()
    input_risk = policy_risk(risk, config)
    for step in range(1, config.max_steps + 1):
        state = observation(input_risk, position, goal, config)
        action, confidence = deterministic_action(agent, state)
        candidate = np.clip(position + action_delta(action), 0, config.grid_size - 1)
        prev_distance = float(np.linalg.norm(goal - position))
        position = candidate.astype(np.int32)
        current_risk = float(risk[int(position[0]), int(position[1])])
        final_distance = float(np.linalg.norm(goal - position))
        path.append(position.copy())
        if final_distance < config.goal_threshold:
            success = True
            reason = "goal_reached"
        elif current_risk >= config.collision_risk_threshold:
            collided = True
            reason = "collision_risk"
        progress = prev_distance - final_distance
        rows.append(
            {
                "episode": episode_index,
                "episode_variant": episode_variant,
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "mode": mode,
                "step": step,
                "action": action,
                "action_confidence": confidence,
                "row": int(position[0]),
                "col": int(position[1]),
                "cell_risk": current_risk,
                "distance_to_goal": final_distance,
                "progress": progress,
                "done": int(success or collided),
                "done_reason": reason if success or collided else "",
            }
        )
        if success or collided:
            break
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    length = path_length(path)
    final_distance = float(np.linalg.norm(goal - position))
    efficiency = float(start_distance / max(length, 1e-6)) if success else 0.0
    trajectory_path = output_dir / "trajectories" / f"episode_{episode_index:03d}_{scenario['id']}_{mode}_trajectory.png"
    render_trajectory(risk, path, start, goal, trajectory_path)
    start_risk = float(risk[int(start[0]), int(start[1])])
    goal_risk = float(risk[int(goal[0]), int(goal[1])])
    summary = {
        "episode": episode_index,
        "episode_variant": episode_variant,
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "mode": mode,
        "frame_id": scenario["frame_id"],
        "start_row": int(start[0]),
        "start_col": int(start[1]),
        "goal_row": int(goal[0]),
        "goal_col": int(goal[1]),
        "start_risk": start_risk,
        "goal_risk": goal_risk,
        "success": int(success),
        "collided": int(collided),
        "done_reason": reason,
        "steps": len(path) - 1,
        "path_length": length,
        "start_distance": start_distance,
        "final_distance": final_distance,
        "path_efficiency": efficiency,
        "mean_path_risk": float(np.mean([risk[int(p[0]), int(p[1])] for p in path])),
        "max_path_risk": float(np.max([risk[int(p[0]), int(p[1])] for p in path])),
        "elapsed_ms": elapsed_ms,
        "trajectory_png": str(trajectory_path),
    }
    return summary, rows


def run_week4(
    scenarios_path: Path | str = DEFAULT_SCENARIOS_PATH,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    output_dir: Path | str = DEFAULT_RESULTS_ROOT,
    run_name: str | None = None,
    selected_scenarios: list[str] | None = None,
    config: NavigationIntegrationConfig | None = None,
) -> Path:
    config = config or NavigationIntegrationConfig()
    scenarios_path = Path(scenarios_path)
    model_path = Path(model_path)
    output_root = Path(output_dir)
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_root / run_name
    ensure_dirs(output_path)
    scenarios = load_scenarios(scenarios_path)
    if selected_scenarios:
        selected = set(selected_scenarios)
        scenarios = [scenario for scenario in scenarios if str(scenario["id"]) in selected]
    agent = PPONavigationAgent((config.grid_size, config.grid_size, 4), 8)
    if not agent.load_model(str(model_path)):
        raise RuntimeError(f"PPO checkpoint could not be loaded: {model_path}")
    run_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = []
    for episode_index, (scenario, mode, variant) in enumerate(episode_plan(scenarios, config), start=1):
        key = "stereo_risk_map" if mode == "stereo" else "mono_risk_map"
        risk = load_risk_map(scenario[key], config.grid_size)
        summary, steps = run_navigation(scenario, mode, risk, agent, config, output_path, episode_index, variant)
        run_rows.append(summary)
        step_rows.extend(steps)
    write_csv(output_path / "metrics" / "week04_navigation_runs.csv", run_rows)
    write_csv(output_path / "metrics" / "week04_navigation_steps.csv", step_rows)
    mode_summaries: list[dict[str, object]] = []
    for mode in ["stereo", "mono"]:
        mode_rows = [row for row in run_rows if row["mode"] == mode]
        runs = len(mode_rows)
        successes = sum(int(row["success"]) for row in mode_rows)
        collisions = sum(int(row["collided"]) for row in mode_rows)
        timeouts = sum(1 for row in mode_rows if row["done_reason"] == "max_steps")
        mode_summaries.append(
            {
                "mode": mode,
                "runs": runs,
                "successes": successes,
                "success_rate": successes / runs if runs else 0.0,
                "collisions": collisions,
                "collision_rate": collisions / runs if runs else 0.0,
                "timeouts": timeouts,
                "timeout_rate": timeouts / runs if runs else 0.0,
                "avg_steps": float(np.mean([float(row["steps"]) for row in mode_rows])),
                "avg_final_distance": float(np.mean([float(row["final_distance"]) for row in mode_rows])),
                "avg_path_length": float(np.mean([float(row["path_length"]) for row in mode_rows])),
                "avg_path_efficiency": float(np.mean([float(row["path_efficiency"]) for row in mode_rows])),
                "avg_mean_path_risk": float(np.mean([float(row["mean_path_risk"]) for row in mode_rows])),
                "max_start_risk": float(np.max([float(row["start_risk"]) for row in mode_rows])),
                "max_goal_risk": float(np.max([float(row["goal_risk"]) for row in mode_rows])),
            }
        )
    write_csv(output_path / "metrics" / "week04_navigation_summary.csv", mode_summaries)
    write_json(
        output_path / "metrics" / "week04_navigation_config.json",
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "scenarios_path": str(scenarios_path),
            "model_path": str(model_path),
            "input_shape": [config.grid_size, config.grid_size, 4],
            "action_dim": 8,
            "training_updates": 0,
            "config": asdict(config),
            "output_dir": str(output_path),
        },
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Paper 2 Week 4 PPO navigation integration.")
    parser.add_argument("--scenarios-path", type=Path, default=DEFAULT_SCENARIOS_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--grid-size", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--collision-risk-threshold", type=float, default=0.95)
    parser.add_argument("--goal-threshold", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--start-goal-risk-threshold", type=float, default=0.5)
    parser.add_argument("--policy-risk-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = NavigationIntegrationConfig(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        collision_risk_threshold=args.collision_risk_threshold,
        goal_threshold=args.goal_threshold,
        episodes=args.episodes,
        start_goal_risk_threshold=args.start_goal_risk_threshold,
        policy_risk_scale=args.policy_risk_scale,
    )
    output_path = run_week4(
        scenarios_path=args.scenarios_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        selected_scenarios=args.scenario,
        config=config,
    )
    print(f"Week 4 PPO navigation integration complete: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
