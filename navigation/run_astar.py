import argparse
import numpy as np
import time
from uav_environment import UAVNavigationEnv
from astar_planner import AStarPlanner


def run_episode(env: UAVNavigationEnv, planner: AStarPlanner, render: bool = False):
    state = env.reset()
    start = env.uav_position.copy()
    goal = env.goal_position.copy()
    path = planner.plan(env.hazard_map, start, goal)
    if len(path) < 2:
        return {
            "success": False,
            "steps": 0,
            "path_length": 0,
            "final_distance": float(np.linalg.norm(env.uav_position - env.goal_position)),
            "hazard_encounters": 0,
            "total_reward": 0.0,
        }
    step_idx = 1
    hazard_hits = 0
    total_reward = 0.0
    steps = 0
    while steps < env.max_steps:
        if step_idx >= len(path):
            break
        next_pos = path[step_idx]
        action = AStarPlanner.action_from_move(env.uav_position, next_pos)
        state, reward, done, info = env.step(action)
        total_reward += reward
        if info.get("in_danger_zone", False):
            hazard_hits += 1
        if done:
            break
        steps += 1
        if np.array_equal(env.uav_position, next_pos):
            step_idx += 1
    success = np.array_equal(env.uav_position, env.goal_position)
    final_distance = float(np.linalg.norm(env.uav_position - env.goal_position))
    return {
        "success": success,
        "steps": steps,
        "path_length": int(len(path)),
        "final_distance": final_distance,
        "hazard_encounters": int(hazard_hits),
        "total_reward": float(total_reward),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--grid", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--classifier", type=str, default="../checkpoints/best_model.pth")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = UAVNavigationEnv(grid_size=args.grid, max_steps=args.max_steps, classifier_path=args.classifier)
    planner = AStarPlanner(grid_size=args.grid, diag=True)

    successes = 0
    total_steps = 0
    total_path = 0
    total_distance = 0.0
    total_hazards = 0
    total_reward = 0.0

    for i in range(args.episodes):
        result = run_episode(env, planner, render=args.render)
        successes += 1 if result["success"] else 0
        total_steps += result["steps"]
        total_path += result["path_length"]
        total_distance += result["final_distance"]
        total_hazards += result["hazard_encounters"]
        total_reward += result["total_reward"]

    eps = max(1, args.episodes)
    print(f"episodes={args.episodes}")
    print(f"success_rate={successes/eps:.3f}")
    print(f"avg_steps={total_steps/eps:.2f}")
    print(f"avg_path_len={total_path/eps:.2f}")
    print(f"avg_final_dist={total_distance/eps:.2f}")
    print(f"avg_hazard_encounters={total_hazards/eps:.2f}")
    print(f"avg_total_reward={total_reward/eps:.2f}")


if __name__ == "__main__":
    main()


