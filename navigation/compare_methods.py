import argparse
import numpy as np
import json
import csv
import time
from datetime import datetime
from typing import Dict, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_environment import UAVNavigationEnv
from astar_planner import AStarPlanner
from ppo_navigation import PPONavigationAgent


def run_astar_episode(
    env: UAVNavigationEnv,
    planner: AStarPlanner,
    encounter_threshold: float = 0.2,
    replan_frequency: int = 0,
) -> Dict:
    state = env.reset()
    
    hazard_hits = 0
    total_reward = 0.0
    steps = 0
    hazard_penalty = 0.0
    geometric_path_length = 0.0
    
    current_path = planner.plan(env.hazard_map, env.uav_position, env.goal_position)
    step_idx = 1
    steps_since_replan = 0
    
    while steps < env.max_steps:
        needs_replan = False
        if step_idx >= len(current_path):
            needs_replan = True
        else:
            nx, ny = current_path[step_idx]
            if env.hazard_map[nx, ny] > 0.2:
                needs_replan = True
        if replan_frequency > 0 and steps_since_replan >= replan_frequency:
            needs_replan = True
                
        if needs_replan:
            current_path = planner.plan(env.hazard_map, env.uav_position, env.goal_position)
            step_idx = 1
            steps_since_replan = 0
            
        if len(current_path) < 2:
            import random
            action = random.randint(0, 7)
            actions_map = {
                0: [-1, 0], 1: [-1, 1], 2: [0, 1], 3: [1, 1],
                4: [1, 0], 5: [1, -1], 6: [0, -1], 7: [-1, -1]
            }
            nx = env.uav_position[0] + actions_map[action][0]
            ny = env.uav_position[1] + actions_map[action][1]
            nx = max(0, min(env.grid_size - 1, nx))
            ny = max(0, min(env.grid_size - 1, ny))
            next_pos = np.array([nx, ny])
        else:
            next_pos = current_path[step_idx]
            action = AStarPlanner.action_from_move(env.uav_position, next_pos)
        
        prev_pos = env.uav_position.copy()
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        step_distance = np.linalg.norm(env.uav_position - prev_pos)
        geometric_path_length += float(step_distance)
        
        gt_hazard_level = info.get("gt_hazard_level", 0.0)
        if gt_hazard_level > encounter_threshold:
            hazard_hits += 1
            hazard_penalty += gt_hazard_level
            
        steps += 1
        steps_since_replan += 1
        
        if len(current_path) >= 2 and np.array_equal(env.uav_position, next_pos):
            step_idx += 1
            
        if done:
            break
    
    success = np.array_equal(env.uav_position, env.goal_position)
    final_distance = float(np.linalg.norm(env.uav_position - env.goal_position))
    
    return {
        "success": success,
        "steps": steps,
        "path_length": float(geometric_path_length),
        "final_distance": final_distance,
        "hazard_encounters": hazard_hits,
        "total_reward": float(total_reward),
        "hazard_penalty": float(hazard_penalty),
    }


def run_ppo_episode(env: UAVNavigationEnv, agent: PPONavigationAgent, deterministic: bool = True,
                    encounter_threshold: float = 0.2) -> Dict:
    state = env.reset()
    hazard_hits = 0
    total_reward = 0.0
    steps = 0
    hazard_penalty = 0.0
    geometric_path_length = 0.0
    
    while steps < env.max_steps:
        action = agent.select_action(state, deterministic=deterministic)
        
        prev_pos = env.uav_position.copy()
        
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        step_distance = np.linalg.norm(env.uav_position - prev_pos)
        geometric_path_length += float(step_distance)
        
        gt_hazard_level = info.get("gt_hazard_level", 0.0)
        if gt_hazard_level > encounter_threshold:
            hazard_hits += 1
            hazard_penalty += gt_hazard_level
        
        state = next_state
        steps += 1
        
        if done:
            break
    
    success = np.array_equal(env.uav_position, env.goal_position)
    final_distance = float(np.linalg.norm(env.uav_position - env.goal_position))
    
    agent.experience_buffer.clear()
    
    return {
        "success": success,
        "steps": steps,
        "path_length": float(geometric_path_length),
        "final_distance": final_distance,
        "hazard_encounters": hazard_hits,
        "total_reward": float(total_reward),
        "hazard_penalty": float(hazard_penalty),
    }


def evaluate_method(
    episodes: int,
    method: str,
    env: UAVNavigationEnv,
    planner=None,
    agent=None,
    encounter_threshold: float = 0.2,
    replan_frequency: int = 0,
) -> Dict:
    print(f"\n{'='*60}")
    print(f"Evaluating {method} method for {episodes} episodes...")
    print(f"{'='*60}")
    
    results = []
    start_time = time.time()
    
    for i in range(episodes):
        if (i + 1) % max(1, episodes // 10) == 0:
            print(f"Progress: {i+1}/{episodes} episodes ({100*(i+1)/episodes:.1f}%)")
        
        if method == "A*":
            result = run_astar_episode(
                env,
                planner,
                encounter_threshold=encounter_threshold,
                replan_frequency=replan_frequency,
            )
        elif method == "PPO":
            result = run_ppo_episode(env, agent, deterministic=True, encounter_threshold=encounter_threshold)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results.append(result)
    
    elapsed_time = time.time() - start_time
    
    successes = sum(1 for r in results if r["success"])
    avg_steps = np.mean([r["steps"] for r in results])
    avg_path_len = np.mean([r["path_length"] for r in results])
    avg_final_dist = np.mean([r["final_distance"] for r in results])
    avg_hazard_encounters = np.mean([r["hazard_encounters"] for r in results])
    avg_total_reward = np.mean([r["total_reward"] for r in results])
    avg_hazard_penalty = np.mean([r["hazard_penalty"] for r in results])
    
    success_rate = successes / episodes
    
    return {
        "method": method,
        "episodes": episodes,
        "success_rate": float(success_rate),
        "successes": successes,
        "avg_steps": float(avg_steps),
        "std_steps": float(np.std([r["steps"] for r in results])),
        "avg_path_len": float(avg_path_len),
        "avg_final_dist": float(avg_final_dist),
        "avg_hazard_encounters": float(avg_hazard_encounters),
        "avg_total_reward": float(avg_total_reward),
        "std_reward": float(np.std([r["total_reward"] for r in results])),
        "avg_hazard_penalty": float(avg_hazard_penalty),
        "elapsed_time": float(elapsed_time),
        "results": results,
    }


def generate_comparison_report(astar_results: Dict, ppo_results: Dict, 
                              output_dir: str = "comparison_results") -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    success_rate_improvement = ((ppo_results["success_rate"] - astar_results["success_rate"]) / 
                                max(astar_results["success_rate"], 0.001)) * 100
    steps_improvement = ((astar_results["avg_steps"] - ppo_results["avg_steps"]) / 
                        max(astar_results["avg_steps"], 0.001)) * 100
    reward_improvement = ((ppo_results["avg_total_reward"] - astar_results["avg_total_reward"]) / 
                         max(abs(astar_results["avg_total_reward"]), 0.001)) * 100
    hazard_improvement = ((astar_results["avg_hazard_encounters"] - ppo_results["avg_hazard_encounters"]) / 
                         max(astar_results["avg_hazard_encounters"], 0.001)) * 100
    
    report = {
        "comparison_date": datetime.now().isoformat(),
        "episodes_per_method": astar_results["episodes"],
        "methods": {
            "A*": {
                "success_rate": astar_results["success_rate"],
                "successes": astar_results["successes"],
                "avg_steps": astar_results["avg_steps"],
                "std_steps": astar_results["std_steps"],
                "avg_path_len": astar_results["avg_path_len"],
                "avg_final_dist": astar_results["avg_final_dist"],
                "avg_hazard_encounters": astar_results["avg_hazard_encounters"],
                "avg_total_reward": astar_results["avg_total_reward"],
                "std_reward": astar_results["std_reward"],
                "avg_hazard_penalty": astar_results["avg_hazard_penalty"],
                "elapsed_time_seconds": astar_results["elapsed_time"],
            },
            "PPO": {
                "success_rate": ppo_results["success_rate"],
                "successes": ppo_results["successes"],
                "avg_steps": ppo_results["avg_steps"],
                "std_steps": ppo_results["std_steps"],
                "avg_path_len": ppo_results["avg_path_len"],
                "avg_final_dist": ppo_results["avg_final_dist"],
                "avg_hazard_encounters": ppo_results["avg_hazard_encounters"],
                "avg_total_reward": ppo_results["avg_total_reward"],
                "std_reward": ppo_results["std_reward"],
                "avg_hazard_penalty": ppo_results["avg_hazard_penalty"],
                "elapsed_time_seconds": ppo_results["elapsed_time"],
            },
        },
        "comparison": {
            "success_rate_improvement_pct": float(success_rate_improvement),
            "steps_improvement_pct": float(steps_improvement),
            "reward_improvement_pct": float(reward_improvement),
            "hazard_encounters_improvement_pct": float(hazard_improvement),
            "better_success_rate": "PPO" if ppo_results["success_rate"] > astar_results["success_rate"] else "A*",
            "fewer_steps": "PPO" if ppo_results["avg_steps"] < astar_results["avg_steps"] else "A*",
            "higher_reward": "PPO" if ppo_results["avg_total_reward"] > astar_results["avg_total_reward"] else "A*",
            "fewer_hazard_encounters": "PPO" if ppo_results["avg_hazard_encounters"] < astar_results["avg_hazard_encounters"] else "A*",
        },
    }
    
    report_path = os.path.join(output_dir, f"comparison_report_{timestamp}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS SUMMARY")
    print("="*60)
    print(f"\nEpisodes evaluated per method: {astar_results['episodes']}")
    print(f"\n{'Metric':<30} {'A*':<15} {'PPO':<15} {'Winner':<10}")
    print("-"*70)
    print(f"{'Success Rate':<30} {astar_results['success_rate']:.3f} ({astar_results['successes']}/{astar_results['episodes']}){'':<8} {ppo_results['success_rate']:.3f} ({ppo_results['successes']}/{ppo_results['episodes']}){'':<8} {report['comparison']['better_success_rate']}")
    print(f"{'Avg Steps':<30} {astar_results['avg_steps']:.2f} ± {astar_results['std_steps']:.2f}{'':<4} {ppo_results['avg_steps']:.2f} ± {ppo_results['std_steps']:.2f}{'':<4} {report['comparison']['fewer_steps']}")
    print(f"{'Avg Path Length':<30} {astar_results['avg_path_len']:.2f}{'':<12} {ppo_results['avg_path_len']:.2f}{'':<12} -")
    print(f"{'Avg Final Distance':<30} {astar_results['avg_final_dist']:.2f}{'':<12} {ppo_results['avg_final_dist']:.2f}{'':<12} -")
    print(f"{'Avg Hazard Encounters':<30} {astar_results['avg_hazard_encounters']:.2f}{'':<12} {ppo_results['avg_hazard_encounters']:.2f}{'':<12} {report['comparison']['fewer_hazard_encounters']}")
    print(f"{'Avg Total Reward':<30} {astar_results['avg_total_reward']:.2f} ± {astar_results['std_reward']:.2f}{'':<4} {ppo_results['avg_total_reward']:.2f} ± {ppo_results['std_reward']:.2f}{'':<4} {report['comparison']['higher_reward']}")
    print(f"{'Avg Hazard Penalty':<30} {astar_results['avg_hazard_penalty']:.2f}{'':<12} {ppo_results['avg_hazard_penalty']:.2f}{'':<12} -")
    print(f"{'Elapsed Time (s)':<30} {astar_results['elapsed_time']:.2f}{'':<12} {ppo_results['elapsed_time']:.2f}{'':<12} -")
    
    print(f"\n{'Improvements (PPO vs A*):':<30}")
    print(f"  Success Rate: {success_rate_improvement:+.2f}%")
    print(f"  Steps: {steps_improvement:+.2f}%")
    print(f"  Reward: {reward_improvement:+.2f}%")
    print(f"  Hazard Encounters: {hazard_improvement:+.2f}%")
    
    print(f"\n{'='*60}")
    print(f"Report saved to: {report_path}")
    print(f"{'='*60}\n")
    
    return report


def export_episode_results_csv(
    astar_results: Dict,
    ppo_results: Dict,
    output_dir: str,
    timestamp: str,
    run_config: Dict,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"episode_results_{timestamp}.csv")
    fieldnames = [
        "run_id",
        "method",
        "episode",
        "success",
        "steps",
        "path_length",
        "final_distance",
        "hazard_encounters",
        "total_reward",
        "hazard_penalty",
        "grid",
        "max_steps",
        "encounter_threshold",
        "termination_threshold",
        "perception_noise",
        "replan_frequency",
        "confidence_decay",
        "observation_radius",
        "timestamp_utc",
    ]

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for method_results in [astar_results, ppo_results]:
            for episode_idx, episode_result in enumerate(method_results["results"], start=1):
                writer.writerow(
                    {
                        "run_id": timestamp,
                        "method": method_results["method"],
                        "episode": episode_idx,
                        "success": int(bool(episode_result["success"])),
                        "steps": episode_result["steps"],
                        "path_length": episode_result["path_length"],
                        "final_distance": episode_result["final_distance"],
                        "hazard_encounters": episode_result["hazard_encounters"],
                        "total_reward": episode_result["total_reward"],
                        "hazard_penalty": episode_result["hazard_penalty"],
                        "grid": run_config["grid"],
                        "max_steps": run_config["max_steps"],
                        "encounter_threshold": run_config["encounter_threshold"],
                        "termination_threshold": run_config["termination_threshold"],
                        "perception_noise": run_config["perception_noise"],
                        "replan_frequency": run_config["replan_frequency"],
                        "confidence_decay": run_config["confidence_decay"],
                        "observation_radius": run_config["observation_radius"],
                        "timestamp_utc": datetime.utcnow().isoformat(),
                    }
                )

    return csv_path


def export_run_summary_csv(
    astar_results: Dict,
    ppo_results: Dict,
    output_dir: str,
    timestamp: str,
    run_config: Dict,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "run_summary.csv")
    fieldnames = [
        "run_id",
        "method",
        "episodes",
        "success_rate",
        "successes",
        "avg_steps",
        "std_steps",
        "avg_path_len",
        "avg_final_dist",
        "avg_hazard_encounters",
        "avg_total_reward",
        "std_reward",
        "avg_hazard_penalty",
        "elapsed_time",
        "grid",
        "max_steps",
        "encounter_threshold",
        "termination_threshold",
        "perception_noise",
        "replan_frequency",
        "confidence_decay",
        "observation_radius",
        "timestamp_utc",
    ]

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for method_results in [astar_results, ppo_results]:
            writer.writerow(
                {
                    "run_id": timestamp,
                    "method": method_results["method"],
                    "episodes": method_results["episodes"],
                    "success_rate": method_results["success_rate"],
                    "successes": method_results["successes"],
                    "avg_steps": method_results["avg_steps"],
                    "std_steps": method_results["std_steps"],
                    "avg_path_len": method_results["avg_path_len"],
                    "avg_final_dist": method_results["avg_final_dist"],
                    "avg_hazard_encounters": method_results["avg_hazard_encounters"],
                    "avg_total_reward": method_results["avg_total_reward"],
                    "std_reward": method_results["std_reward"],
                    "avg_hazard_penalty": method_results["avg_hazard_penalty"],
                    "elapsed_time": method_results["elapsed_time"],
                    "grid": run_config["grid"],
                    "max_steps": run_config["max_steps"],
                    "encounter_threshold": run_config["encounter_threshold"],
                    "termination_threshold": run_config["termination_threshold"],
                    "perception_noise": run_config["perception_noise"],
                    "replan_frequency": run_config["replan_frequency"],
                    "confidence_decay": run_config["confidence_decay"],
                    "observation_radius": run_config["observation_radius"],
                    "timestamp_utc": datetime.utcnow().isoformat(),
                }
            )

    return csv_path


def main():
    parser = argparse.ArgumentParser(description='Compare A* vs PPO navigation methods')
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes to evaluate per method")
    parser.add_argument("--grid", type=int, default=50, help="Grid size")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--classifier", type=str, default="../checkpoints/best_model.pth", 
                       help="Path to disaster classifier model")
    parser.add_argument("--ppo_model", type=str, default="navigation_models/best_navigation_model.pth",
                       help="Path to trained PPO model")
    parser.add_argument("--output_dir", type=str, default="comparison_results",
                       help="Directory to save comparison results")
    parser.add_argument("--no-cache", action="store_true", help="Disable image caching")
    parser.add_argument("--encounter_threshold", type=float, default=0.2, 
                       help="Hazard threshold for counting encounters")
    parser.add_argument("--termination_threshold", type=float, default=0.9,
                       help="Hazard threshold for terminal failure collision")
    parser.add_argument("--perception_noise", type=float, default=0.0,
                       help="Amount of perception noise (std dev) to add to image classifier outputs")
    parser.add_argument("--replan_frequency", type=int, default=0,
                       help="A* forced replan cadence in steps (0 means on-demand only)")
    parser.add_argument("--confidence_decay", type=float, default=0.95,
                       help="Confidence decay factor for hazard confidence memory")
    parser.add_argument("--observation_radius", type=int, default=2,
                       help="Radius of local cell classification around UAV")
    args = parser.parse_args()
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print("="*60)
    print("UAV Navigation Methods Comparison")
    print("="*60)
    print(f"Episodes per method: {args.episodes}")
    print(f"Grid size: {args.grid}x{args.grid}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Classifier: {args.classifier}")
    print(f"PPO Model: {args.ppo_model}")
    print(f"A* Replan Frequency: {args.replan_frequency} (0=on-demand)")
    print(f"Confidence Decay: {args.confidence_decay}")
    print(f"Observation Radius: {args.observation_radius}")
    print("="*60)
    
    env = UAVNavigationEnv(
        grid_size=args.grid, 
        max_steps=args.max_steps, 
        classifier_path=args.classifier,
        cache_imagery=not args.no_cache,
        observation_radius=args.observation_radius,
        confidence_decay=args.confidence_decay,
        termination_threshold=args.termination_threshold,
        perception_noise_std=args.perception_noise
    )
    
    planner = AStarPlanner(grid_size=args.grid, diag=True)
    
    input_shape = (args.grid, args.grid, 4)
    action_dim = 8
    agent = PPONavigationAgent(input_shape, action_dim)
    
    model_path = args.ppo_model
    if os.path.exists(model_path):
        print(f"\nLoading PPO model from: {model_path}")
        if agent.load_model(model_path):
            print("✓ PPO model loaded successfully")
        else:
            print("⚠ Failed to load PPO model, using untrained model")
    else:
        print(f"\n⚠ PPO model not found at: {model_path}")
        print("⚠ Running with untrained PPO agent (results may be poor)")
    
    astar_results = evaluate_method(args.episodes, "A*", env, planner=planner, 
                                    encounter_threshold=args.encounter_threshold,
                                    replan_frequency=args.replan_frequency)
    
    ppo_results = evaluate_method(args.episodes, "PPO", env, agent=agent,
                                  encounter_threshold=args.encounter_threshold)
    
    report = generate_comparison_report(astar_results, ppo_results, args.output_dir)
    run_config = {
        "grid": args.grid,
        "max_steps": args.max_steps,
        "encounter_threshold": args.encounter_threshold,
        "termination_threshold": args.termination_threshold,
        "perception_noise": args.perception_noise,
        "replan_frequency": args.replan_frequency,
        "confidence_decay": args.confidence_decay,
        "observation_radius": args.observation_radius,
    }
    episode_csv_path = export_episode_results_csv(
        astar_results, ppo_results, args.output_dir, run_id, run_config
    )
    summary_csv_path = export_run_summary_csv(
        astar_results, ppo_results, args.output_dir, run_id, run_config
    )
    
    print("\nComparison completed successfully!")
    print(f"Episode-level CSV saved to: {episode_csv_path}")
    print(f"Run-summary CSV saved to: {summary_csv_path}")


if __name__ == "__main__":
    main()

