import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from astar_planner import AStarPlanner
from ppo_navigation import PPONavigationAgent
from uav_environment import UAVNavigationEnv


PROFILES = [
    ("fire_light", "Light/Sparse Fire"),
    ("fire_moderate", "Moderate Fire"),
    ("fire_dense", "Dense/Heavy Fire"),
]

METHODS = ["A*", "PPO"]

ACTIONS = {
    0: np.array([-1, 0]),
    1: np.array([-1, 1]),
    2: np.array([0, 1]),
    3: np.array([1, 1]),
    4: np.array([1, 0]),
    5: np.array([1, -1]),
    6: np.array([0, -1]),
    7: np.array([-1, -1]),
}

FIRE_DENSITY_RANK = {
    "fire_light": 1,
    "fire_moderate": 2,
    "fire_dense": 3,
}


def parsed_grid_sizes(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def make_env(args, profile, grid_size):
    return UAVNavigationEnv(
        grid_size=grid_size,
        max_steps=args.max_steps,
        classifier_path=args.classifier,
        cache_imagery=False,
        observation_radius=args.observation_radius,
        confidence_decay=args.confidence_decay,
        termination_threshold=args.termination_threshold,
        perception_noise_std=args.perception_noise,
        lightweight_info=False,
        aerial_cell_px=args.aerial_cell_px,
        quiet_scene=True,
        scene_profile=profile,
    )


def clear_device_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def fire_stats(env):
    fires = [d for d in env.disaster_locations if d["type"] == "fire"]
    intensities = [float(d["intensity"]) for d in fires]
    gt = np.asarray(env.gt_hazard_map, dtype=float)
    return {
        "fire_zone_count": len(fires),
        "avg_fire_intensity": float(np.mean(intensities)) if intensities else 0.0,
        "max_gt_hazard": float(np.max(gt)) if gt.size else 0.0,
        "high_risk_cell_fraction": float(np.mean(gt > 0.2)) if gt.size else 0.0,
    }


def path_metrics(env):
    final_distance = float(np.linalg.norm(env.uav_position - env.goal_position))
    path = np.asarray(env.path_history, dtype=float)
    path_length = 0.0
    if len(path) > 1:
        path_length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    ideal = float(np.linalg.norm(env.goal_position - np.array([2, 2])))
    path_efficiency = float(ideal / path_length) if path_length > 0 and final_distance < 1.0 else 0.0
    return final_distance, path_length, path_efficiency


def finish_result(env, phase, grid_size, method, profile, label, episode, seed, total_reward, hazard_hits, hazard_penalty, start_time):
    final_distance, path_length, path_efficiency = path_metrics(env)
    return {
        "phase": phase,
        "grid_size": grid_size,
        "profile": profile,
        "environment": label,
        "fire_density_rank": FIRE_DENSITY_RANK[profile],
        "method": method,
        "episode": episode,
        "seed": seed,
        "success": int(final_distance < 1.0),
        "steps": env.current_step,
        "path_length": path_length,
        "path_efficiency": path_efficiency,
        "final_distance": final_distance,
        "hazard_encounters": hazard_hits,
        "total_reward": float(total_reward),
        "hazard_penalty": float(hazard_penalty),
        "elapsed_time": float(time.time() - start_time),
        **fire_stats(env),
    }


def run_astar(env, planner, args, phase, grid_size, profile, label, episode, seed):
    start_time = time.time()
    env.reset(seed=seed)
    hazard_hits = 0
    hazard_penalty = 0.0
    total_reward = 0.0
    current_path = planner.plan(env.hazard_map, env.uav_position, env.goal_position)
    step_idx = 1
    steps_since_replan = 0
    while env.current_step < env.max_steps:
        needs_replan = step_idx >= len(current_path)
        if not needs_replan:
            nx, ny = current_path[step_idx]
            if env.hazard_map[nx, ny] > args.encounter_threshold:
                needs_replan = True
        if args.replan_frequency > 0 and steps_since_replan >= args.replan_frequency:
            needs_replan = True
        if needs_replan:
            current_path = planner.plan(env.hazard_map, env.uav_position, env.goal_position)
            step_idx = 1
            steps_since_replan = 0
        if len(current_path) < 2 or step_idx >= len(current_path):
            best_action = 0
            best_distance = float("inf")
            for action, delta in ACTIONS.items():
                candidate = np.clip(env.uav_position + delta, 0, env.grid_size - 1)
                distance = float(np.linalg.norm(candidate - env.goal_position))
                if distance < best_distance:
                    best_distance = distance
                    best_action = action
            action = best_action
            expected_pos = np.clip(env.uav_position + ACTIONS[action], 0, env.grid_size - 1)
        else:
            expected_pos = current_path[step_idx]
            action = AStarPlanner.action_from_move(env.uav_position, expected_pos)
        _, reward, done, info = env.step(action)
        total_reward += reward
        gt_hazard = float(info.get("gt_hazard_level", 0.0))
        if gt_hazard > args.encounter_threshold:
            hazard_hits += 1
            hazard_penalty += gt_hazard
        steps_since_replan += 1
        if len(current_path) >= 2 and np.array_equal(env.uav_position, expected_pos):
            step_idx += 1
        if done:
            break
    return finish_result(env, phase, grid_size, "A*", profile, label, episode, seed, total_reward, hazard_hits, hazard_penalty, start_time)


def run_ppo(env, agent, args, phase, grid_size, profile, label, episode, seed):
    start_time = time.time()
    state = env.reset(seed=seed)
    hazard_hits = 0
    hazard_penalty = 0.0
    total_reward = 0.0
    while env.current_step < env.max_steps:
        action = agent.select_action(state, deterministic=True)
        state, reward, done, info = env.step(action)
        total_reward += reward
        gt_hazard = float(info.get("gt_hazard_level", 0.0))
        if gt_hazard > args.encounter_threshold:
            hazard_hits += 1
            hazard_penalty += gt_hazard
        if done:
            break
    agent.experience_buffer.clear()
    return finish_result(env, phase, grid_size, "PPO", profile, label, episode, seed, total_reward, hazard_hits, hazard_penalty, start_time)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def coerce_row(row):
    ints = {"grid_size", "fire_density_rank", "episode", "seed", "success", "steps", "hazard_encounters", "fire_zone_count"}
    floats = {"path_length", "path_efficiency", "final_distance", "total_reward", "hazard_penalty", "elapsed_time", "avg_fire_intensity", "max_gt_hazard", "high_risk_cell_fraction"}
    out = dict(row)
    for key in ints:
        if key in out:
            out[key] = int(float(out[key]))
    for key in floats:
        if key in out:
            out[key] = float(out[key])
    return out


def read_existing_rows(path):
    if not path.exists():
        return []
    with path.open() as f:
        return [coerce_row(row) for row in csv.DictReader(f)]


def row_key(row):
    return (row["phase"], int(row["grid_size"]), row["profile"], row["method"], int(row["episode"]))


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["phase"], row["grid_size"], row["profile"], row["environment"], row["method"])].append(row)
    output = []
    for (phase, grid_size, profile, environment, method), items in sorted(groups.items()):
        output.append({
            "phase": phase,
            "grid_size": grid_size,
            "profile": profile,
            "environment": environment,
            "fire_density_rank": FIRE_DENSITY_RANK[profile],
            "method": method,
            "episodes": len(items),
            "success_rate": float(np.mean([r["success"] for r in items])),
            "successes": int(sum(r["success"] for r in items)),
            "avg_reward": float(np.mean([r["total_reward"] for r in items])),
            "avg_path_length": float(np.mean([r["path_length"] for r in items])),
            "avg_path_efficiency": float(np.mean([r["path_efficiency"] for r in items])),
            "avg_hazard_encounters": float(np.mean([r["hazard_encounters"] for r in items])),
            "avg_final_distance": float(np.mean([r["final_distance"] for r in items])),
            "avg_steps": float(np.mean([r["steps"] for r in items])),
            "avg_fire_zone_count": float(np.mean([r["fire_zone_count"] for r in items])),
            "avg_fire_intensity": float(np.mean([r["avg_fire_intensity"] for r in items])),
            "avg_high_risk_cell_fraction": float(np.mean([r["high_risk_cell_fraction"] for r in items])),
        })
    return output


def training_rows(training_metrics_path):
    data = json.loads(Path(training_metrics_path).read_text())
    rows = []
    for i in range(len(data["episode_rewards"])):
        rows.append({
            "episode": i + 1,
            "reward": float(data["episode_rewards"][i]),
            "success": int(bool(data["success_episodes"][i])),
            "steps": int(data["episode_lengths"][i]),
            "navigation_efficiency": float(data["navigation_efficiency"][i]),
        })
    return rows


def smooth_same_length(values, window):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    if len(arr) < window:
        window = max(1, len(arr) // 3)
    if window <= 1:
        return arr
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    weights = np.ones(window) / window
    return np.convolve(padded, weights, mode="valid")


def rolling(values, window):
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan)
    if len(arr) >= window:
        out[window - 1:] = np.convolve(arr, np.ones(window) / window, mode="valid")
    return out


def style_for(method):
    if method == "A*":
        return {"color": "#2F6DAE", "marker": "o", "face": "#DCEBFA", "edge": "#1F4E79"}
    return {"color": "#E66101", "marker": "^", "face": "#FAD7B5", "edge": "#9C3D00"}


def plot_training(rows, output_dir, smooth_window, marker_every):
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = np.asarray([r["episode"] for r in rows], dtype=float)
    plot_defs = [
        ("paper1_saved_training_reward_vs_episode.png", "Reward", "Saved PPO Training Reward vs Episode", [r["reward"] for r in rows], False),
        ("paper1_saved_training_success_rate_vs_episode.png", "Success Rate", "Saved PPO Training Success Rate vs Episode", [r["success"] for r in rows], True),
        ("paper1_saved_training_episode_length_vs_episode.png", "Episode Length", "Saved PPO Training Episode Length vs Episode", [r["steps"] for r in rows], False),
        ("paper1_saved_training_efficiency_vs_episode.png", "Navigation Efficiency", "Saved PPO Training Efficiency vs Episode", [r["navigation_efficiency"] for r in rows], False),
    ]
    for name, ylabel, title, values, use_rolling in plot_defs:
        arr = np.asarray(values, dtype=float)
        bg = rolling(arr, 100) if use_rolling else arr
        fg = smooth_same_length(arr, smooth_window)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(episodes, bg, linewidth=0.6, alpha=0.22, color="#4E79A7")
        ax.plot(episodes, fg, linewidth=2.0, color="#1F4E79", marker="o", markevery=marker_every, markersize=7, markerfacecolor="#FFFFFF", markeredgecolor="#1F4E79", markeredgewidth=1.7, label="PPO smoothed")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(left=0)
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / name, dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / name.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_episode_comparisons(rows, output_dir, prefix, smooth_window, marker_every):
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = [(p, l) for p, l in PROFILES if any(r["profile"] == p for r in rows)]
    if not profiles:
        return
    if prefix == "paper1_fire_density":
        metrics = [
            ("success", "Success Indicator / Smoothed Rate", f"{prefix}_success_vs_episode_comparison.png", "Success Rate vs Episode"),
            ("total_reward", "Total Reward", f"{prefix}_reward_vs_episode_comparison.png", "Reward vs Episode"),
            ("path_efficiency", "Path Efficiency", f"{prefix}_path_efficiency_vs_episode_comparison.png", "Path Efficiency vs Episode"),
        ]
    else:
        metrics = [
            ("success", "Success Indicator / Smoothed Rate", f"{prefix}_success_rate_vs_episode.png", "Success Rate vs Episode"),
            ("total_reward", "Total Reward", f"{prefix}_reward_vs_episode.png", "Reward vs Episode"),
            ("path_efficiency", "Path Efficiency", f"{prefix}_path_efficiency_vs_episode.png", "Path Efficiency vs Episode"),
        ]
    for metric, ylabel, filename, title in metrics:
        fig, axes = plt.subplots(1, len(profiles), figsize=(max(7, 5.4 * len(profiles)), 4.3), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, (profile, label) in zip(axes, profiles):
            for method in METHODS:
                subset = sorted([r for r in rows if r["profile"] == profile and r["method"] == method], key=lambda x: x["episode"])
                if not subset:
                    continue
                episodes = np.asarray([r["episode"] for r in subset], dtype=float)
                values = np.asarray([r[metric] for r in subset], dtype=float)
                smoothed = smooth_same_length(values, min(smooth_window, max(3, len(values) // 8)))
                markevery = max(1, min(marker_every, max(1, len(values) // 8)))
                s = style_for(method)
                ax.plot(episodes, values, linewidth=0.55, alpha=0.18, color=s["color"])
                ax.plot(episodes, smoothed, linewidth=2.0, color=s["color"], marker=s["marker"], markevery=markevery, markersize=6.5, markerfacecolor=s["face"], markeredgecolor=s["edge"], markeredgewidth=1.5, label=f"{method} smoothed")
            ax.set_title(label)
            ax.set_xlabel("Episode")
            ax.set_xlim(left=0)
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            axes[-1].legend(loc="best", frameon=True)
        fig.suptitle(title, y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / filename.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_baseline_summary(summary_rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [label for _, label in PROFILES if any(r["environment"] == label for r in summary_rows)]
    if not labels:
        return
    metrics = [
        ("success_rate", "Success Rate", "paper1_final_success_rate_vs_fire_density.png"),
        ("avg_path_efficiency", "Path Efficiency", "paper1_final_path_efficiency_vs_fire_density.png"),
        ("avg_hazard_encounters", "Average Hazard Encounters", "paper1_final_hazard_encounters_vs_fire_density.png"),
        ("avg_reward", "Average Reward", "paper1_final_reward_vs_fire_density.png"),
    ]
    by_key = {(r["environment"], r["method"]): r for r in summary_rows}
    for metric, ylabel, name in metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(labels))
        width = 0.36
        for idx, method in enumerate(METHODS):
            values = [by_key[(label, method)][metric] for label in labels if (label, method) in by_key]
            x_values = [i for i, label in enumerate(labels) if (label, method) in by_key]
            ax.bar(np.asarray(x_values) + (idx - 0.5) * width, values, width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by Fire Density")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / name, dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / name.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_grid_summary(summary_rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("success_rate", "Success Rate", "paper1_grid_success_rate_by_environment.png"),
        ("avg_path_efficiency", "Path Efficiency", "paper1_grid_path_efficiency_by_environment.png"),
        ("avg_hazard_encounters", "Average Hazard Encounters", "paper1_grid_hazard_encounters_by_environment.png"),
    ]
    for metric, ylabel, name in metrics:
        fig, axes = plt.subplots(1, len(PROFILES), figsize=(16, 4.5), sharey=False)
        for ax, (profile, label) in zip(axes, PROFILES):
            for method in METHODS:
                subset = sorted([r for r in summary_rows if r["profile"] == profile and r["method"] == method], key=lambda x: int(x["grid_size"]))
                if not subset:
                    continue
                x = np.asarray([r["grid_size"] for r in subset], dtype=float)
                y = np.asarray([r[metric] for r in subset], dtype=float)
                s = style_for(method)
                ax.plot(x, y, linewidth=2.0, color=s["color"], marker=s["marker"], markersize=7, markerfacecolor=s["face"], markeredgecolor=s["edge"], markeredgewidth=1.5, label=method)
            ax.set_title(label)
            ax.set_xlabel("Grid Size")
            ax.set_xticks(sorted(set(int(r["grid_size"]) for r in summary_rows)))
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(output_dir / name, dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / name.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)


def save_environment_visual(env, astar_path, ppo_path, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(env.aerial_image)
    axes[0].set_title(title)
    axes[0].axis("off")
    im = axes[1].imshow(env.gt_hazard_map, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("Ground-Truth Fire Risk")
    axes[1].set_xlim(-0.5, env.grid_size - 0.5)
    axes[1].set_ylim(env.grid_size - 0.5, -0.5)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(env.gt_hazard_map, cmap="Reds", vmin=0, vmax=1, alpha=0.75)
    if len(astar_path) > 1:
        arr = np.asarray(astar_path)
        axes[2].plot(arr[:, 1], arr[:, 0], color="#2F6DAE", linewidth=2.2, label="A*")
    if len(ppo_path) > 1:
        arr = np.asarray(ppo_path)
        axes[2].plot(arr[:, 1], arr[:, 0], color="#E66101", linewidth=2.2, label="PPO")
    axes[2].scatter([2], [2], c="lime", s=65, edgecolors="black", label="Start")
    axes[2].scatter([env.grid_size - 3], [env.grid_size - 3], c="yellow", s=90, marker="*", edgecolors="black", label="Goal")
    axes[2].set_title("Representative Trajectories")
    axes[2].set_xlim(-0.5, env.grid_size - 0.5)
    axes[2].set_ylim(env.grid_size - 0.5, -0.5)
    axes[2].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def training_summary(rows):
    output = []
    for window in [100, 300, 1000]:
        items = rows[-window:]
        output.append({
            "source": "Saved PPO Training Summary",
            "window": f"last_{window}",
            "episodes": len(items),
            "success_rate": float(np.mean([r["success"] for r in items])),
            "avg_reward": float(np.mean([r["reward"] for r in items])),
            "avg_steps": float(np.mean([r["steps"] for r in items])),
            "avg_navigation_efficiency": float(np.mean([r["navigation_efficiency"] for r in items])),
        })
    output.append({
        "source": "Saved PPO Training Summary",
        "window": "all_3000",
        "episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "avg_reward": float(np.mean([r["reward"] for r in rows])),
        "avg_steps": float(np.mean([r["steps"] for r in rows])),
        "avg_navigation_efficiency": float(np.mean([r["navigation_efficiency"] for r in rows])),
    })
    return output


def parameter_rows(args, grid_sizes):
    return [
        {"parameter": "project", "value": "Disaster-Aware UAV Path Planning with Vision-Integrated PPO"},
        {"parameter": "fire_environments", "value": "light/sparse fire, moderate fire, dense/heavy fire"},
        {"parameter": "baseline_grid", "value": f"{args.baseline_grid} x {args.baseline_grid}"},
        {"parameter": "grid_sweep_sizes", "value": ",".join(str(g) for g in grid_sizes)},
        {"parameter": "baseline_episodes_per_environment_method", "value": args.baseline_episodes},
        {"parameter": "grid_sweep_episodes_per_environment_method", "value": args.grid_sweep_episodes},
        {"parameter": "classifier_checkpoint", "value": args.classifier},
        {"parameter": "classifier_usage", "value": "real CNN inference inside the navigation environment"},
        {"parameter": "ppo_checkpoint", "value": args.ppo_model},
        {"parameter": "saved_training_metrics", "value": args.training_metrics},
        {"parameter": "ppo_action_space", "value": "8 discrete moves"},
        {"parameter": "max_steps", "value": args.max_steps},
        {"parameter": "encounter_threshold", "value": args.encounter_threshold},
        {"parameter": "termination_threshold", "value": args.termination_threshold},
        {"parameter": "perception_noise", "value": args.perception_noise},
        {"parameter": "observation_radius", "value": args.observation_radius},
        {"parameter": "confidence_decay", "value": args.confidence_decay},
        {"parameter": "aerial_cell_px", "value": args.aerial_cell_px},
        {"parameter": "a_star_replan_frequency", "value": args.replan_frequency},
        {"parameter": "smoothing_window", "value": args.smooth_window},
        {"parameter": "marker_spacing", "value": args.marker_every},
    ]


def legacy_episode_rows(rows):
    legacy_keys = [
        "profile",
        "environment",
        "method",
        "episode",
        "seed",
        "success",
        "steps",
        "path_length",
        "path_efficiency",
        "final_distance",
        "hazard_encounters",
        "total_reward",
        "hazard_penalty",
        "elapsed_time",
        "fire_zone_count",
        "avg_fire_intensity",
        "max_gt_hazard",
        "high_risk_cell_fraction",
    ]
    return [{key: row[key] for key in legacy_keys if key in row} for row in rows]


def legacy_summary_rows(rows):
    legacy_rows = []
    for row in rows:
        legacy_rows.append({
            "profile": row["profile"],
            "environment": row["environment"],
            "method": row["method"],
            "episodes": row["episodes"],
            "success_rate": row["success_rate"],
            "successes": row["successes"],
            "avg_steps": row["avg_steps"],
            "avg_path_length": row["avg_path_length"],
            "avg_path_efficiency": row["avg_path_efficiency"],
            "avg_final_distance": row["avg_final_distance"],
            "avg_hazard_encounters": row["avg_hazard_encounters"],
            "avg_total_reward": row["avg_reward"],
            "avg_fire_zone_count": row["avg_fire_zone_count"],
            "avg_fire_intensity": row["avg_fire_intensity"],
            "avg_high_risk_cell_fraction": row["avg_high_risk_cell_fraction"],
        })
    return legacy_rows


def run_condition(args, agent, phase, profile, label, grid_size, episodes, seed_offset, figures_dir, aggregate_rows, aggregate_path, existing_keys):
    planner = AStarPlanner(grid_size, diag=True)
    env = make_env(args, profile, grid_size)
    rows = []
    visual_path = None
    for episode in range(1, episodes + 1):
        seed = args.seed_base + seed_offset + episode
        astar_path = None
        ppo_path = None
        if (phase, grid_size, profile, "A*", episode) not in existing_keys:
            astar_result = run_astar(env, planner, args, phase, grid_size, profile, label, episode, seed)
            rows.append(astar_result)
            aggregate_rows.append(astar_result)
            existing_keys.add(row_key(astar_result))
            astar_path = np.asarray(env.path_history).copy()
        if (phase, grid_size, profile, "PPO", episode) not in existing_keys:
            ppo_result = run_ppo(env, agent, args, phase, grid_size, profile, label, episode, seed)
            rows.append(ppo_result)
            aggregate_rows.append(ppo_result)
            existing_keys.add(row_key(ppo_result))
            ppo_path = np.asarray(env.path_history).copy()
        if rows:
            write_csv(aggregate_path, aggregate_rows)
        if episode == 1 and grid_size == args.baseline_grid and astar_path is not None and ppo_path is not None:
            name = "paper1_" + label.lower().replace("/", "_").replace(" ", "_").replace("__", "_") + f"_{phase}_environment.png"
            visual_path = figures_dir / name
            save_environment_visual(env, astar_path, ppo_path, f"{label} ({grid_size} x {grid_size})", visual_path)
        clear_device_cache()
    return rows, visual_path


def run_evaluation(args, profiles, grid_sizes):
    root = Path(args.output_dir)
    metrics_dir = root / "metrics"
    figures_dir = root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    agent = PPONavigationAgent((args.baseline_grid, args.baseline_grid, 4), 8)
    if not agent.load_model(args.ppo_model):
        raise FileNotFoundError(args.ppo_model)

    baseline_path = metrics_dir / "paper1_final_baseline_episodes.csv"
    grid_path = metrics_dir / "paper1_grid_sweep_episodes.csv"
    baseline_rows = read_existing_rows(baseline_path)
    grid_rows = read_existing_rows(grid_path)
    baseline_keys = {row_key(r) for r in baseline_rows}
    grid_keys = {row_key(r) for r in grid_rows}
    visual_paths = {}

    if args.package in {"validation", "final", "fire_density"}:
        for profile_index, (profile, label) in enumerate(profiles):
            episodes = args.validation_episodes if args.package == "validation" else args.baseline_episodes
            rows, visual = run_condition(args, agent, "baseline_50x50", profile, label, args.baseline_grid, episodes, profile_index * 100000, figures_dir, baseline_rows, baseline_path, baseline_keys)
            if visual:
                visual_paths[f"{profile}_baseline"] = str(visual)

    if args.package == "final":
        for profile_index, (profile, label) in enumerate(profiles):
            for grid_index, grid_size in enumerate(grid_sizes):
                rows, visual = run_condition(args, agent, "grid_sweep", profile, label, grid_size, args.grid_sweep_episodes, 1000000 + profile_index * 100000 + grid_index * 10000, figures_dir, grid_rows, grid_path, grid_keys)
                if visual:
                    visual_paths[f"{profile}_grid_{grid_size}"] = str(visual)

    baseline_summary = summarize(baseline_rows)
    grid_summary = summarize(grid_rows)
    t_rows = training_rows(args.training_metrics)
    t_summary = training_summary(t_rows)

    write_csv(metrics_dir / "paper1_final_baseline_episodes.csv", baseline_rows)
    write_csv(metrics_dir / "paper1_final_baseline_summary.csv", baseline_summary)
    write_csv(metrics_dir / "paper1_grid_sweep_episodes.csv", grid_rows)
    write_csv(metrics_dir / "paper1_grid_sweep_summary.csv", grid_summary)
    write_csv(metrics_dir / "paper1_saved_ppo_training_summary.csv", t_summary)
    write_csv(metrics_dir / "paper1_saved_ppo_training_episode_metrics_3000.csv", t_rows)
    write_csv(metrics_dir / "paper1_experiment_parameters.csv", parameter_rows(args, grid_sizes))
    if args.package == "fire_density":
        write_csv(metrics_dir / "paper1_episode_results.csv", legacy_episode_rows(baseline_rows))
        write_csv(metrics_dir / "paper1_environment_summary.csv", legacy_summary_rows(baseline_summary))
        write_csv(metrics_dir / "paper1_training_3000_summary.csv", [
            {key: value for key, value in row.items() if key != "source"} for row in t_summary
        ])
        write_csv(metrics_dir / "paper1_training_episode_metrics_3000.csv", t_rows)

    if baseline_rows:
        plot_episode_comparisons(baseline_rows, figures_dir, "paper1_final_baseline", args.smooth_window, args.marker_every)
        if args.package == "fire_density":
            plot_episode_comparisons(baseline_rows, figures_dir, "paper1_fire_density", args.smooth_window, args.marker_every)
        plot_baseline_summary(baseline_summary, figures_dir)
    if grid_rows:
        plot_grid_summary(grid_summary, figures_dir)
    plot_training(t_rows, figures_dir, args.smooth_window, args.marker_every)

    config = vars(args).copy()
    config["profiles"] = [{"id": p, "label": l} for p, l in profiles]
    config["grid_sizes"] = grid_sizes
    config["created_at"] = datetime.now().isoformat()
    config["visuals"] = visual_paths
    (metrics_dir / "paper1_config.json").write_text(json.dumps(config, indent=2))
    write_markdown(root / "PAPER1_FINAL_UPDATE_SUMMARY.md", baseline_summary, grid_summary, t_summary, args, grid_sizes)

    return {
        "output_dir": str(root),
        "baseline_episode_rows": len(baseline_rows),
        "baseline_summary_rows": len(baseline_summary),
        "grid_episode_rows": len(grid_rows),
        "grid_summary_rows": len(grid_summary),
        "saved_training_rows": len(t_rows),
        "figures_dir": str(figures_dir),
    }


def write_markdown(path, baseline_summary, grid_summary, training_summary_rows, args, grid_sizes):
    lines = [
        "# Paper 1 Post-Meeting Results Package",
        "",
        "## Purpose",
        "",
        "This package responds to the post-meeting request to separate the Paper 1 results into light/sparse fire, moderate fire, and dense/heavy fire settings, while keeping the earlier 50-episode result as validation rather than final paper evidence.",
        "",
        "## Main 50 x 50 Evaluation Table",
        "",
        "| Environment | Method | Episodes | Success Rate | Avg Reward | Avg Path Length | Path Efficiency | Hazard Encounters | Final Distance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in baseline_summary:
        lines.append(f"| {r['environment']} | {r['method']} | {r['episodes']} | {r['success_rate']:.3f} | {r['avg_reward']:.3f} | {r['avg_path_length']:.3f} | {r['avg_path_efficiency']:.3f} | {r['avg_hazard_encounters']:.3f} | {r['avg_final_distance']:.3f} |")
    lines.extend([
        "",
        "## Grid-Size Generalization Table",
        "",
        "| Environment | Grid Size | Method | Episodes | Success Rate | Avg Reward | Path Efficiency | Hazard Encounters | Final Distance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in grid_summary:
        lines.append(f"| {r['environment']} | {r['grid_size']} | {r['method']} | {r['episodes']} | {r['success_rate']:.3f} | {r['avg_reward']:.3f} | {r['avg_path_efficiency']:.3f} | {r['avg_hazard_encounters']:.3f} | {r['avg_final_distance']:.3f} |")
    lines.extend([
        "",
        "## Saved PPO Training Summary",
        "",
        "This table is the saved PPO training-history record from the checkpoint metrics file. It is not the same as the three-environment evaluation table above.",
        "",
        "| Source | Window | Episodes | Success Rate | Avg Reward | Avg Steps | Avg Navigation Efficiency |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for r in training_summary_rows:
        lines.append(f"| {r['source']} | {r['window']} | {r['episodes']} | {r['success_rate']:.3f} | {r['avg_reward']:.3f} | {r['avg_steps']:.3f} | {r['avg_navigation_efficiency']:.3f} |")
    lines.extend([
        "",
        "## Figures",
        "",
        "- `figures/paper1_final_success_rate_vs_fire_density.png`",
        "- `figures/paper1_grid_success_rate_by_environment.png`",
        "- `figures/paper1_final_baseline_reward_vs_episode.png`",
        "- `figures/paper1_final_baseline_success_rate_vs_episode.png`",
        "- `figures/paper1_grid_path_efficiency_by_environment.png`",
        "- `figures/paper1_grid_hazard_encounters_by_environment.png`",
        "",
        "## Experiment Setup",
        "",
        f"The baseline grid is `{args.baseline_grid} x {args.baseline_grid}`. The grid-size sweep uses `{', '.join(str(g) + ' x ' + str(g) for g in grid_sizes)}`.",
        "",
        "PPO and A* are evaluated on the same fixed seeded fire-layout variants for each environment condition. The navigation environment uses the real CNN classifier during evaluation.",
        "",
        "## Interpretation",
        "",
        "The final paper table should use the main 50 x 50 evaluation results, while the grid-size table should be presented as a generalization and sensitivity analysis.",
    ])
    path.write_text("\n".join(lines) + "\n")


def validate_outputs(root, package):
    metrics_dir = Path(root) / "metrics"
    checks = {}
    baseline_rows = list(csv.DictReader((metrics_dir / "paper1_final_baseline_episodes.csv").open())) if (metrics_dir / "paper1_final_baseline_episodes.csv").exists() else []
    grid_rows = list(csv.DictReader((metrics_dir / "paper1_grid_sweep_episodes.csv").open())) if (metrics_dir / "paper1_grid_sweep_episodes.csv").exists() else []
    baseline_summary = list(csv.DictReader((metrics_dir / "paper1_final_baseline_summary.csv").open())) if (metrics_dir / "paper1_final_baseline_summary.csv").exists() else []
    grid_summary = list(csv.DictReader((metrics_dir / "paper1_grid_sweep_summary.csv").open())) if (metrics_dir / "paper1_grid_sweep_summary.csv").exists() else []
    all_rows = baseline_rows + grid_rows + baseline_summary + grid_summary
    bad = []
    for row in all_rows:
        for key, value in row.items():
            try:
                val = float(value)
            except ValueError:
                continue
            if not np.isfinite(val):
                bad.append((key, value))
    checks["baseline_episode_rows"] = len(baseline_rows)
    checks["baseline_summary_rows"] = len(baseline_summary)
    checks["grid_episode_rows"] = len(grid_rows)
    checks["grid_summary_rows"] = len(grid_summary)
    checks["nonfinite_values"] = len(bad)
    checks["baseline_environments"] = sorted(set(r["environment"] for r in baseline_rows))
    checks["baseline_methods"] = sorted(set(r["method"] for r in baseline_rows))
    checks["grid_sizes"] = sorted(set(r["grid_size"] for r in grid_rows), key=lambda x: int(x)) if grid_rows else []
    if package == "validation":
        checks["expected_baseline_episode_rows"] = 10
    if package == "final":
        checks["expected_baseline_episode_rows"] = 18000
        checks["expected_grid_episode_rows"] = 3600
        checks["expected_baseline_summary_rows"] = 6
        checks["expected_grid_summary_rows"] = 36
    if package == "fire_density":
        checks["expected_baseline_summary_rows"] = 6
    (metrics_dir / "paper1_validation_checks.json").write_text(json.dumps(checks, indent=2))
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", choices=["validation", "final", "fire_density"], default="validation")
    parser.add_argument("--baseline_episodes", type=int, default=3000)
    parser.add_argument("--grid_sweep_episodes", type=int, default=100)
    parser.add_argument("--validation_episodes", type=int, default=5)
    parser.add_argument("--baseline_grid", type=int, default=50)
    parser.add_argument("--grid_sizes", default="40,50,60,70,85,100")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--classifier", default="checkpoints/checkpoint.pth")
    parser.add_argument("--ppo_model", default="navigation/navigation_models/uav_navigation_3000.pth")
    parser.add_argument("--training_metrics", default="navigation/navigation_models/training_metrics_3000.json")
    parser.add_argument("--output_dir", default="navigation/comparison_results/paper1_post_meeting_final")
    parser.add_argument("--seed_base", type=int, default=20260529)
    parser.add_argument("--encounter_threshold", type=float, default=0.2)
    parser.add_argument("--termination_threshold", type=float, default=0.9)
    parser.add_argument("--perception_noise", type=float, default=0.0)
    parser.add_argument("--replan_frequency", type=int, default=0)
    parser.add_argument("--confidence_decay", type=float, default=0.95)
    parser.add_argument("--observation_radius", type=int, default=2)
    parser.add_argument("--aerial_cell_px", type=int, default=8)
    parser.add_argument("--smooth_window", type=int, default=300)
    parser.add_argument("--marker_every", type=int, default=200)
    args = parser.parse_args()

    grid_sizes = parsed_grid_sizes(args.grid_sizes)
    profiles = PROFILES[:1] if args.package == "validation" else PROFILES
    result = run_evaluation(args, profiles, grid_sizes)
    result["checks"] = validate_outputs(args.output_dir, args.package)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
