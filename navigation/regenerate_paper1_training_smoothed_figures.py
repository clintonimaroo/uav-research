import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_training_rows(path):
    with Path(path).open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return sorted(rows, key=lambda row: int(float(row["episode"])))


def trailing_rolling(values, window):
    arr = np.asarray(values, dtype=float)
    output = np.zeros(len(arr), dtype=float)
    cumsum = np.cumsum(arr, dtype=float)
    for idx in range(len(arr)):
        start = max(0, idx - window + 1)
        total = cumsum[idx] - (cumsum[start - 1] if start > 0 else 0.0)
        output[idx] = total / (idx - start + 1)
    return output


def save_old_style_plot(episodes, values, ylabel, title, output_base, raw_window=100, smooth_window=300):
    raw_smoothed = trailing_rolling(values, raw_window)
    smoothed = trailing_rolling(values, smooth_window)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(episodes, raw_smoothed, color="#1f77b4", linewidth=1.2, alpha=0.22)
    ax.plot(
        episodes,
        smoothed,
        color="#0B4F93",
        linewidth=2.6,
        marker="o",
        markevery=max(1, len(episodes) // 12),
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="#0B4F93",
        markeredgewidth=1.8,
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-csv",
        default="navigation/comparison_results/paper1_fire_density_final/metrics/paper1_training_episode_metrics_3000.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="navigation/comparison_results/paper1_fire_density_final/figures",
    )
    parser.add_argument("--raw-window", type=int, default=100)
    parser.add_argument("--smooth-window", type=int, default=300)
    args = parser.parse_args()

    rows = read_training_rows(args.training_csv)
    episodes = np.asarray([int(float(row["episode"])) for row in rows], dtype=int)
    output_dir = Path(args.output_dir)

    save_old_style_plot(
        episodes,
        [float(row["reward"]) for row in rows],
        "Reward",
        "Reward vs Episode",
        output_dir / "paper1_reward_vs_episode_smoothed",
        args.raw_window,
        args.smooth_window,
    )
    save_old_style_plot(
        episodes,
        [float(row["success"]) for row in rows],
        "Success Rate",
        "Success Rate vs Episode",
        output_dir / "paper1_success_rate_vs_episode_smoothed",
        args.raw_window,
        args.smooth_window,
    )
    save_old_style_plot(
        episodes,
        [float(row["navigation_efficiency"]) for row in rows],
        "Path Efficiency",
        "Path Efficiency vs Episode",
        output_dir / "paper1_path_efficiency_vs_episode_smoothed",
        args.raw_window,
        args.smooth_window,
    )
    save_old_style_plot(
        episodes,
        [float(row["navigation_efficiency"]) for row in rows],
        "Navigation Efficiency",
        "Navigation Efficiency vs Episode",
        output_dir / "paper1_navigation_efficiency_vs_episode_smoothed",
        args.raw_window,
        args.smooth_window,
    )


if __name__ == "__main__":
    main()
