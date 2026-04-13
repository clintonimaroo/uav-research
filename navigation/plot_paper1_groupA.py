import argparse
import csv
import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BASELINE = {
    "grid": 50,
    "encounter_threshold": 0.2,
    "termination_threshold": 0.9,
    "perception_noise": 0.0,
    "replan_frequency": 0,
    "confidence_decay": 0.95,
    "observation_radius": 2,
}


def _to_float(row: Dict, key: str) -> float:
    return float(row[key])


def _to_int(row: Dict, key: str) -> int:
    return int(float(row[key]))


def _ensure_columns(rows: List[Dict], required: List[str], context: str):
    if not rows:
        raise ValueError(f"{context}: no rows available.")
    missing = [col for col in required if col not in rows[0]]
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


def _load_csv(path: str) -> List[Dict]:
    with open(path, "r", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _rolling_mean(values: List[float], window: int) -> Tuple[List[int], List[float]]:
    if len(values) < window:
        return [], []
    output = []
    x_values = []
    for i in range(window - 1, len(values)):
        output.append(float(np.mean(values[i - window + 1 : i + 1])))
        x_values.append(i + 1)
    return x_values, output


def _plot_single_line(x, y, xlabel, ylabel, title, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_bar(methods, values, xlabel, ylabel, title, output_path):
    plt.figure(figsize=(8, 6))
    plt.bar(methods, values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _matches_baseline(row: Dict, baseline: Dict, ignore_key: str, tol: float = 1e-9) -> bool:
    for key, value in baseline.items():
        if key == ignore_key:
            continue
        if abs(_to_float(row, key) - float(value)) > tol:
            return False
    return True


def _plot_multi_line(rows: List[Dict], x_key: str, y_key: str, y_label: str, title: str, output_path: str):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append((_to_float(row, x_key), _to_float(row, y_key)))

    if len(rows) == 0:
        print(f"Warning: no rows for {title}, skipping.")
        return

    unique_x = sorted({_to_float(r, x_key) for r in rows})
    if len(unique_x) < 2:
        print(f"Warning: {title} has fewer than 2 x-values, skipping.")
        return

    plt.figure(figsize=(10, 6))
    for method in sorted(grouped.keys()):
        # Average repeated x-values across runs for cleaner sweep curves.
        by_x = defaultdict(list)
        for x_val, y_val in grouped[method]:
            by_x[x_val].append(y_val)
        points = sorted((x_val, float(np.mean(y_vals))) for x_val, y_vals in by_x.items())
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.plot(x, y, marker="o", label=method)

    plt.xlabel(x_key.replace("_", " ").title())
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _find_latest_training_csv(explicit_path: str = None) -> str:
    if explicit_path:
        return explicit_path
    candidates = sorted(glob.glob("navigation_models/training_episode_metrics_*.csv"))
    return candidates[-1] if candidates else ""


def _plot_training_group_a(training_csv: str, output_dir: str, rolling_window: int):
    if not training_csv:
        print("Warning: no training CSV found; skipping training episode plots.")
        return
    rows = _load_csv(training_csv)
    _ensure_columns(
        rows,
        ["episode", "reward", "success", "steps", "navigation_efficiency"],
        "Training plots",
    )
    rows = sorted(rows, key=lambda r: _to_int(r, "episode"))

    episodes = [_to_int(r, "episode") for r in rows]
    rewards = [_to_float(r, "reward") for r in rows]
    success = [_to_int(r, "success") for r in rows]
    steps = [_to_int(r, "steps") for r in rows]
    efficiency = [_to_float(r, "navigation_efficiency") for r in rows]

    _plot_single_line(
        episodes,
        rewards,
        "Episode (episodes)",
        "Reward",
        "Reward vs Episode",
        os.path.join(output_dir, "reward_vs_episode.png"),
    )
    _plot_single_line(
        episodes,
        success,
        "Episode (episodes)",
        "Success Rate",
        "Success Indicator (0/1) vs Episode",
        os.path.join(output_dir, "success_indicator_vs_episode.png"),
    )
    rolling_x, rolling_success = _rolling_mean(success, rolling_window)
    if rolling_x:
        _plot_single_line(
            rolling_x,
            rolling_success,
            "Episode (episodes)",
            "Success Rate",
            f"Success Rate vs Episode ({rolling_window}-episode rolling)",
            os.path.join(output_dir, "success_rate_vs_episode_rolling.png"),
        )
    else:
        print(
            f"Warning: only {len(success)} training episodes; "
            f"cannot compute {rolling_window}-episode rolling success."
        )
    _plot_single_line(
        episodes,
        steps,
        "Episode (episodes)",
        "Steps",
        "Episode Length vs Episode",
        os.path.join(output_dir, "episode_length_vs_episode.png"),
    )
    _plot_single_line(
        episodes,
        efficiency,
        "Episode (episodes)",
        "Efficiency Score",
        "Navigation Efficiency vs Episode",
        os.path.join(output_dir, "navigation_efficiency_vs_episode.png"),
    )


def _latest_run_rows(summary_rows: List[Dict]) -> List[Dict]:
    run_order = {}
    for idx, row in enumerate(summary_rows):
        run_order[row["run_id"]] = idx
    latest_run_id = max(run_order, key=lambda rid: run_order[rid])
    return [r for r in summary_rows if r["run_id"] == latest_run_id]


def _plot_method_bars(summary_rows: List[Dict], output_dir: str):
    _ensure_columns(
        summary_rows,
        [
            "run_id",
            "method",
            "success_rate",
            "avg_steps",
            "avg_path_len",
            "avg_final_dist",
            "avg_hazard_encounters",
            "avg_total_reward",
        ],
        "Method bar plots",
    )

    latest_rows = _latest_run_rows(summary_rows)
    latest_rows = sorted(latest_rows, key=lambda r: 0 if r["method"] == "A*" else 1)
    methods = [r["method"] for r in latest_rows]

    _plot_bar(
        methods,
        [_to_float(r, "success_rate") for r in latest_rows],
        "Method",
        "Success Rate",
        "Success Rate by Method",
        os.path.join(output_dir, "bar_success_rate_by_method.png"),
    )
    _plot_bar(
        methods,
        [_to_float(r, "avg_steps") for r in latest_rows],
        "Method",
        "Average Steps",
        "Average Steps by Method",
        os.path.join(output_dir, "bar_average_steps_by_method.png"),
    )
    _plot_bar(
        methods,
        [_to_float(r, "avg_path_len") for r in latest_rows],
        "Method",
        "Average Path Length",
        "Average Path Length by Method",
        os.path.join(output_dir, "bar_average_path_length_by_method.png"),
    )
    _plot_bar(
        methods,
        [_to_float(r, "avg_final_dist") for r in latest_rows],
        "Method",
        "Final Distance",
        "Final Distance by Method",
        os.path.join(output_dir, "bar_final_distance_by_method.png"),
    )
    _plot_bar(
        methods,
        [_to_float(r, "avg_hazard_encounters") for r in latest_rows],
        "Method",
        "Hazard Encounters",
        "Hazard Encounters by Method",
        os.path.join(output_dir, "bar_hazard_encounters_by_method.png"),
    )
    _plot_bar(
        methods,
        [_to_float(r, "avg_total_reward") for r in latest_rows],
        "Method",
        "Total Reward",
        "Total Reward by Method",
        os.path.join(output_dir, "bar_total_reward_by_method.png"),
    )


def _plot_sweep_lines(summary_rows: List[Dict], output_dir: str):
    _ensure_columns(
        summary_rows,
        [
            "method",
            "success_rate",
            "avg_steps",
            "avg_hazard_encounters",
            "grid",
            "encounter_threshold",
            "termination_threshold",
            "perception_noise",
            "replan_frequency",
            "confidence_decay",
            "observation_radius",
        ],
        "Sweep plots",
    )

    encounter_rows = [
        r
        for r in summary_rows
        if _matches_baseline(r, DEFAULT_BASELINE, "encounter_threshold")
    ]
    termination_rows = [
        r
        for r in summary_rows
        if _matches_baseline(r, DEFAULT_BASELINE, "termination_threshold")
    ]
    noise_rows = [
        r
        for r in summary_rows
        if _matches_baseline(r, DEFAULT_BASELINE, "perception_noise")
    ]
    grid_rows = [
        r
        for r in summary_rows
        if _matches_baseline(r, DEFAULT_BASELINE, "grid")
    ]
    replan_rows = [
        r
        for r in summary_rows
        if _matches_baseline(r, DEFAULT_BASELINE, "replan_frequency")
    ]
    confidence_decay_rows = [
        r
        for r in summary_rows
        if _matches_baseline(r, DEFAULT_BASELINE, "confidence_decay")
    ]
    observation_radius_rows = [
        r
        for r in summary_rows
        if _matches_baseline(r, DEFAULT_BASELINE, "observation_radius")
    ]

    _plot_multi_line(
        encounter_rows,
        "encounter_threshold",
        "success_rate",
        "Success Rate",
        "Success Rate vs Encounter Threshold",
        os.path.join(output_dir, "line_success_rate_vs_encounter_threshold.png"),
    )
    _plot_multi_line(
        encounter_rows,
        "encounter_threshold",
        "avg_hazard_encounters",
        "Hazard Encounters",
        "Hazard Encounters vs Encounter Threshold",
        os.path.join(output_dir, "line_hazard_encounters_vs_encounter_threshold.png"),
    )
    _plot_multi_line(
        termination_rows,
        "termination_threshold",
        "success_rate",
        "Success Rate",
        "Success Rate vs Termination Threshold",
        os.path.join(output_dir, "line_success_rate_vs_termination_threshold.png"),
    )
    _plot_multi_line(
        noise_rows,
        "perception_noise",
        "success_rate",
        "Success Rate",
        "Success Rate vs Perception Noise",
        os.path.join(output_dir, "line_success_rate_vs_perception_noise.png"),
    )
    _plot_multi_line(
        noise_rows,
        "perception_noise",
        "avg_hazard_encounters",
        "Hazard Encounters",
        "Hazard Encounters vs Perception Noise",
        os.path.join(output_dir, "line_hazard_encounters_vs_perception_noise.png"),
    )
    _plot_multi_line(
        grid_rows,
        "grid",
        "success_rate",
        "Success Rate",
        "Success Rate vs Grid Size",
        os.path.join(output_dir, "line_success_rate_vs_grid_size.png"),
    )
    _plot_multi_line(
        grid_rows,
        "grid",
        "avg_steps",
        "Average Steps",
        "Average Steps vs Grid Size",
        os.path.join(output_dir, "line_average_steps_vs_grid_size.png"),
    )
    _plot_multi_line(
        replan_rows,
        "replan_frequency",
        "success_rate",
        "Success Rate",
        "Success Rate vs A* Replan Frequency",
        os.path.join(output_dir, "line_success_rate_vs_replan_frequency.png"),
    )
    _plot_multi_line(
        replan_rows,
        "replan_frequency",
        "avg_hazard_encounters",
        "Hazard Encounters",
        "Hazard Encounters vs A* Replan Frequency",
        os.path.join(output_dir, "line_hazard_encounters_vs_replan_frequency.png"),
    )
    _plot_multi_line(
        confidence_decay_rows,
        "confidence_decay",
        "success_rate",
        "Success Rate",
        "Success Rate vs Confidence Decay",
        os.path.join(output_dir, "line_success_rate_vs_confidence_decay.png"),
    )
    _plot_multi_line(
        observation_radius_rows,
        "observation_radius",
        "success_rate",
        "Success Rate",
        "Success Rate vs Observation Radius",
        os.path.join(output_dir, "line_success_rate_vs_observation_radius.png"),
    )


def generate_all_figures(
    comparison_root: str,
    output_dir: str,
    training_csv: str = "",
    rolling_window: int = 100,
    exclude_verify_runs: bool = True,
) -> Tuple[str, str]:
    """
    Regenerate all Paper 1 Group A-style figures from saved comparison CSVs
    and optional PPO training episode metrics.

    Returns (output_dir, training_csv_path_used).
    """
    os.makedirs(output_dir, exist_ok=True)

    training_csv_used = _find_latest_training_csv(training_csv)
    _plot_training_group_a(training_csv_used, output_dir, rolling_window)

    summary_files = glob.glob(os.path.join(comparison_root, "**", "run_summary.csv"), recursive=True)
    if exclude_verify_runs:
        summary_files = [p for p in summary_files if f"{os.sep}verify_" not in p]
    if not summary_files:
        raise ValueError(f"No run_summary.csv files found under: {comparison_root}")

    summary_rows: List[Dict] = []
    for csv_path in summary_files:
        summary_rows.extend(_load_csv(csv_path))

    baseline_csv = os.path.join(comparison_root, "baseline", "run_summary.csv")
    if os.path.isfile(baseline_csv):
        bar_rows = _load_csv(baseline_csv)
    else:
        bar_rows = _latest_run_rows(summary_rows)

    _plot_method_bars(bar_rows, output_dir)
    _plot_sweep_lines(summary_rows, output_dir)

    print(f"Saved Group A/B plots to: {output_dir}")
    if training_csv_used:
        print(f"Training CSV used: {training_csv_used}")
    else:
        print("Training CSV was not found; training line plots were skipped.")

    return output_dir, training_csv_used


def main():
    parser = argparse.ArgumentParser(description="Generate Paper 1 Group A/B plots from CSV files")
    parser.add_argument(
        "--comparison_root",
        type=str,
        default="comparison_results",
        help="Root directory containing comparison run folders",
    )
    parser.add_argument(
        "--training_csv",
        type=str,
        default="",
        help="Optional explicit path to training episode CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_results/paper1_groupA_figures",
        help="Directory where figures are saved",
    )
    parser.add_argument(
        "--rolling_window",
        type=int,
        default=100,
        help="Rolling window for success-rate training plot",
    )
    parser.add_argument(
        "--include-verify-runs",
        action="store_true",
        help="Include comparison_results/verify_* folders in sweep curves (default: exclude)",
    )
    args = parser.parse_args()

    generate_all_figures(
        comparison_root=args.comparison_root,
        output_dir=args.output_dir,
        training_csv=args.training_csv,
        rolling_window=args.rolling_window,
        exclude_verify_runs=not args.include_verify_runs,
    )


if __name__ == "__main__":
    main()
