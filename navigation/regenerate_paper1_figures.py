"""
Regenerate all Paper 1 Group A figures from saved experiment outputs.

What gets plotted
-----------------
* Training curves (if a training_episode_metrics_*.csv exists under navigation_models/,
  or you pass --training-csv): reward, success indicator, rolling success, steps, efficiency.
* Baseline A* vs PPO bar charts from comparison_results/baseline/run_summary.csv.
* Parameter sweep line plots from every other run_summary.csv under comparison_results/
  (grid, encounter, termination, noise, replan, confidence decay, observation radius).

Requirements: numpy, matplotlib (see repository requirements.txt).

Usage (recommended: run from this directory)
----------------------------------------------
    cd navigation
    python regenerate_paper1_figures.py

    # Custom locations (paths can be absolute or relative to cwd)
    python regenerate_paper1_figures.py \\
        --comparison-root /path/to/comparison_results \\
        --output-dir /path/to/paper1_groupA_figures \\
        --training-csv /path/to/training_episode_metrics_YYYY-MM-DD_HH-MM-SS.csv

    # From repo root (script cd's into navigation/ automatically)
    python navigation/regenerate_paper1_figures.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")


def _navigation_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    nav_dir = _navigation_dir()
    os.chdir(nav_dir)
    if nav_dir not in sys.path:
        sys.path.insert(0, nav_dir)

    import plot_paper1_groupA as plots  # noqa: E402

    parser = argparse.ArgumentParser(
        description="Regenerate Paper 1 Group A figures from comparison_results CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--comparison-root",
        default="comparison_results",
        help="Folder tree containing baseline/ and sweep subfolders with run_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_results/paper1_groupA_figures",
        help="Where PNG figures are written (created if missing)",
    )
    parser.add_argument(
        "--training-csv",
        default="",
        help="Training metrics CSV (columns: episode, reward, success, steps, navigation_efficiency). "
        "Default: latest navigation_models/training_episode_metrics_*.csv",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=100,
        help="Rolling window for success_rate_vs_episode_rolling.png",
    )
    parser.add_argument(
        "--include-verify-runs",
        action="store_true",
        help="Include comparison_results/verify_* in sweep curves (default: exclude)",
    )
    args = parser.parse_args()

    out, training_used = plots.generate_all_figures(
        comparison_root=args.comparison_root,
        output_dir=args.output_dir,
        training_csv=args.training_csv,
        rolling_window=args.rolling_window,
        exclude_verify_runs=not args.include_verify_runs,
    )
    print(f"Done. Figures directory: {os.path.abspath(out)}")
    if training_used:
        print(f"Training data: {os.path.abspath(training_used)}")


if __name__ == "__main__":
    main()
