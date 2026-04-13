# UAV Disaster Detection System

A lightweight PyTorch-based disaster detection system designed for autonomous UAV deployment. This system enables drones to detect disasters mid-flight and trigger autonomous rerouting for emergency response.

## Paper 1 Group A/B Reproducibility

From the `navigation/` directory:

- Run denser A* vs PPO sweeps and auto-generate Group A/B figures:
  - `bash run_experiments.sh 50`
- Generate plots only from existing CSV outputs:
  - `python plot_paper1_groupA.py --comparison_root comparison_results --output_dir comparison_results/paper1_groupA_figures`
- Regenerate all Group A figures in one step (uses `comparison_results/` and the latest `navigation_models/training_episode_metrics_*.csv` by default; safe for headless runs):
  - `python regenerate_paper1_figures.py`  
  - From the repo root: `python navigation/regenerate_paper1_figures.py`
- Include a specific training CSV (optional):
  - `python plot_paper1_groupA.py --comparison_root comparison_results --training_csv navigation_models/training_episode_metrics_<run_id>.csv --output_dir comparison_results/paper1_groupA_figures`
  - Or with the helper: `python regenerate_paper1_figures.py --training-csv navigation_models/training_episode_metrics_<run_id>.csv`

**Repro note**: PPO was retrained for 3,000 episode with **diverse maps** (`cache_imagery=False`) and updated observation channels (UAV/goal as distance fields) plus rebalanced rewards so evaluation generalizes; older CSVs from before that change are not comparable.
