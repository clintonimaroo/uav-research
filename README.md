# UAV Disaster Detection System

A lightweight PyTorch-based disaster detection system designed for autonomous UAV deployment. This system enables drones to detect disasters mid-flight and trigger autonomous rerouting for emergency response.

## Paper 1 Group A/B Reproducibility

From the `navigation/` directory:

- Run denser A* vs PPO sweeps and auto-generate Group A/B figures:
  - `bash run_experiments.sh 50`
- Generate plots only from existing CSV outputs:
  - `python plot_paper1_groupA.py --comparison_root comparison_results --output_dir comparison_results/paper1_groupA_figures`
- Include a specific training CSV (optional):
  - `python plot_paper1_groupA.py --comparison_root comparison_results --training_csv navigation_models/training_episode_metrics_<run_id>.csv --output_dir comparison_results/paper1_groupA_figures`

CSV outputs produced by the pipeline:

- PPO training (episode-level): `navigation_models/training_episode_metrics_<run_id>.csv`
- Method comparison (episode-level): `<comparison_run_dir>/episode_results_<run_id>.csv`
- Method comparison (run summary): `<comparison_run_dir>/run_summary.csv`
