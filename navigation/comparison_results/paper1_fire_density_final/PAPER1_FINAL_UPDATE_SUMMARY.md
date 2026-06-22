# Paper 1 Post-Meeting Results Package

## Purpose

This package responds to the post-meeting request to separate the Paper 1 results into light/sparse fire, moderate fire, and dense/heavy fire settings, while keeping the earlier 50-episode result as validation rather than final paper evidence.

## Main 50 x 50 Evaluation Table

| Environment | Method | Episodes | Success Rate | Avg Reward | Avg Path Length | Path Efficiency | Hazard Encounters | Final Distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense/Heavy Fire | A* | 500 | 0.766 | 318.594 | 95.165 | 0.573 | 0.070 | 8.834 |
| Dense/Heavy Fire | PPO | 500 | 0.034 | -17.240 | 17.955 | 0.034 | 0.102 | 45.685 |
| Light/Sparse Fire | A* | 500 | 0.998 | 411.687 | 68.499 | 0.926 | 0.018 | 0.126 |
| Light/Sparse Fire | PPO | 500 | 0.002 | 20.958 | 27.116 | 0.002 | 0.424 | 36.523 |
| Moderate Fire | A* | 500 | 0.976 | 403.182 | 74.346 | 0.838 | 0.078 | 0.771 |
| Moderate Fire | PPO | 500 | 0.000 | -3.123 | 22.444 | 0.000 | 0.234 | 41.196 |

## Grid-Size Generalization Table

| Environment | Grid Size | Method | Episodes | Success Rate | Avg Reward | Path Efficiency | Hazard Encounters | Final Distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Saved PPO Training Summary

This table is the saved PPO training-history record from the checkpoint metrics file. It is not the same as the three-environment evaluation table above.

| Source | Window | Episodes | Success Rate | Avg Reward | Avg Steps | Avg Navigation Efficiency |
|---|---|---:|---:|---:|---:|---:|
| Saved PPO Training Summary | last_100 | 100 | 0.270 | 238.920 | 136.990 | 0.187 |
| Saved PPO Training Summary | last_300 | 300 | 0.293 | 217.002 | 123.537 | 0.180 |
| Saved PPO Training Summary | last_1000 | 1000 | 0.368 | 226.985 | 117.106 | 0.200 |
| Saved PPO Training Summary | all_3000 | 3000 | 0.320 | 212.999 | 120.251 | 0.183 |

## Figures

- `figures/paper1_final_success_rate_vs_fire_density.png`
- `figures/paper1_grid_success_rate_by_environment.png`
- `figures/paper1_final_baseline_reward_vs_episode.png`
- `figures/paper1_final_baseline_success_rate_vs_episode.png`
- `figures/paper1_grid_path_efficiency_by_environment.png`
- `figures/paper1_grid_hazard_encounters_by_environment.png`

## Experiment Setup

The baseline grid is `50 x 50`. The grid-size sweep uses `40 x 40, 50 x 50, 60 x 60, 70 x 70, 85 x 85, 100 x 100`.

PPO and A* are evaluated on the same fixed seeded fire-layout variants for each environment condition. The navigation environment uses the real CNN classifier during evaluation.

## Interpretation

The final paper table should use the main 50 x 50 evaluation results, while the grid-size table should be presented as a generalization and sensitivity analysis.
