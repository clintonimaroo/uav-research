# Paper 1 Results Update

## Purpose
This package reorganizes the Paper 1 results around three representative synthetic fire environments and updates the plotted evidence using the 3,000-episode PPO training record.

## Three Fire Environments

| Environment | Method | Episodes | Success Rate | Avg Path Efficiency | Avg Hazard Encounters | Avg Reward |
|---|---:|---:|---:|---:|---:|---:|
| Dense/Heavy Fire | A* | 50 | 0.760 | 0.587 | 0.100 | 319.570 |
| Dense/Heavy Fire | PPO | 50 | 0.080 | 0.080 | 0.120 | 17.850 |
| Light/Sparse Fire | A* | 50 | 1.000 | 0.980 | 0.000 | 413.449 |
| Light/Sparse Fire | PPO | 50 | 0.640 | 0.640 | 0.040 | 295.936 |
| Moderate Fire | A* | 50 | 1.000 | 0.943 | 0.000 | 412.805 |
| Moderate Fire | PPO | 50 | 0.380 | 0.380 | 0.100 | 162.905 |

## 3,000-Episode PPO Training Summary

| Window | Episodes | Success Rate | Avg Reward | Avg Steps | Avg Navigation Efficiency |
|---|---:|---:|---:|---:|---:|
| last_100 | 100 | 0.270 | 238.920 | 136.990 | 0.187 |
| last_300 | 300 | 0.293 | 217.002 | 123.537 | 0.180 |
| last_1000 | 1000 | 0.368 | 226.985 | 117.106 | 0.200 |
| all_3000 | 3000 | 0.320 | 212.999 | 120.251 | 0.183 |

## Interpretation
The previous figure folder was not clearly separated into light, moderate, and dense fire environments. The updated package locks those three environment profiles and reports PPO and A* on the same seeded fire-layout variants.

The episode-based plots now follow the requested format: raw episode behavior remains visible in the background, while smoothed trend lines with markers make the learning trend easier to read.

A* is the stronger planner in this three-environment evaluation. PPO remains competitive only in the light-fire setting and degrades as fire density increases. This should be presented as a useful result rather than hidden: the current PPO policy is more adaptive than a static route follower, but the present checkpoint is not yet matching the classical hazard-aware planner in dense fire layouts.

The current plots in `navigation/comparison_results/paper1_groupA_figures` should be treated as the earlier general comparison set. They are useful as baseline/sweep evidence, but they were not explicitly separated into the three representative fire-density environments Peter requested.

## Talking Points For Meeting

- I organized Paper 1 around three controlled fire-density environments: light, moderate, and dense.
- I reran A* and PPO on the same seeded layouts for each environment using the real CNN classifier and the saved PPO checkpoint.
- I updated the evidence from the older 300-episode paper table toward the available 3,000-episode PPO training record.
- I replotted the episode-based curves using raw background traces plus smoothed trend lines and markers, following the plotting style you suggested.
- The result is honest: A* is stronger in the current three-environment comparison, while PPO weakens as fire density increases. This gives us a clear discussion point and supports future hybrid PPO + A* work.

## Main Files

- `metrics/paper1_episode_results.csv`
- `metrics/paper1_environment_summary.csv`
- `metrics/paper1_training_3000_summary.csv`
- `metrics/paper1_experiment_parameters.csv`
- `figures/paper1_reward_vs_episode_smoothed.png`
- `figures/paper1_success_rate_vs_episode_smoothed.png`
- `figures/paper1_fire_density_reward_vs_episode_comparison.png`
- `figures/paper1_fire_density_success_vs_episode_comparison.png`
- `figures/paper1_success_rate_fire_density_paper.pdf`
- `figures/paper1_fire_environment_success_rate.png`
- `figures/paper1_light_sparse_fire_environment.png`
- `figures/paper1_moderate_fire_environment.png`
- `figures/paper1_dense_heavy_fire_environment.png`

## Limitations
The three-environment comparison uses controlled synthetic fire-density profiles over the existing grid navigation simulator. The 3,000-episode table comes from the saved PPO training record, while the fire-density comparison is a separate evaluation over fixed seeded environment variants.
