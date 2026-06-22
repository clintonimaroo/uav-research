# Paper 1 Fire-Density PPO Training Summary

These runs use controlled Paper 1 fire-density maps recreated from the original profile and seed used for the earlier environment figures. Training loads the saved map package so the same map condition is used throughout each run.

- Episodes per map: 3000
- Grid size: 50 x 50
- Smoothing window: 100

| Map | Profile | Seed | Episodes | Last-100 Success | Last-100 Reward | Last-100 Path Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Light/Sparse Fire | fire_light | 20260530 | 3000 | 0.990 | 412.005 | 0.638 |
| Moderate Fire | fire_moderate | 20360530 | 3000 | 1.000 | 415.706 | 0.638 |
| Dense/Heavy Fire | fire_dense | 20460530 | 3000 | 0.000 | 170.074 | 0.033 |

## Key Output Files

- `figures/paper1_fire_density_reward_vs_episode.png`
- `figures/paper1_fire_density_success_rate_vs_episode.png`
- `figures/paper1_fire_density_path_efficiency_vs_episode.png`
- `metrics/paper1_fire_density_training_summary.csv`
