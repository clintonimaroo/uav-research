# Paper 1 Fire-Density PPO Training Summary

These runs use the exact Paper 1 fire-density maps recreated from the original profile and seed used for the earlier environment figures. Training loads the saved map package and does not regenerate a new map inside the episode loop.

- Episodes per map: 2
- Grid size: 50 x 50
- Smoothing window: 2

| Map | Profile | Seed | Episodes | Last-100 Success | Last-100 Reward | Last-100 Path Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Light/Sparse Fire | fire_light | 20260530 | 2 | 0.000 | 50.373 | 0.020 |
| Moderate Fire | fire_moderate | 20360530 | 2 | 0.000 | -0.969 | 0.020 |
| Dense/Heavy Fire | fire_dense | 20460530 | 2 | 0.000 | -66.315 | 0.019 |

## Key Output Files

- `figures/paper1_fire_density_reward_vs_episode.png`
- `figures/paper1_fire_density_success_rate_vs_episode.png`
- `figures/paper1_fire_density_path_efficiency_vs_episode.png`
- `metrics/paper1_fire_density_training_summary.csv`
