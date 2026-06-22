# Weekly Research Report

## Week Ending
May 22, 2026

## Researcher
Clinton Imaro

## Project Title
Performance Evaluation of Monocular and Stereo Vision for Autonomous UAV Collision Avoidance in Disaster Response Environments

## Objectives (Week 8 Checklist)
- Focus the final contribution narrative.
- Run ablation studies for noise levels, hazard density, and perception degradation.
- Generate final paper-ready plots and tables.
- Extract efficiency-versus-safety trade-offs.

## Summary of Work for the Week
This week consolidated the Week 6 clean-condition evaluation and Week 7 robustness results into final paper evidence. A new hazard-density ablation was added so that the disaster-risk component is isolated separately from perception degradation. The PPO checkpoint and network architecture were not changed.

## Accomplishments
- Consolidated clean-condition, controller-ablation, perception-degradation, and hazard-density results.
- Added a direct hazard-density ablation using the same PPO+shield controller.
- Generated final tables and selected paper-ready figures.
- Identified the main safety-efficiency trade-offs for the paper discussion.

## Clean-Condition Controller Result

| Controller | Mode | Success Rate | Collision Rate | Path Efficiency | Min Obstacle Distance | Total Inference Latency |
|---|---:|---:|---:|---:|---:|---:|
| ppo_shield | mono | 0.956 | 0.000 | 0.760 | 1.337 | 40.84 ms |
| ppo_shield | stereo | 0.808 | 0.000 | 0.641 | 2.798 | 12.26 ms |
| raw_ppo | mono | 0.000 | 0.600 | 0.000 | 0.576 | 39.99 ms |
| raw_ppo | stereo | 0.000 | 0.600 | 0.000 | 2.271 | 11.40 ms |

## Controller Ablation

| Mode | Raw Success | Shield Success | Success Gain | Raw Collision | Shield Collision | Collision Reduction |
|---|---:|---:|---:|---:|---:|---:|
| mono | 0.000 | 0.956 | 0.956 | 0.600 | 0.000 | 0.600 |
| stereo | 0.000 | 0.808 | 0.808 | 0.600 | 0.000 | 0.600 |

## Hazard-Density Ablation

| Mode | Hazard Density | Success Rate | Collision Rate | Hazard Exposure Steps | Shield Interventions |
|---|---:|---:|---:|---:|---:|
| mono | none | 1.000 | 0.000 | 0.000 | 13.160 |
| mono | low | 1.000 | 0.000 | 0.000 | 13.220 |
| mono | moderate | 0.952 | 0.000 | 0.032 | 20.644 |
| mono | high | 0.772 | 0.000 | 0.196 | 47.400 |
| stereo | none | 0.920 | 0.000 | 0.000 | 31.440 |
| stereo | low | 0.908 | 0.000 | 0.000 | 33.336 |
| stereo | moderate | 0.740 | 0.000 | 0.032 | 58.968 |
| stereo | high | 0.436 | 0.000 | 0.396 | 105.932 |

## Key Insights
- Raw PPO collision rate reached 0.600, so it should be presented as an ablation baseline rather than the final controller.
- PPO+shield reduced clean-condition collision rate to 0.000 for both sensing modes.
- Monocular PPO+shield had higher clean success (0.956) but higher latency (40.84 ms).
- Stereo PPO+shield was faster (12.26 ms) but less robust under partial occlusion and reduced-resolution stress.
- Hazard density mainly increases exposure and shield workload; physical collision remains evaluated from structural obstacle risk.

## Testing and Validation

| Validation Item | Result |
|---|---|
| Hazard-density episode rows | 2,000 |
| Hazard-density summary rows | 8 |
| Rows per mode/hazard level | 250 |
| Week 6 logs reused | Passed |
| Week 7 logs reused | Passed |
| PPO checkpoint reused without retraining | Passed |
| PPO input shape remained `(50, 50, 4)` | Passed |
| Training updates during evaluation | 0 |

Generated outputs were saved under:

`/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week08_analysis_ablation/final`

## Visuals

Figure 1. `/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week08_analysis_ablation/final/visuals/week08_clean_success_rate.png`

![Week 8 visual 1](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week08_analysis_ablation/final/visuals/week08_clean_success_rate.png)

Figure 2. `/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week08_analysis_ablation/final/visuals/week08_clean_collision_rate.png`

![Week 8 visual 2](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week08_analysis_ablation/final/visuals/week08_clean_collision_rate.png)

Figure 3. `/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week08_analysis_ablation/final/visuals/week08_hazard_density_success_rate.png`

![Week 8 visual 3](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week08_analysis_ablation/final/visuals/week08_hazard_density_success_rate.png)

Figure 4. `/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week08_analysis_ablation/final/visuals/week08_hazard_density_exposure.png`

![Week 8 visual 4](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week08_analysis_ablation/final/visuals/week08_hazard_density_exposure.png)

Figure 5. `/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week08_analysis_ablation/final/visuals/week08_perception_degradation_success.png`

![Week 8 visual 5](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week08_analysis_ablation/final/visuals/week08_perception_degradation_success.png)

Figure 6. `/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week08_analysis_ablation/final/visuals/week08_latency_success_tradeoff.png`

![Week 8 visual 6](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week08_analysis_ablation/final/visuals/week08_latency_success_tradeoff.png)

Figure 7. `/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week08_analysis_ablation/final/visuals/week08_selected_stereo_none_hazard_density_trajectory.png`

![Week 8 visual 7](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week08_analysis_ablation/final/visuals/week08_selected_stereo_none_hazard_density_trajectory.png)

Figure 8. `/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week08_analysis_ablation/final/visuals/week08_selected_mono_none_hazard_density_trajectory.png`

![Week 8 visual 8](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week08_analysis_ablation/final/visuals/week08_selected_mono_none_hazard_density_trajectory.png)

## Next Steps
- Use Week 8 tables and figures as the Paper 2 results backbone.
- Move the clean-condition, degradation, hazard-density, and trade-off figures into the final paper draft.
- Keep limitations explicit: controlled UAVStereo-derived maps, perception-interface perturbations, and grid-cell clearance rather than meter-scale flight testing.
