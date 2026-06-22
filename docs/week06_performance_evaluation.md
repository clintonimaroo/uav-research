# Weekly Research Report

## Week Ending
May 8, 2026

## Researcher
Clinton Imaro

## Project Title
Performance Evaluation of Monocular and Stereo Vision for Autonomous UAV Collision Avoidance in Disaster Response Environments

## Objectives (Week 6 Checklist)
- Define success rate, collision rate, path efficiency, minimum obstacle distance, and inference latency.
- Implement automated batch evaluation.
- Add logged episode-level, step-level, event-level, and summary results.
- Run monocular versus stereo clean-condition experiments.
- Produce first comparison tables for raw PPO and PPO with safety shielding.

## Summary of Work for the Week
This week focused on converting the collision-avoidance work into a larger evaluation framework suitable for paper reporting. The same frozen PPO checkpoint from the previous navigation experiments was reused, and no retraining or network architecture changes were introduced.

The evaluation framework runs paired clean-condition experiments for stereo and monocular perception. Clean condition means that no additional depth noise, blur, occlusion, or resolution degradation was added. The deterministic hazard overlays from Week 5 were held fixed so the controller continued to operate in the disaster-response setting while the comparison focused on sensing mode and controller behavior.

The final evaluation used 1,000 total episodes across four controller and sensing conditions: raw PPO with stereo, raw PPO with monocular perception, PPO with safety shielding using stereo, and PPO with safety shielding using monocular perception.

## Accomplishments
- Added an automated Week 6 performance evaluation runner.
- Reused the Week 5 controller logic and fixed scenario maps.
- Reused the same PPO checkpoint for all controller and sensing conditions.
- Logged episode-level results, step-level latency traces, shield interventions, and safety events.
- Recorded success rate, collision rate, timeout rate, path length, path efficiency, minimum obstacle distance, hazard exposure, and inference latency.
- Used Week 1 StereoSGBM timing and Week 2 MiDaS Small timing as the perception-latency components.
- Generated comparison tables for raw PPO and PPO with safety shielding under stereo and monocular perception.

## Quantitative Results (1,000 Episodes)

| Controller | Mode | Runs | Success Rate | Collision Rate | Timeout Rate | Avg Path Length | Avg Path Efficiency | Avg Min Obstacle Distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw PPO | Stereo | 250 | 0.000 | 0.600 | 0.400 | 30.46 | 0.000 | 2.27 |
| Raw PPO | Monocular | 250 | 0.000 | 0.600 | 0.400 | 51.78 | 0.000 | 0.58 |
| PPO + Shield | Stereo | 250 | 0.808 | 0.000 | 0.192 | 92.08 | 0.641 | 2.80 |
| PPO + Shield | Monocular | 250 | 0.956 | 0.000 | 0.044 | 61.26 | 0.760 | 1.34 |

Raw PPO did not reach the goal in either sensing mode and collided in 60% of episodes. Adding the safety shield reduced collision rate to 0% for both stereo and monocular perception. The shielded monocular setup achieved the strongest clean-condition result, reaching the goal in 239 of 250 episodes. The shielded stereo setup reached the goal in 202 of 250 episodes and timed out without collision in the remaining episodes.

## Inference Latency Comparison

| Controller | Mode | Perception Latency (ms) | PPO Decision Latency (ms) | Shield Decision Latency (ms) | Total Inference Latency (ms) |
|---|---:|---:|---:|---:|---:|
| Raw PPO | Stereo | 10.81 | 0.60 | 0.00 | 11.40 |
| Raw PPO | Monocular | 39.39 | 0.59 | 0.00 | 39.99 |
| PPO + Shield | Stereo | 10.81 | 0.63 | 0.82 | 12.26 |
| PPO + Shield | Monocular | 39.39 | 0.63 | 0.82 | 40.84 |

Stereo perception was faster than monocular perception in this implementation because OpenCV StereoSGBM ran faster than MiDaS Small on the local evaluation machine. The safety shield added less than 1 ms average decision time while substantially improving collision avoidance.

## Testing and Validation

| Validation Item | Result |
|---|---|
| PPO checkpoint reused without retraining | Passed |
| PPO input shape remained `(50, 50, 4)` | Passed |
| Training updates during evaluation | 0 |
| Total episode rows logged | 1,000 |
| Rows per controller/mode condition | 250 |
| Summary rows generated | 4 |
| Latency fields present in step-level logs | Passed |
| CSV outputs checked for invalid numeric values | Passed |
| Selected visuals generated and verified | Passed |

Generated outputs were saved under:

`navigation/results/week06_performance_evaluation/final`

Main logged files:

- `navigation/results/week06_performance_evaluation/final/metrics/week06_episode_results.csv`
- `navigation/results/week06_performance_evaluation/final/metrics/week06_step_results.csv`
- `navigation/results/week06_performance_evaluation/final/metrics/week06_summary.csv`
- `navigation/results/week06_performance_evaluation/final/metrics/week06_mono_stereo_comparison.csv`
- `navigation/results/week06_performance_evaluation/final/metrics/week06_latency_comparison.csv`
- `navigation/results/week06_performance_evaluation/final/metrics/week06_interventions.csv`
- `navigation/results/week06_performance_evaluation/final/metrics/week06_events.csv`
- `navigation/results/week06_performance_evaluation/final/metrics/week06_config.json`

## Visuals

Selected visuals to include in the report:

Figure 1a. Raw PPO stereo trajectory.

`navigation/results/week06_performance_evaluation/final/visuals/week06_selected_raw_ppo_stereo_trajectory.png`

![Raw PPO stereo trajectory](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week06_performance_evaluation/final/visuals/week06_selected_raw_ppo_stereo_trajectory.png)

Figure 1b. Raw PPO monocular trajectory.

`navigation/results/week06_performance_evaluation/final/visuals/week06_selected_raw_ppo_mono_trajectory.png`

![Raw PPO monocular trajectory](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week06_performance_evaluation/final/visuals/week06_selected_raw_ppo_mono_trajectory.png)

Figure 2a. PPO+shield stereo trajectory.

`navigation/results/week06_performance_evaluation/final/visuals/week06_selected_ppo_shield_stereo_trajectory.png`

![PPO+shield stereo trajectory](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week06_performance_evaluation/final/visuals/week06_selected_ppo_shield_stereo_trajectory.png)

Figure 2b. PPO+shield monocular trajectory.

`navigation/results/week06_performance_evaluation/final/visuals/week06_selected_ppo_shield_mono_trajectory.png`

![PPO+shield monocular trajectory](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week06_performance_evaluation/final/visuals/week06_selected_ppo_shield_mono_trajectory.png)

## Limitations
- The 1,000 episodes are controlled route variants over locked UAVStereo-derived maps, not 1,000 unique real-world flight recordings.
- Raw PPO remains unreliable after risk-map input adaptation, so the paper should report it as an ablation baseline rather than a deployable controller.
- The stronger result comes from PPO with evaluation-time safety shielding, so the controller should be described as PPO plus a collision-safety layer.
- Minimum obstacle distance is measured in grid-cell clearance from high-risk obstacle cells, not physical meters.
- Week 7 should evaluate how the mono and stereo pipelines degrade under noise, blur, lower resolution, and occlusion.

## Next Steps
- Use the Week 6 evaluation framework as the baseline for Week 7 robustness experiments.
- Add controlled perturbations for depth noise, image blur, reduced resolution, and partial occlusion.
- Compare degradation curves for monocular and stereo sensing under the same PPO+shield controller.
