# Weekly Research Report

## Week Ending
May 15, 2026

## Researcher
Clinton Imaro

## Project Title
Performance Evaluation of Monocular and Stereo Vision for Autonomous UAV Collision Avoidance in Disaster Response Environments

## Objectives (Week 7 Checklist)
- Introduce controlled perception perturbations.
- Evaluate depth noise, image blur, reduced resolution, and partial occlusion.
- Measure performance degradation curves.
- Identify major failure modes.
- Compare monocular and stereo robustness using the Week 6 clean-condition baseline.

## Summary of Work for the Week
This week focused on robustness evaluation for the PPO+shield collision-avoidance system. The same frozen PPO checkpoint, fixed scenarios, and Week 6 evaluation structure were reused. No PPO retraining, optimizer update, or architecture change was introduced.

The experiment applied controlled perturbations to the perception risk-map interface before navigation. Physical collision was still measured against the unperturbed obstacle map so that occlusion or distortion could not hide real obstacle contact. This keeps the robustness results aligned with the paper objective: degraded perception affects the controller input, while safety is evaluated against the underlying obstacle field.

The final robustness evaluation used 8,000 perturbation episodes across four perturbation types, four severity levels, and two sensing modes.

## Accomplishments
- Added a Week 7 robustness evaluation runner.
- Reused the Week 6 PPO+shield evaluation path.
- Added deterministic perturbations for depth noise, image blur, reduced resolution, and partial occlusion.
- Logged episode-level, step-level, intervention, event, degradation, and failure-mode results.
- Generated degradation-curve CSVs and selected PNG figures.
- Preserved the Week 6 raw PPO result as the ablation baseline while focusing Week 7 on the working PPO+shield controller.

## Quantitative Results (8,000 Episodes)

| Mode | Perturbation | Low Success | Medium Success | High Success | Extreme Success | Worst Collision Rate |
|---|---:|---:|---:|---:|---:|---:|
| Monocular | Depth noise | 0.952 | 0.944 | 0.944 | 0.908 | 0.000 |
| Monocular | Image blur | 0.920 | 0.848 | 0.728 | 0.728 | 0.048 |
| Monocular | Reduced resolution | 0.920 | 0.768 | 0.728 | 0.724 | 0.052 |
| Monocular | Partial occlusion | 0.920 | 0.868 | 0.844 | 0.832 | 0.128 |
| Stereo | Depth noise | 0.848 | 0.848 | 0.828 | 0.764 | 0.012 |
| Stereo | Image blur | 0.832 | 0.836 | 0.756 | 0.824 | 0.160 |
| Stereo | Reduced resolution | 0.828 | 0.836 | 0.680 | 0.792 | 0.224 |
| Stereo | Partial occlusion | 0.648 | 0.524 | 0.460 | 0.404 | 0.508 |

The monocular pipeline was more stable than stereo under most perturbations in this controlled evaluation. Depth noise produced the smallest degradation for both modes. Partial occlusion produced the strongest failure mode, especially for stereo, where success dropped to 40.4% and collision rate increased to 50.8% at extreme severity.

## Clean Baseline Comparison

| Mode | Clean Week 6 Success | Worst Week 7 Success | Largest Success Drop | Largest Collision Increase |
|---|---:|---:|---:|---:|
| Monocular | 0.956 | 0.724 | 0.232 | 0.128 |
| Stereo | 0.808 | 0.404 | 0.404 | 0.508 |

Compared with the Week 6 clean-condition baseline, monocular robustness degraded most under reduced resolution and image blur. Stereo robustness degraded most under partial occlusion, indicating that missing or hidden structural risk regions are a major failure mode for the stereo-driven navigation interface.

## Failure Modes
- Partial occlusion created the most severe failures because it can hide obstacle risk from the controller while physical collision is still checked against the clean obstacle map.
- Image blur and reduced resolution mainly reduced path efficiency and increased timeouts, especially in monocular mode at high and extreme severity.
- Depth noise was the least damaging perturbation, suggesting that the shield is relatively tolerant to moderate continuous risk-map noise.
- Stereo showed higher collision sensitivity under partial occlusion and reduced resolution, while monocular showed more timeout-oriented degradation under blur and reduced resolution.

## Testing and Validation

| Validation Item | Result |
|---|---|
| PPO checkpoint reused without retraining | Passed |
| PPO input shape remained `(50, 50, 4)` | Passed |
| Training updates during evaluation | 0 |
| Total episode rows logged | 8,000 |
| Rows per mode/perturbation/severity condition | 250 |
| Summary rows generated | 32 |
| Degradation rows generated | 32 |
| Failure-mode rows generated | 32 |
| CSV outputs checked for invalid numeric values | Passed |
| Selected visuals generated and verified | Passed |

Generated outputs were saved under:

`navigation/results/week07_robustness/final`

Main logged files:

- `navigation/results/week07_robustness/final/metrics/week07_episode_results.csv`
- `navigation/results/week07_robustness/final/metrics/week07_step_results.csv`
- `navigation/results/week07_robustness/final/metrics/week07_summary.csv`
- `navigation/results/week07_robustness/final/metrics/week07_degradation_curves.csv`
- `navigation/results/week07_robustness/final/metrics/week07_failure_modes.csv`
- `navigation/results/week07_robustness/final/metrics/week07_interventions.csv`
- `navigation/results/week07_robustness/final/metrics/week07_events.csv`
- `navigation/results/week07_robustness/final/metrics/week07_config.json`

## Visuals

Selected visuals to include in the report:

Figure 1. Success-rate degradation curves.

`navigation/results/week07_robustness/final/visuals/week07_degradation_success_rate.png`

![Week 7 success-rate degradation curves](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week07_robustness/final/visuals/week07_degradation_success_rate.png)

Figure 2. Collision-rate degradation curves.

`navigation/results/week07_robustness/final/visuals/week07_degradation_collision_rate.png`

![Week 7 collision-rate degradation curves](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week07_robustness/final/visuals/week07_degradation_collision_rate.png)

Figure 3. Path-efficiency degradation curves.

`navigation/results/week07_robustness/final/visuals/week07_degradation_path_efficiency.png`

![Week 7 path-efficiency degradation curves](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week07_robustness/final/visuals/week07_degradation_path_efficiency.png)

Figure 4. Minimum obstacle-distance degradation curves.

`navigation/results/week07_robustness/final/visuals/week07_degradation_min_obstacle_distance.png`

![Week 7 minimum obstacle-distance degradation curves](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week07_robustness/final/visuals/week07_degradation_min_obstacle_distance.png)

Figure 5. Stereo trajectory under extreme partial occlusion.

`navigation/results/week07_robustness/final/visuals/week07_selected_stereo_partial_occlusion_extreme_trajectory.png`

![Stereo trajectory under extreme partial occlusion](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week07_robustness/final/visuals/week07_selected_stereo_partial_occlusion_extreme_trajectory.png)

Figure 6. Monocular trajectory under extreme partial occlusion.

`navigation/results/week07_robustness/final/visuals/week07_selected_mono_partial_occlusion_extreme_trajectory.png`

![Monocular trajectory under extreme partial occlusion](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week07_robustness/final/visuals/week07_selected_mono_partial_occlusion_extreme_trajectory.png)

## Limitations
- Perturbations are controlled perception-interface perturbations over locked UAVStereo-derived maps.
- The 8,000 episodes are route variants over fixed benchmark maps, not 8,000 unique real-world flight recordings.
- Physical clearance is reported in grid-cell distance from high-risk obstacle cells, not meters.
- Week 7 evaluates robustness of PPO+shield; raw PPO remains the Week 6 ablation baseline.
- Partial occlusion is intentionally severe because it simulates hidden or missing obstacle evidence, which is a realistic failure mode for visual navigation.

## Next Steps
- Use the Week 6 clean-condition table and Week 7 robustness curves as the main Paper 2 evaluation evidence.
- Prepare final paper figures comparing monocular and stereo sensing across clean and perturbed conditions.
- Add concise discussion explaining why PPO+shield is required and why raw PPO should be treated as an ablation baseline.
