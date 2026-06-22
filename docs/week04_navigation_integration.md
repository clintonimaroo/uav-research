# Weekly Research Report

## Week Ending
April 24, 2026

## Researcher
Clinton Imaro

## Project Title
Performance Evaluation of Monocular and Stereo Vision for Autonomous UAV Collision Avoidance in Disaster Response Environments

## Objectives (Week 4 Checklist)
- Integrate obstacle/risk maps into the existing PPO navigation input.
- Use risk maps instead of raw depth as the navigation perception channel.
- Reuse the trained PPO policy without changing the PPO architecture.
- Use input adaptation only.
- Run stereo-driven PPO navigation.
- Run monocular-driven PPO navigation.
- Verify that no retraining instability is introduced.

## Summary of Work for the Week
This week connected the aligned Week 3 perception outputs to the existing PPO navigation system. The stereo and monocular risk maps were converted into the same four-channel PPO observation format used by the trained navigation model.

The PPO architecture was kept unchanged. The existing checkpoint at `navigation/navigation_models/best_navigation_model.pth` was loaded for deterministic inference only. No optimizer step, policy update, or retraining process was used.

The adapted PPO input used four channels:

| Channel | Input |
|---|---|
| 0 | Selected perception risk map, stereo or monocular |
| 1 | Known-map confidence mask |
| 2 | UAV position distance field |
| 3 | Goal distance field |

Navigation was tested across 50 total episodes using the five locked Week 3 scenario maps. The run included 25 stereo-driven PPO episodes and 25 monocular-driven PPO episodes. Before the final run, the start and goal cells were audited so the UAV and destination were not placed inside high-risk cells for either perception mode. Repeated episodes used deterministic start/goal variants derived from the corrected scenario definitions so the evaluation remained reproducible.

## Accomplishments
- Added the Week 4 navigation integration runner.
- Reused the existing PPO navigation agent and checkpoint.
- Preserved the PPO input shape as `(50, 50, 4)`.
- Preserved the eight-action discrete movement space.
- Integrated stereo risk maps into the PPO input.
- Integrated monocular risk maps into the same PPO input interface.
- Generated per-run navigation logs, step-level logs, summary metrics, configuration output, and trajectory overlays.
- Confirmed that the navigation system runs end-to-end for both stereo and monocular perception modes.
- Confirmed that input adaptation alone is not sufficient for reliable goal completion on the current depth-derived obstacle maps.

## Quantitative Results (50 Episodes)

| Mode | Runs | Success Rate | Collision Rate | Timeout Rate | Average Steps | Average Final Distance | Average Path Length | Average Path Risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stereo | 25 | 0.00 | 0.60 | 0.40 | 82.72 | 22.98 | 30.46 | 0.30 |
| Monocular | 25 | 0.00 | 0.60 | 0.40 | 100.64 | 25.53 | 51.78 | 0.03 |

The stereo-driven and monocular-driven PPO setups both completed the inference workflow and produced full logs, but neither mode reached the goal under the current frozen PPO policy. The result should be interpreted as a successful integration test, not as a successful collision-avoidance performance result. A separate empty-map diagnostic confirmed that the PPO checkpoint can still reach the goal when no obstacle-risk field is present, so the weak result is caused by the current zero-shot transfer from depth-derived risk maps into the trained PPO hazard channel.

## Testing and Validation

| Validation Item | Result |
|---|---|
| PPO checkpoint loaded successfully | Passed |
| PPO input shape remained `(50, 50, 4)` | Passed |
| PPO action space remained eight discrete moves | Passed |
| Stereo and monocular modes both ran end-to-end | Passed |
| Five locked Week 3 scenario maps were used | Passed |
| Fifty total navigation episodes were run | Passed |
| Fifty trajectory overlays were generated | Passed |
| CSV logs contained no NaN or infinite values | Passed |
| Start and goal cells were below the collision-risk threshold | Passed |
| No PPO retraining or policy update was used | Passed |
| Source compilation check passed | Passed |

Generated outputs were saved under:

`navigation/results/week04_navigation_integration/final`

Main logged files:

- `navigation/results/week04_navigation_integration/final/metrics/week04_navigation_runs.csv`
- `navigation/results/week04_navigation_integration/final/metrics/week04_navigation_steps.csv`
- `navigation/results/week04_navigation_integration/final/metrics/week04_navigation_summary.csv`
- `navigation/results/week04_navigation_integration/final/metrics/week04_navigation_config.json`

## Visuals

The trajectory overlays show how the PPO policy moved through the fixed scenario risk maps when driven by stereo and monocular perception.

Selected visuals to include in the report:

Figure 1a. Stereo-driven PPO trajectory on the Baseline Open scenario.

`navigation/results/week04_navigation_integration/final/trajectories/episode_001_baseline_open_stereo_trajectory.png`

![Stereo-driven PPO trajectory on the Baseline Open scenario](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week04_navigation_integration/final/trajectories/episode_001_baseline_open_stereo_trajectory.png)

Figure 1b. Monocular-driven PPO trajectory on the Baseline Open scenario.

`navigation/results/week04_navigation_integration/final/trajectories/episode_002_baseline_open_mono_trajectory.png`

![Monocular-driven PPO trajectory on the Baseline Open scenario](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week04_navigation_integration/final/trajectories/episode_002_baseline_open_mono_trajectory.png)

## Next Steps
- Continue to Week 5 collision-avoidance validation.
- Use the fixed scenario set to analyze collision events, minimum obstacle distance, and avoidance action traces.
- Refine collision thresholds and safety handling without changing the PPO architecture.
- Only consider PPO retraining if later approved.
