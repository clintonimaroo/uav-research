# Weekly Research Report

## Week Ending
May 1, 2026

## Researcher
Clinton Imaro

## Project Title
Performance Evaluation of Monocular and Stereo Vision for Autonomous UAV Collision Avoidance in Disaster Response Environments

## Objectives (Week 5 Checklist)
- Validate obstacle avoidance behavior.
- Validate hazard and obstacle coexistence using disaster-style hazard overlays.
- Tune only collision thresholds and evaluation-time safety penalties.
- Keep the same PPO policy usable for stereo and monocular sensing.
- Improve reliable avoidance behavior under both sensing modes.

## Summary of Work for the Week
This week focused on validating collision avoidance behavior after the Week 4 integration showed that raw PPO input adaptation alone was not reliable. The PPO architecture and checkpoint were kept unchanged. A multi-step safety shield was added around the PPO action output so the same trained policy could be used with both stereo and monocular risk maps.

The shield evaluates the PPO-proposed action before execution. If the proposed action leads toward high obstacle risk, high hazard exposure, negative goal progress, or a repeated-position loop, the shield compares the available discrete actions using local lookahead and selects a safer action that still preserves goal progress.

Hazard and obstacle coexistence was evaluated by adding deterministic disaster-style hazard overlays to the existing structural obstacle risk maps. The policy input used the maximum of obstacle risk and hazard risk, while physical collision was measured only from the obstacle-risk map.

## Accomplishments
- Added a Week 5 collision-avoidance validation runner.
- Reused the Week 4 stereo and monocular risk-map interface.
- Reused the same frozen PPO checkpoint for both sensing modes.
- Added a three-step evaluation-time safety shield.
- Logged shield interventions, collision events, hazard exposure, and joint hazard+obstacle events.
- Ran paired raw PPO and PPO+shield evaluations across 50 total episodes per controller condition.
- Generated run-level logs, step-level logs, event logs, intervention logs, summary tables, trajectory overlays, and selected report visuals.

## Quantitative Results (50 Episodes Per Controller)

| Controller | Mode | Runs | Success Rate | Collision Rate | Timeout Rate | Avg Final Distance | Avg Shield Interventions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw PPO | Stereo | 25 | 0.00 | 0.60 | 0.40 | 22.98 | 0.00 |
| Raw PPO | Monocular | 25 | 0.00 | 0.60 | 0.40 | 25.53 | 0.00 |
| PPO + Shield | Stereo | 25 | 0.88 | 0.00 | 0.12 | 3.36 | 37.36 |
| PPO + Shield | Monocular | 25 | 1.00 | 0.00 | 0.00 | 0.00 | 13.40 |

The safety shield reduced collision rate from 60% to 0% for both stereo and monocular modes. The monocular setup reached the goal in all 25 shielded episodes. The stereo setup reached the goal in 22 of 25 shielded episodes and timed out in the remaining 3 episodes without collision. This supports the Week 5 claim that reliable avoidance behavior can be achieved under both sensing modes while reusing the same PPO checkpoint.

## Hazard + Obstacle Coexistence

| Controller | Mode | Avg Hazard Exposure Steps | Avg Joint Hazard+Obstacle Steps |
|---|---:|---:|---:|
| Raw PPO | Stereo | 0.56 | 0.00 |
| Raw PPO | Monocular | 0.32 | 0.00 |
| PPO + Shield | Stereo | 0.04 | 0.00 |
| PPO + Shield | Monocular | 0.04 | 0.00 |

The shielded runs reduced hazard exposure while also avoiding physical obstacle collision. No joint hazard+obstacle collision events occurred in any controller or sensing mode.

## Testing and Validation

| Validation Item | Result |
|---|---|
| PPO checkpoint reused without retraining | Passed |
| PPO input shape remained `(50, 50, 4)` | Passed |
| Stereo and monocular modes used the same shield parameters | Passed |
| Raw PPO and PPO+shield were evaluated on paired episodes | Passed |
| Fifty raw PPO and fifty PPO+shield episodes were completed and logged | Passed |
| Start and goal cells were below the collision-risk threshold | Passed |
| Event and intervention logs were generated | Passed |

Generated outputs were saved under:

`navigation/results/week05_collision_avoidance/final`

Main logged files:

- `navigation/results/week05_collision_avoidance/final/metrics/week05_collision_runs.csv`
- `navigation/results/week05_collision_avoidance/final/metrics/week05_collision_steps.csv`
- `navigation/results/week05_collision_avoidance/final/metrics/week05_collision_interventions.csv`
- `navigation/results/week05_collision_avoidance/final/metrics/week05_collision_events.csv`
- `navigation/results/week05_collision_avoidance/final/metrics/week05_collision_summary.csv`
- `navigation/results/week05_collision_avoidance/final/metrics/week05_collision_config.json`

## Visuals

The selected trajectory overlays show PPO+shield navigation using the same policy with stereo and monocular perception.

Selected visuals to include in the report:

Figure 1a. Stereo-driven PPO+shield trajectory.

`navigation/results/week05_collision_avoidance/final/visuals/week05_selected_stereo_shield_trajectory.png`

![Stereo-driven PPO+shield trajectory](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week05_collision_avoidance/final/visuals/week05_selected_stereo_shield_trajectory.png)

Figure 1b. Monocular-driven PPO+shield trajectory.

`navigation/results/week05_collision_avoidance/final/visuals/week05_selected_mono_shield_trajectory.png`

![Monocular-driven PPO+shield trajectory](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week05_collision_avoidance/final/visuals/week05_selected_mono_shield_trajectory.png)

## Limitations
- The shield makes frequent interventions, especially in stereo mode, so the result should be described as PPO with safety shielding rather than raw PPO navigation.
- Stereo still timed out in 3 of 25 shielded episodes, showing that avoidance improved more strongly than path completion.
- Hazard overlays are deterministic synthetic overlays, so broader hazard perturbation testing should remain part of Week 7 robustness work.

## Next Steps
- Use the Week 5 shielded controller as the safety baseline for Week 6 performance evaluation.
- Report success rate, collision rate, path efficiency, minimum obstacle margin, and inference latency in the Week 6 evaluation framework.
- Keep raw PPO results as the ablation baseline showing why the safety shield is necessary.
