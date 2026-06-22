# Weekly Research Report

## Week Ending

April 17, 2026

## Researcher

Clinton Imaro

## Project Title

Performance Evaluation of Monocular and Stereo Vision for Autonomous UAV Collision Avoidance in Disaster Response Environments

## Week 3 Focus

The focus this week was sensor alignment and validation between the stereo and monocular depth pipelines. The goal was to confirm that both sensing approaches produce comparable obstacle/risk-map outputs before using them in later navigation evaluation.

## Objectives

- Normalize mono and stereo outputs for depth scaling, resolution, and field-of-view consistency.
- Validate that the same scene produces comparable obstacle and risk maps across sensing modes.
- Define fixed evaluation scenarios using locked maps and start/goal pairs.
- Produce a verified mono-vs-stereo comparability baseline for later navigation testing.

## Summary of Work

The Week 3 validation used the 15 shared UAVStereo residential frames generated in Week 1 and Week 2. Stereo risk maps from the OpenCV StereoSGBM baseline were compared against monocular risk maps from the MiDaS Small pipeline. Both outputs were normalized to the same frame IDs, resolution, risk-map range, and binary obstacle-map format.

Depth mismatch was handled by estimating a per-frame depth scale between the stereo and monocular depth arrays wherever valid overlap existed. This allowed the comparison to measure both risk-map agreement and depth-scale consistency without assuming that monocular depth is naturally metric.

Five fixed evaluation scenarios were also locked for later weeks. Each scenario includes a frame ID, stereo/mono map references, fixed start and goal cells, grid size, and collision threshold.

## Accomplishments

- Normalized stereo and monocular risk maps into the same `0.0` to `1.0` range.
- Confirmed mono and stereo risk maps share the same resolution and array shape.
- Confirmed mono and stereo obstacle maps use the same binary `0/1` interface.
- Generated per-frame alignment metrics for all 15 shared frames.
- Generated risk-difference heatmaps and mono-vs-stereo alignment montages.
- Locked five evaluation scenarios for later navigation experiments.

## Testing and Validation

Testing was conducted with an initial sustainable validation test on a small fixed subset of frames, followed by the full 15-frame shared validation run.

Validation results:

| Validation Item | Result |
|---|---:|
| Evaluated frames | 15 |
| Frame IDs matched across mono and stereo | Yes |
| Mono/stereo risk-map shapes matched | Yes |
| Mono/stereo obstacle-map shapes matched | Yes |
| Risk-map value range | 0.0 to 1.0 |
| Obstacle-map format | Binary 0/1 |
| Locked evaluation scenarios | 5 |
| Scenario start/goal cells inside 50 x 50 grid | Yes |

Summary metrics:

| Metric | Value |
|---|---:|
| Average risk MAE | 0.710 |
| Average risk RMSE | 0.993 |
| Average risk correlation | -0.853 |
| Average structural similarity score | -0.123 |
| Average binary obstacle agreement | 0.254 |
| Average obstacle IoU | 0.052 |
| Average depth scale factor | 1.093 |
| Average aligned depth MAE | 0.042 m |
| Average aligned depth RMSE | 0.060 m |
| Average valid depth overlap | 0.036 |

The validation confirms that the stereo and monocular pipelines are structurally comparable because they produce the same risk-map and obstacle-map interface on the same frames. However, the numerical agreement is still limited, especially for obstacle IoU and risk-map correlation. This means the outputs are ready for controlled comparison, but the results should not yet be interpreted as monocular depth matching stereo accuracy.

## Locked Evaluation Scenarios

| Scenario | Frame ID | Start | Goal | Purpose |
|---|---:|---:|---:|---|
| Baseline Open | 0020 | [2, 2] | [47, 47] | Lower obstacle-density baseline |
| Dense Corridor | 0050 | [5, 5] | [45, 45] | Higher obstacle-density case |
| Obstacle Cluster | 0060 | [2, 47] | [47, 2] | Low mono-stereo obstacle agreement |
| Lateral Maneuver | 0010 | [10, 5] | [40, 45] | Low mono-stereo risk correlation |
| Stress Density | 0040 | [5, 45] | [45, 5] | High mono-stereo risk error |

The locked scenario definitions are saved in:

`navigation/scenarios.json`

## Visuals

Visual evidence was generated as frame-level alignment montages and risk-difference heatmaps. Each alignment montage shows:

- Stereo risk map
- Monocular risk map
- Absolute risk-difference heatmap
- Stereo obstacle map
- Monocular obstacle map
- Risk-map overlay

Selected visual to include in the report:

Figure 1: Mono-vs-stereo sensor alignment for the Stress Density scenario.

`navigation/results/week03_sensor_alignment/final/montage/frame_0040_week3_alignment.png`

![Figure 1: Mono-vs-stereo sensor alignment for the Stress Density scenario.](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week03_sensor_alignment/final/montage/frame_0040_week3_alignment.png)

The generated Week 3 outputs are saved under:

`navigation/results/week03_sensor_alignment/final`

## Next Steps

For Week 4, the next step is to connect the aligned risk-map interface to the navigation system. The locked scenarios from Week 3 should be used to keep future mono-vs-stereo navigation tests reproducible.
