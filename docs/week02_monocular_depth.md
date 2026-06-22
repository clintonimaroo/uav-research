# Weekly Research Report

## Week Ending

April 10, 2026

## Researcher

Clinton Imaro

## Project Title

Performance Evaluation of Monocular and Stereo Vision for Autonomous UAV Collision Avoidance in Disaster Response Environments

## Week 2 Focus

The focus this week was implementing the monocular depth model baseline for Paper 2. The goal was to evaluate whether a single-camera depth model can produce obstacle and risk-map outputs that are comparable in structure to the Week 1 stereo baseline.

## Objectives

- Integrate a monocular depth model after reviewing MiDaS and DPT options.
- Generate monocular depth maps from the same UAVStereo residential scenes used in Week 1.
- Convert monocular depth output into obstacle and risk maps using the same formulation as the stereo baseline.
- Ensure monocular and stereo outputs use the same interface for later navigation evaluation.
- Produce monocular obstacle maps, risk maps, visual comparisons, and logged metrics.

## Summary of Work

The Week 2 monocular depth pipeline was implemented using Intel MiDaS Small as the default monocular model. MiDaS Small was selected because it is lightweight and fits the paper’s motivation of comparing a single-camera setup against a more demanding stereo-camera setup for small UAVs. DPT-compatible model selection was also preserved for future testing, but the Week 2 results were generated with MiDaS Small.

The monocular pipeline uses only the left UAVStereo image as input. The same 15 residential frames from Week 1 were used so the monocular outputs could be compared directly against the stereo baseline. Since monocular depth is relative rather than metric by default, the MiDaS output was calibrated against the available UAVStereo disparity reference before conversion into depth proxy, obstacle mask, risk map, and collision-proxy decision outputs.

## Accomplishments

- Integrated a monocular depth pipeline using MiDaS Small.
- Used the same UAVStereo residential scenes as the Week 1 stereo baseline.
- Generated monocular disparity/depth outputs from the left camera only.
- Converted monocular outputs into obstacle and risk maps using the same Week 1 risk-map formulation.
- Produced mono-vs-stereo visual comparisons for all selected frames.
- Logged frame-level metrics and summary metrics for later comparison.

## Testing and Validation

Testing was conducted with an initial two-frame validation test followed by the full 15-frame Week 2 result set. The monocular outputs were checked against the Week 1 stereo baseline to confirm that both pipelines produce the same output structure.

| Validation Item | Result |
|---|---:|
| Evaluated frames | 15 |
| Frame IDs matched Week 1 stereo baseline | Yes |
| Monocular input source | Left camera image |
| Risk-map value range | 0.0 to 1.0 |
| Obstacle mask format | Binary 0/1 |
| Mono risk-map shape matched stereo | Yes |
| Mono obstacle-map shape matched stereo | Yes |

Summary metrics from the Week 2 run:

| Metric | Value |
|---|---:|
| Average MiDaS inference latency | 26.89 ms |
| Average valid estimate fraction | 0.117 |
| Average obstacle fraction | 0.117 |
| Average center alert fraction | 0.069 |
| Average risk agreement vs stereo at 0.5 threshold | 0.290 |
| Average obstacle IoU vs stereo | 0.052 |
| Average disparity MAE vs reference | 78.44 px |
| Average disparity RMSE vs reference | 85.80 px |

The results show that the monocular pipeline is structurally aligned with the stereo baseline and can generate the required obstacle/risk-map interface. However, the numerical agreement with stereo is still limited. This is expected because MiDaS estimates relative monocular depth, while stereo disparity provides stronger metric depth information. The Week 2 output is therefore a working monocular baseline, but stronger alignment and validation are needed before making safety-performance claims.

## Visuals

Visual evidence was generated as per-frame montages and a replayable video. Each montage shows:

- Left RGB input image
- MiDaS monocular disparity output
- Week 1 stereo risk-map baseline
- Monocular depth proxy
- Monocular risk-map decision
- Monocular obstacle mask

Selected visual to include in the report:

Figure 1: Monocular depth and risk-map output compared with the Week 1 stereo baseline.

`navigation/results/week02_monocular_depth/final/montage/frame_0040_week2_montage.png`

![Figure 1: Monocular depth and risk-map output compared with the Week 1 stereo baseline.](/Users/clintonimaro/Documents/Projects/uav-research/navigation/results/week02_monocular_depth/final/montage/frame_0040_week2_montage.png)

The generated visual outputs are saved under:

`navigation/results/week02_monocular_depth/final`

## Next Steps

For Week 3, the focus will be sensor alignment and validation. The next stage is to lock the shared evaluation scenes, normalize mono and stereo outputs, add stronger disparity/risk-map validation metrics, and generate comparison plots or heatmaps that explain where monocular depth agrees or disagrees with stereo.
