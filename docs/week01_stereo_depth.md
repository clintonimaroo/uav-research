# Weekly Research Report

**Week Ending:** April 03, 2026  
**Researcher:** Clinton Imaro  
**Project Title:** Performance Evaluation of Monocular and Stereo Vision for Autonomous UAV Collision Avoidance in Disaster Response Environments

## Objectives (Week 1 Checklist)

- Acquire an appropriate UAV stereo dataset.
- Create a stereo depth pipeline using OpenCV.
- Generate disparity and depth maps from stereo imagery.
- Convert depth into a first-person obstacle/risk map.
- Establish a mathematically defined collision proxy.
- Freeze the pipeline as the baseline stereo -> depth -> obstacle configuration.

## Summary of Work for the Week

This week focused on establishing the foundational stereo vision baseline for Paper 2. The main objective was to use a UAV-centered stereo dataset and implement a repeatable pipeline that converts raw left/right stereo imagery into disparity, depth proxy, obstacle masks, and normalized collision-risk maps.

The UAVStereo residential sequence `ZM-Baseline-25` was used as the Week 1 production dataset. The final run processed 15 synchronized stereo frames from `0010` through `0150`. The output package is saved under `navigation/results/week01_stereo_depth/final`.

## Accomplishments

**Dataset Integration:** Integrated the UAVStereo residential dataset through the local `data/uav_stereo` path. The Week 1 run uses synchronized left/right image pairs and available `.pfm` ground-truth disparity files.

**Stereo Pipeline Development:** Implemented the stereo baseline in `navigation/stereo_disparity.py` using OpenCV StereoSGBM. The supporting `.pfm` loader is implemented in `navigation/read_pfm.py`.

**Disparity and Depth Generation:** Generated stereo disparity arrays, visual disparity maps, and depth proxy outputs for all 15 Week 1 frames.

**Risk Map Formulation:** Converted the estimated depth proxy into normalized obstacle-risk maps using:

`risk = clip(collision_threshold_m / depth, 0, 1)`

**Collision Proxy Definition:** Defined the Week 1 binary collision proxy using a fixed 2.5 m distance threshold:

`obstacle = depth < 2.5 m`

**Pipeline Freezing:** Froze the Week 1 baseline as:

`stereo image pair -> StereoSGBM disparity -> depth proxy -> obstacle/risk map`

## Testing and Validation

Validation was performed against the generated Week 1 production artifacts.

| Check | Result |
|---|---:|
| Frames processed | 15 |
| Per-frame metric rows | 15 |
| Raw array outputs | 60 |
| Depth map images | 15 |
| Disparity images | 30 |
| Left/right frame images | 30 |
| Montage images | 15 |
| Obstacle mask images | 15 |
| Risk map images | 15 |
| Metrics/config files | 3 |
| Demo video | 1 |

The final metrics CSV contains unique frame IDs from `0010` through `0150` with no `NaN` metric values. The summary output reports an average disparity MAE of `42.6750` px, average valid estimate fraction of `0.6653`, average obstacle fraction of `0.6653`, and average inference latency of `11.8679` ms for the OpenCV StereoSGBM baseline.

Compile validation passed with:

`python3 -m py_compile navigation/read_pfm.py navigation/stereo_disparity.py`

The Week 1 source files were also checked to confirm they contain no source comments or docstrings.

## Visuals

The Week 1 visual evidence is saved in:

- `navigation/results/week01_stereo_depth/final/week01_stereo_depth_demo.mp4`
- `navigation/results/week01_stereo_depth/final/montage/`
- `navigation/results/week01_stereo_depth/final/disparity/`
- `navigation/results/week01_stereo_depth/final/depth/`
- `navigation/results/week01_stereo_depth/final/risk/`
- `navigation/results/week01_stereo_depth/final/obstacles/`

Each montage frame shows the left RGB input, right RGB input, StereoSGBM disparity, depth proxy, obstacle-risk decision view, and collision proxy mask.

## Next Steps

Move to Week 2 by integrating the monocular depth baseline using Intel MiDaS Small on the left image only. Week 2 should reuse the Week 1 stereo output as the comparison baseline and add monocular depth/risk outputs with scale calibration against the available UAVStereo disparity reference.
