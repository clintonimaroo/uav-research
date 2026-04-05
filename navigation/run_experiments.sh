#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

EPISODES="${1:-50}"
COMPARISON_ROOT="comparison_results"
BASE_GRID=50
BASE_ENCOUNTER=0.2
BASE_TERMINATION=0.9
BASE_NOISE=0.0
BASE_REPLAN=0
BASE_CONFIDENCE_DECAY=0.95
BASE_OBSERVATION_RADIUS=2
COMPARE_EXTRA=(--low_memory --no-cache)

source ../venv/bin/activate

echo "========================================================="
echo "Running UAV Navigation Evaluation Suite (Denser Sweeps)"
echo "Episodes per test: ${EPISODES}"
echo "========================================================="
echo ""

mkdir -p "${COMPARISON_ROOT}"

echo "---------------------------------------------------------"
echo "BASELINE RUN (for bar plots)"
echo "---------------------------------------------------------"
python compare_methods.py \
    --episodes "${EPISODES}" \
    --grid "${BASE_GRID}" \
    --encounter_threshold "${BASE_ENCOUNTER}" \
    --termination_threshold "${BASE_TERMINATION}" \
    --perception_noise "${BASE_NOISE}" \
    --replan_frequency "${BASE_REPLAN}" \
    --confidence_decay "${BASE_CONFIDENCE_DECAY}" \
    --observation_radius "${BASE_OBSERVATION_RADIUS}" \
    --output_dir "${COMPARISON_ROOT}/baseline" \
    "${COMPARE_EXTRA[@]}"

echo "---------------------------------------------------------"
echo "TEST 1: Scaling Grid Sizes"
echo "---------------------------------------------------------"
for grid in 40 50 60 70 85 100; do
    echo ">> Running Grid Size ${grid}x${grid}..."
    python compare_methods.py \
        --episodes "${EPISODES}" \
        --grid "${grid}" \
        --encounter_threshold "${BASE_ENCOUNTER}" \
        --termination_threshold "${BASE_TERMINATION}" \
        --perception_noise "${BASE_NOISE}" \
        --replan_frequency "${BASE_REPLAN}" \
        --confidence_decay "${BASE_CONFIDENCE_DECAY}" \
        --observation_radius "${BASE_OBSERVATION_RADIUS}" \
        --output_dir "${COMPARISON_ROOT}/grid/grid_${grid}" \
        "${COMPARE_EXTRA[@]}"
done

echo "---------------------------------------------------------"
echo "TEST 2: Encounter Threshold Sweep"
echo "---------------------------------------------------------"
for enc in 0.10 0.15 0.20 0.25 0.30; do
    enc_label="${enc/./p}"
    echo ">> Running Encounter Threshold ${enc}..."
    python compare_methods.py \
        --episodes "${EPISODES}" \
        --grid "${BASE_GRID}" \
        --encounter_threshold "${enc}" \
        --termination_threshold "${BASE_TERMINATION}" \
        --perception_noise "${BASE_NOISE}" \
        --replan_frequency "${BASE_REPLAN}" \
        --confidence_decay "${BASE_CONFIDENCE_DECAY}" \
        --observation_radius "${BASE_OBSERVATION_RADIUS}" \
        --output_dir "${COMPARISON_ROOT}/encounter/enc_${enc_label}" \
        "${COMPARE_EXTRA[@]}"
done

echo "---------------------------------------------------------"
echo "TEST 3: Termination Threshold Sweep"
echo "---------------------------------------------------------"
for term in 0.70 0.80 0.85 0.90 0.95; do
    term_label="${term/./p}"
    echo ">> Running Termination Threshold ${term}..."
    python compare_methods.py \
        --episodes "${EPISODES}" \
        --grid "${BASE_GRID}" \
        --encounter_threshold "${BASE_ENCOUNTER}" \
        --termination_threshold "${term}" \
        --perception_noise "${BASE_NOISE}" \
        --replan_frequency "${BASE_REPLAN}" \
        --confidence_decay "${BASE_CONFIDENCE_DECAY}" \
        --observation_radius "${BASE_OBSERVATION_RADIUS}" \
        --output_dir "${COMPARISON_ROOT}/termination/term_${term_label}" \
        "${COMPARE_EXTRA[@]}"
done

echo "---------------------------------------------------------"
echo "TEST 4: Telemetry Robustness (Perception Noise)"
echo "---------------------------------------------------------"
for noise in 0.00 0.05 0.10 0.15 0.20; do
    noise_label="${noise/./p}"
    echo ">> Running with Perception Noise ${noise}..."
    python compare_methods.py \
        --episodes "${EPISODES}" \
        --grid "${BASE_GRID}" \
        --encounter_threshold "${BASE_ENCOUNTER}" \
        --termination_threshold "${BASE_TERMINATION}" \
        --perception_noise "${noise}" \
        --replan_frequency "${BASE_REPLAN}" \
        --confidence_decay "${BASE_CONFIDENCE_DECAY}" \
        --observation_radius "${BASE_OBSERVATION_RADIUS}" \
        --output_dir "${COMPARISON_ROOT}/noise/noise_${noise_label}" \
        "${COMPARE_EXTRA[@]}"
done

echo "---------------------------------------------------------"
echo "TEST 5: A* Replan Frequency Sweep"
echo "---------------------------------------------------------"
for replan in 0 1 2 4 8; do
    echo ">> Running A* Replan Frequency ${replan}..."
    python compare_methods.py \
        --episodes "${EPISODES}" \
        --grid "${BASE_GRID}" \
        --encounter_threshold "${BASE_ENCOUNTER}" \
        --termination_threshold "${BASE_TERMINATION}" \
        --perception_noise "${BASE_NOISE}" \
        --replan_frequency "${replan}" \
        --confidence_decay "${BASE_CONFIDENCE_DECAY}" \
        --observation_radius "${BASE_OBSERVATION_RADIUS}" \
        --output_dir "${COMPARISON_ROOT}/replan/replan_${replan}" \
        "${COMPARE_EXTRA[@]}"
done

echo "---------------------------------------------------------"
echo "TEST 6: Confidence Decay Sweep"
echo "---------------------------------------------------------"
for decay in 0.80 0.90 0.95 0.98 0.99; do
    decay_label="${decay/./p}"
    echo ">> Running Confidence Decay ${decay}..."
    python compare_methods.py \
        --episodes "${EPISODES}" \
        --grid "${BASE_GRID}" \
        --encounter_threshold "${BASE_ENCOUNTER}" \
        --termination_threshold "${BASE_TERMINATION}" \
        --perception_noise "${BASE_NOISE}" \
        --replan_frequency "${BASE_REPLAN}" \
        --confidence_decay "${decay}" \
        --observation_radius "${BASE_OBSERVATION_RADIUS}" \
        --output_dir "${COMPARISON_ROOT}/confidence_decay/decay_${decay_label}" \
        "${COMPARE_EXTRA[@]}"
done

echo "---------------------------------------------------------"
echo "TEST 7: Observation Radius Sweep"
echo "---------------------------------------------------------"
for radius in 1 2 3 4; do
    echo ">> Running Observation Radius ${radius}..."
    python compare_methods.py \
        --episodes "${EPISODES}" \
        --grid "${BASE_GRID}" \
        --encounter_threshold "${BASE_ENCOUNTER}" \
        --termination_threshold "${BASE_TERMINATION}" \
        --perception_noise "${BASE_NOISE}" \
        --replan_frequency "${BASE_REPLAN}" \
        --confidence_decay "${BASE_CONFIDENCE_DECAY}" \
        --observation_radius "${radius}" \
        --output_dir "${COMPARISON_ROOT}/observation_radius/radius_${radius}" \
        "${COMPARE_EXTRA[@]}"
done

echo "---------------------------------------------------------"
echo "Generating Group A figures from CSV outputs"
echo "---------------------------------------------------------"
python plot_paper1_groupA.py \
    --comparison_root "${COMPARISON_ROOT}" \
    --output_dir "${COMPARISON_ROOT}/paper1_groupA_figures"

echo "========================================================="
echo "All tests complete. Results saved under ${COMPARISON_ROOT}/"
echo "========================================================="
