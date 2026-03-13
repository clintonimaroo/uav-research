EPISODES=$1
if [ -z "$EPISODES" ]; then
    EPISODES=50
fi

source ../venv/bin/activate

echo "========================================================="
echo "Running UAV Navigation Evaluation Suite"
echo "Episodes per test: $EPISODES"
echo "========================================================="
echo ""

echo "---------------------------------------------------------"
echo "TEST 1: Scaling Grid Sizes"
echo "---------------------------------------------------------"
for grid in 50 70 100; do
    echo ">> Running Grid Size ${grid}x${grid}..."
    python compare_methods.py --episodes $EPISODES --grid $grid --output_dir "comparison_results/grid_${grid}"
    echo ">> Finished Grid ${grid}"
done

echo "---------------------------------------------------------"
echo "TEST 2: Threshold Sensitivity Checks (Grid 50x50)"
echo "---------------------------------------------------------"
echo ">> Running Encounter Threshold 0.1..."
python compare_methods.py --episodes $EPISODES --grid 50 --encounter_threshold 0.1 --output_dir "comparison_results/thresh_enc01"
echo ">> Running Termination Threshold 0.7..."
python compare_methods.py --episodes $EPISODES --grid 50 --termination_threshold 0.7 --output_dir "comparison_results/thresh_term07"

echo "---------------------------------------------------------"
echo "TEST 3: Telemetry Robustness (Perception Noise)"
echo "---------------------------------------------------------"
echo ">> Running with 10% Perception Noise..."
python compare_methods.py --episodes $EPISODES --grid 50 --perception_noise 0.1 --output_dir "comparison_results/noise_01"

echo ">> Running with 20% Perception Noise..."
python compare_methods.py --episodes $EPISODES --grid 50 --perception_noise 0.2 --output_dir "comparison_results/noise_02"

echo "========================================================="
echo "All tests complete. Results saved to comparison_results/"
echo "========================================================="
