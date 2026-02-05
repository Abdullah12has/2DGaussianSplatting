#!/bin/bash
#
# Batch Evaluation Script for 2D Gaussian Splatting
# Compares multiple model outputs: baseline, depth reinit, monosdf, combined
#

# Configuration
DATASET="/cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr"
OUTPUT_DIR="./output"
RESULTS_FILE="evaluation_results.csv"

# Model paths
BASELINE="${OUTPUT_DIR}/baseline"
DEPTH_REINIT="${OUTPUT_DIR}/depth_reinit"
MONOSDF="${OUTPUT_DIR}/monosdf"
COMBINED="${OUTPUT_DIR}/combined"

# Check if models exist
echo "Checking model paths..."
missing=0
for model in "$BASELINE" "$DEPTH_REINIT" "$MONOSDF" "$COMBINED"; do
    if [ ! -d "$model" ]; then
        echo "  Warning: $model not found"
        missing=$((missing + 1))
    else
        echo "  ✓ Found: $model"
    fi
done

if [ $missing -eq 4 ]; then
    echo ""
    echo "Error: No model output directories found!"
    echo "Please train models first or adjust paths in this script."
    exit 1
fi

echo ""
echo "Starting evaluation..."
echo "Dataset: $DATASET"
echo "Output:  $RESULTS_FILE"
echo ""

# Run evaluation
python 2dGScode/evaluate.py \
  -m "$BASELINE" "$DEPTH_REINIT" "$MONOSDF" "$COMBINED" \
  --names "Baseline" "Task2-DepthReinit" "Task4-MonoSDF" "Task2+4-Combined" \
  -s "$DATASET" \
  --output "$RESULTS_FILE"

echo ""
echo "Evaluation complete! Results saved to: $RESULTS_FILE"
