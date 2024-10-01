#!/bin/bash

# Default parameter values
dataset_name="citations"
model_name="DyGFormer"
patch_size=2
max_input_sequence_length=64
negative_sample_strategy="random"
num_runs=1
gpu=0

# Function to display help message
show_help() {
    echo "Usage: ./run_evaluate_link_prediction.sh [options]"
    echo
    echo "Options:"
    echo "  --dataset_name              Name of the dataset (default: $dataset_name)"
    echo "  --model_name                Name of the model (default: $model_name)"
    echo "  --patch_size                Patch size for the model (default: $patch_size)"
    echo "  --max_input_sequence_length Max input sequence length (default: $max_input_sequence_length)"
    echo "  --negative_sample_strategy  Strategy for negative sampling (default: $negative_sample_strategy)"
    echo "  --num_runs                  Number of runs (default: $num_runs)"
    echo "  --gpu                       GPU to use (default: $gpu)"
    echo "  --help                      Show this help message and exit"
    echo
    echo "Example:"
    echo "  ./run_evaluate_link_prediction.sh --dataset_name citations --model_name DyGFormer --num_runs 3"
    exit 0
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset_name) dataset_name="$2"; shift ;;
        --model_name) model_name="$2"; shift ;;
        --patch_size) patch_size="$2"; shift ;;
        --max_input_sequence_length) max_input_sequence_length="$2"; shift ;;
        --negative_sample_strategy) negative_sample_strategy="$2"; shift ;;
        --num_runs) num_runs="$2"; shift ;;
        --gpu) gpu="$2"; shift ;;
        --help) show_help ;;
        *) echo "Unknown parameter passed: $1"; show_help ;;
    esac
    shift
done

# Change to the dygnn/DyGLib directory
echo "Changing directory to dygnn/DyGLib/"
cd dygnn/DyGLib/ || { echo "Failed to change directory to dygnn/DyGLib/"; exit 1; }

# Run the evaluate_link_prediction.py script with the provided or default arguments
echo "=============================================="
echo " Running evaluate_link_prediction.py with the following arguments:"
echo " dataset_name=$dataset_name"
echo " model_name=$model_name"
echo " patch_size=$patch_size"
echo " max_input_sequence_length=$max_input_sequence_length"
echo " negative_sample_strategy=$negative_sample_strategy"
echo " num_runs=$num_runs"
echo " gpu=$gpu"
echo "=============================================="

python3 evaluate_link_prediction.py \
    --dataset_name $dataset_name \
    --model_name $model_name \
    --patch_size $patch_size \
    --max_input_sequence_length $max_input_sequence_length \
    --negative_sample_strategy $negative_sample_strategy \
    --num_runs $num_runs \
    --gpu $gpu

echo "=============================================="
echo " Evaluation script executed successfully."
echo "=============================================="

# Return to the original directory
cd - > /dev/null
