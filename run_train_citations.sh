#!/bin/bash

# Default parameter values
dataset_name="citations"
model_name="DyGFormer"
patch_size=2
max_input_sequence_length=64
num_runs=1
gpu=0
batch_size=256
num_epochs=5
num_walk_heads=2

# Function to display help message
show_help() {
    echo "Usage: ./run_train_link_prediction.sh [options]"
    echo
    echo "Options:"
    echo "  --dataset_name              Name of the dataset (default: $dataset_name)"
    echo "  --model_name                Name of the model (default: $model_name)"
    echo "  --patch_size                Patch size for the model (default: $patch_size)"
    echo "  --max_input_sequence_length Max input sequence length (default: $max_input_sequence_length)"
    echo "  --num_runs                  Number of runs (default: $num_runs)"
    echo "  --gpu                       GPU to use (default: $gpu)"
    echo "  --batch_size                Batch size for training (default: $batch_size)"
    echo "  --num_epochs                Number of epochs for training (default: $num_epochs)"
    echo "  --num_walk_heads            Number of walk heads for the model (default: $num_walk_heads)"
    echo "  --help                      Show this help message and exit"
    echo
    echo "Example:"
    echo "  ./run_train_link_prediction.sh --dataset_name citations --model_name DyGFormer --num_epochs 10"
    exit 0
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset_name) dataset_name="$2"; shift ;;
        --model_name) model_name="$2"; shift ;;
        --patch_size) patch_size="$2"; shift ;;
        --max_input_sequence_length) max_input_sequence_length="$2"; shift ;;
        --num_runs) num_runs="$2"; shift ;;
        --gpu) gpu="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --num_epochs) num_epochs="$2"; shift ;;
        --num_walk_heads) num_walk_heads="$2"; shift ;;
        --help) show_help ;;
        *) echo "Unknown parameter passed: $1"; show_help ;;
    esac
    shift
done

# Change to the dygnn/DyGLib directory
cd dygnn/DyGLib/ || { echo "Failed to change directory to dygnn/DyGLib/"; exit 1; }

# Run the train_link_prediction.py script with the provided or default arguments
echo "=============================================="
echo " Running train_link_prediction.py with the following arguments:"
echo " dataset_name=$dataset_name"
echo " model_name=$model_name"
echo " patch_size=$patch_size"
echo " max_input_sequence_length=$max_input_sequence_length"
echo " num_runs=$num_runs"
echo " gpu=$gpu"
echo " batch_size=$batch_size"
echo " num_epochs=$num_epochs"
echo " num_walk_heads=$num_walk_heads"
echo "=============================================="

python3 train_link_prediction.py \
    --dataset_name $dataset_name \
    --model_name $model_name \
    --patch_size $patch_size \
    --max_input_sequence_length $max_input_sequence_length \
    --num_runs $num_runs \
    --gpu $gpu \
    --batch_size $batch_size \
    --num_epochs $num_epochs \
    --num_walk_heads $num_walk_heads

echo "=============================================="
echo " Script executed successfully."
echo "=============================================="

# Return to the original directory
cd - > /dev/null
