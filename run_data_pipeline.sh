#!/bin/bash

# Default argument values
n=10000000
year_start=2010
n_components=170
n_col=172

# Function to display help message
show_help() {
    echo "Usage: ./run_pipeline.sh [options]"
    echo
    echo "Options:"
    echo "  --n                Number of entries to process (default: $n)"
    echo "  --year_start       Starting year for data processing (default: $year_start)"
    echo "  --n_components     Number of components for dimensionality reduction (default: $n_components)"
    echo "  --n_col            Number of columns for the node features matrix (default: $n_col)"
    echo "  --help             Show this help message and exit"
    echo
    echo "Example:"
    echo "  bash run_pipeline.sh --n 5000000 --year_start 2015 --n_components 200 --n_col 180"
    exit 0
}

# Parse command-line arguments
if [[ "$#" -eq 0 ]]; then
    show_help
fi

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --n) n="$2"; shift ;;
        --year_start) year_start="$2"; shift ;;
        --n_components) n_components="$2"; shift ;;
        --n_col) n_col="$2"; shift ;;
        --help) show_help ;;
        *) echo "Unknown parameter passed: $1"; show_help ;;
    esac
    shift
done

# Run process_raw_data.py
echo "=============================================="
echo " Starting: process_raw_data.py"
echo " Arguments: n=$n, year_start=$year_start"
echo "=============================================="
python3 dygnn/dataset/process_raw_data.py --n $n --year_start $year_start
echo " Finished: process_raw_data.py"
echo

# Run create_edge_features.py
echo "=============================================="
echo " Starting: create_edge_features.py"
echo " Arguments: n_components=$n_components"
echo "=============================================="
python3 dygnn/dataset/create_edge_features.py --n_components $n_components
echo " Finished: create_edge_features.py"
echo

# Run create_edge_index_node_feat.py
echo "=============================================="
echo " Starting: create_edge_index_node_feat.py"
echo " Arguments: n_col=$n_col"
echo "=============================================="
python3 dygnn/dataset/create_edge_index_node_feat.py --n_col $n_col
echo " Finished: create_edge_index_node_feat.py"
echo

echo "=============================================="
echo " All scripts executed successfully."
echo "=============================================="
