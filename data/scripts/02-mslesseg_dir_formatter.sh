#!/bin/bash

# PreProcesses the MSLesSeg to remove the scans from the time period directories (T1, T2, T3)

# Check if the user provided a path as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <PATH-to-MSLesSeg-train-or-test-data>"
    exit 1
fi

# Assign the input argument to a variable
input_dir="$1"

# Check if the provided path is a valid directory
if [ ! -d "$input_dir" ]; then
    echo "Error: '$input_dir' is not a valid directory."
    exit 1
fi

# Loop through each P* directory (P7, P8, P9, ...) inside the given input directory
for p_dir in "$input_dir"/*/; do
    # Check if it's a directory starting with "P"
    if [[ -d "$p_dir" ]]; then
        # Loop through each T* subdirectory (T1, T2, T3, ...) inside the P* directory
        for t_dir in "$p_dir"*/; do
            # Check if it's a directory starting with "T"
            if [[ -d "$t_dir" ]]; then
                # Move all .nii.gz files from the T directory to the parent P directory
                mv "$t_dir"/*.nii.gz "$p_dir"
                # Remove the empty T directory
                rmdir "$t_dir"
            fi
        done
    fi
done

echo "Files moved and empty T directories removed."

