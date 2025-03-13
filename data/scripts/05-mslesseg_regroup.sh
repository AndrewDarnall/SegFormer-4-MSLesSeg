#!/bin/bash

# Performs the final regrouping of the processed files

# Check if a directory path is provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <PATH-to-train-or-test-dir> <Counter-Variable>"
    exit 1
fi

# Set the directory path
dir_path="$1"
counter="$2"

# Check if the given path is a valid directory
if [ ! -d "$dir_path" ]; then
    echo "Error: '$dir_path' is not a valid directory."
    exit 1
fi

# Iterate over files in the directory
for file in "$dir_path"/MSLS_*_*; do
    if [[ -f "$file" ]]; then
        # Extract the counter value
        base_name=$(basename "$file")
        counter=$(echo "$base_name" | grep -oP 'MSLS_\K\d+')

        if [[ -n "$counter" ]]; then
            # Define the target directory
            target_dir="$dir_path/MSLS_0$counter"

            # Create the directory if it doesn't exist
            mkdir -p "$target_dir"

            # Move all matching files into the directory
            mv "$dir_path/MSLS_${counter}_"* "$target_dir/"
        fi
    fi
done

echo "Files have been organized into respective directories."
