#!/bin/bash

# Brings all MSLesSeg files to the same level (directory level)


# Check if the user provided a path as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <PATH-to-train-ONLY>"
    exit 1
fi

# Assign the input argument to a variable
input_dir="$1"

# Check if the provided path is a valid directory
if [ ! -d "$input_dir" ]; then
    echo "Error: '$input_dir' is not a valid directory."
    exit 1
fi

# Move all files from subdirectories to the input directory
find "$input_dir" -mindepth 2 -type f -exec mv {} "$input_dir" \;

# Remove empty subdirectories
find "$input_dir" -mindepth 1 -type d -empty -exec rmdir {} \;

echo "All files have been moved to $input_dir, and empty directories removed."
