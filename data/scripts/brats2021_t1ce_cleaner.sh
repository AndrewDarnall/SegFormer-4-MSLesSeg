#!/bin/bash


# Removes the .t1ce modality from the MRI scans #


# Check if the user provided a path as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <PATH-to-dir-with-t1ce>"
    exit 1
fi

# Assign the input argument to a variable
input_dir="$1"

# Check if the provided path is a valid directory
if [ ! -d "$input_dir" ]; then
    echo "Error: '$input_dir' is not a valid directory."
    exit 1
fi

# Iterate through all subdirectories and files within the given directory
find "$input_dir" -type f | while read file; do
    # Check if the filename contains 't1ce'
    if [[ "$(basename "$file")" == *"t1ce"* ]]; then
        echo "Deleting file: $file"
        rm "$file"
    fi
done

echo "Files containing 't1ce' have been deleted."
