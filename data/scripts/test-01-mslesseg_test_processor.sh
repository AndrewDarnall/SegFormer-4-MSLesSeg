#!/bin/bash

# Processes The Test Set Accordingly

# Check if correct number of arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <PATH-to-test-dir> <Counter-Value>"
    exit 1
fi

# Get the directory path and counter from the arguments
dir_path="$1"
counter="$2"

# Check if the given path is a valid directory
if [ ! -d "$dir_path" ]; then
    echo "Error: '$dir_path' is not a valid directory."
    exit 1
fi

# Step 1: Iterate through all directories to rename them from PNUMBER to MSLS_$counter
for dir in "$dir_path"/*/; do
    if [ -d "$dir" ]; then
        # Extract the directory name
        dir_name=$(basename "$dir")

        # Check if the directory name starts with 'P' followed by numbers (PNUMBER)
        if [[ "$dir_name" =~ ^P[0-9]+$ ]]; then
            # Construct the new directory name as MSLS_$counter
            new_dir_name="MSLS_$counter"
            
            # Rename the directory
            mv "$dir" "$(dirname "$dir")/$new_dir_name"
            
            # Update the directory path to reflect the new name
            dir="$dir_path/$new_dir_name"
            
            # Increment the counter for the next directory
            ((counter++))
        fi
    fi
done

counter="$2"

# Step 2: Iterate through all directories again to rename files inside each directory
for dir in "$dir_path"/*/; do
    if [ -d "$dir" ]; then
        # Extract the directory name
        dir_name=$(basename "$dir")

        # Check if the directory name is MSLS_$counter
        if [[ "$dir_name" == "MSLS_$counter" ]]; then
            # Iterate through all files inside the directory
            for file in "$dir"*; do
                if [ -f "$file" ]; then
                    # Get the base name of the file (e.g., P59_FLAIR.nii.gz)
                    base_name=$(basename "$file")

                    # Extract the prefix (e.g., P59)
                    prefix="${base_name%%_*}"

                    # Construct the new prefix using MSLS_$counter
                    new_prefix="MSLS_$counter"

                    # Extract the rest of the name after the prefix (e.g., FLAIR.nii.gz)
                    rest_of_name="${base_name#*_}"

                    # Check if the remaining part contains 'MASK' or not
                    if [[ "$rest_of_name" == *"MASK"* ]]; then
                        # Replace 'MASK' with 'seg'
                        new_name="${new_prefix}_$(echo "$rest_of_name" | sed 's/MASK/seg/g')"
                    else
                        # Convert the remaining part to lowercase
                        new_name="${new_prefix}_$(echo "$rest_of_name" | tr '[:upper:]' '[:lower:]')"
                    fi

                    # Rename the file
                    mv "$file" "$dir/$new_name"
                fi
            done

            # Increment the counter for the next directory and files
            ((counter++))
        fi
    fi
done


echo "Directories and files have been renamed successfully."
