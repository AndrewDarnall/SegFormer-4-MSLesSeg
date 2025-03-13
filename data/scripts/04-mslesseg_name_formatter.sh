#!/bin/bash

# ================================================ #
#             Changes MASK --> seg                 #
#         Maintains MSLS_ prefix in all caps       #
#        and lowercases the rest of the name       #
# ================================================ # 

# Check if the user provided a path as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <PATH-to-train-or-test-dir>"
    exit 1
fi

# Assign the input argument to a variable
input_dir="$1"

# Check if the provided path is a valid directory
if [ ! -d "$input_dir" ]; then
    echo "Error: '$input_dir' is not a valid directory."
    exit 1
fi

# Walk through each subdirectory and file within the given directory
find "$input_dir" -type f | while read file; do
    # Get the filename without the path
    filename=$(basename "$file")

    # Ensure the filename starts with MSLS_ (skip if it doesn’t)
    if [[ "$filename" != MSLS_* ]]; then
        echo "Skipping: $filename (does not start with MSLS_)"
        continue
    fi

    # Remove MSLS_ from the filename temporarily for processing
    name_without_prefix="${filename#MSLS_}"

    # Replace 'MASK' with 'seg' and lowercase the rest of the name
    processed_name=$(echo "$name_without_prefix" | sed 's/MASK/seg/g' | tr 'A-Z' 'a-z')

    # Construct the final filename with MSLS_ prefix intact
    new_filename="MSLS_$processed_name"

    # If the new filename is different, rename the file
    if [[ "$filename" != "$new_filename" ]]; then
        mv "$file" "$(dirname "$file")/$new_filename"
        echo "Renamed: $filename -> $new_filename"
    fi
done

echo "Files renamed successfully."
