#!/bin/bash

# Function to find and decompress all .gz files
decompress_nii_gz() {
  local directory="$1"
  
  # Find all .nii.gz files and decompress them in place
  find "$directory" -type f -name "*.nii.gz" | while read -r file; do
    echo "Decompressing: $file"
    gunzip -k "$file"  # -k keeps the original compressed file
  done
}

# Run the function with the first argument as the directory
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

decompress_nii_gz "$1"

