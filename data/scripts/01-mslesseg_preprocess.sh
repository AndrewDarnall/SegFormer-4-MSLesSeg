#!/bin/bash



# First Pre-Process for extending the patients #


# Check if the user provided the root directory
if [ -z "$2" ]; then
    echo "Usage: $0 <PATH-to-train-or-test-dir> <Initial-Counter-Value>"
    exit 1
fi

root_dir="$1"
count="$2"

# Iterate through all directories starting with "P"
for p_dir in "$root_dir"/P*; do
    if [ -d "$p_dir" ]; then
        # Iterate through all subdirectories starting with "T"
        for t_dir in "$p_dir"/T*; do
            if [ -d "$t_dir" ]; then
                # Rename files within the T directories
                for file in "$t_dir"/*; do
                    if [ -f "$file" ]; then
                        # Extract filename and extension
                        filename=$(basename "$file")
                        extension="${filename##*.}"
                        filename_no_ext="${filename%.*}"
                        
                        # Extract the substring before the last underscore (P9_T1_ or P8_T2_ etc.)
                        prefix=$(echo "$filename_no_ext" | sed -E 's/^(P[0-9]+_T[0-9]+)_.*$/\1/')
                        
                        # Construct the new filename
                        new_filename=$(echo "$filename" | sed "s/^$prefix/MSLS_$count/")

                        # Rename the file
                        mv "$file" "$t_dir/$new_filename"
                    fi
                done
                
                # Increment the counter after processing a T directory
                ((count++))
            fi
        done
    fi
done

echo "Renaming completed."
