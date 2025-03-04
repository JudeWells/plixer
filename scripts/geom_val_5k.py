#!/usr/bin/env python3
import os
import random
import shutil
from pathlib import Path

# Define source and destination directories
source_dir = "/mnt/disk2/VoxelDiffOuter/geom/rdkit_folder/drugs/val"
dest_dir = "/mnt/disk2/VoxelDiffOuter/geom/rdkit_folder/drugs/val_5k"

def main():
    # Create destination directory if it doesn't exist
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all pickle files from source directory
    pickle_files = [f for f in os.listdir(source_dir) if f.endswith('.pickle')]
    
    # Check if we have enough files
    total_files = len(pickle_files)
    if total_files < 5000:
        print(f"Warning: Only {total_files} pickle files found, which is less than 5000")
        sample_size = total_files
    else:
        sample_size = 5000
    
    # Randomly sample 5000 files
    sampled_files = random.sample(pickle_files, sample_size)
    
    # Copy the sampled files to the destination directory
    for i, file in enumerate(sampled_files, 1):
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(dest_dir, file)
        shutil.copy2(src_path, dst_path)
        
        # Print progress every 500 files
        if i % 500 == 0:
            print(f"Progress: {i}/{sample_size} files copied ({i/sample_size*100:.1f}%)")
    
    print(f"Successfully copied {sample_size} random pickle files to {dest_dir}")

if __name__ == "__main__":
    main()