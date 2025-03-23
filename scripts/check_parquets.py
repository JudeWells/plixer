#!/usr/bin/env python
# Script to check parquet files for 'mol_block' column and minimum length

import os
import pandas as pd
from pathlib import Path
import argparse

def check_parquet_directory(directory_path):
    """
    Check all parquet files in a directory for 'mol_block' column and minimum length.
    
    Args:
        directory_path (str): Path to directory containing parquet files
    
    Returns:
        tuple: (total_files, invalid_files, error_files)
    """
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Directory does not exist: {directory_path}")
        return 0, 0, 0
    
    total_files = 0
    invalid_files = 0
    error_files = 0
    
    for file_path in directory.glob("**/*.parquet"):
        total_files += 1
        try:
            # Read the parquet file metadata without loading all data
            parquet_file = pd.read_parquet(file_path, columns=None)
            
            # Check if 'mol_block' column exists and file has at least one row
            if 'mol_block' not in parquet_file.columns or len(parquet_file) < 1:
                print(f"Invalid file: {file_path}")
                print(f"  - Has 'mol_block' column: {'mol_block' in parquet_file.columns}")
                print(f"  - Number of rows: {len(parquet_file)}")
                invalid_files += 1
                
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            error_files += 1
    
    return total_files, invalid_files, error_files

def main():
    # Directories to check
    directories = [
        "../PDBbind_v2020_refined_set_mol_parquets",
        "../zinc20_parquet",
        "../geom/rdkit_folder/drugs/train_parquet_v2"
    ]
    
    print("Checking parquet files in specified directories...")
    
    for directory in directories:
        print(f"\nProcessing directory: {directory}")
        total, invalid, errors = check_parquet_directory(directory)
        
        print(f"Summary for {directory}:")
        print(f"  - Total parquet files: {total}")
        print(f"  - Files without 'mol_block' column or empty: {invalid}")
        print(f"  - Files with processing errors: {errors}")
        print(f"  - Valid files: {total - invalid - errors}")
        
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()