#!/usr/bin/env python3
"""
generate_parquet_indices.py - Create index files for a directory of processed parquet files.

Usage:
    python generate_parquet_indices.py --input_dir /path/to/processed_parquets [--output_dir /path/to/output]

If output_dir is not specified, indices will be created in a subdirectory called 'indices' within the input directory.
"""

import os
import glob
import pandas as pd
import json
import argparse
from collections import defaultdict
from tqdm import tqdm

def create_indices(input_dir, output_dir):
    """
    Create index files for faster dataset loading.
    
    Args:
        input_dir: Directory containing processed parquet files
        output_dir: Directory to save index files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all parquet files in the input directory
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files to process")
    
    # Create global indices
    all_samples = []
    file_indices = []
    
    # Create cluster-based indices
    cluster_samples = defaultdict(list)
    
    # Process each parquet file
    for file_idx, file_path in enumerate(tqdm(parquet_files, desc="Processing files")):
        try:
            # Read the parquet file
            chunk_df = pd.read_parquet(file_path)
            
            # Process each row
            for idx, row in chunk_df.iterrows():
                sample = {
                    'system_id': row['system_id'],
                    'smiles': row['smiles'],
                    'weight': float(row.get('weight', 1.0)),
                    'cluster': row.get('cluster', '0'),  # Keep cluster as string
                }
                
                all_samples.append(sample)
                file_indices.append(file_idx)
                
                # Add to cluster-based indices
                cluster_samples[sample['cluster']].append({
                    'sample_idx': len(all_samples) - 1,
                    'file_idx': file_idx,
                    'system_id': sample['system_id'],
                    'smiles': sample['smiles'],
                    'weight': sample['weight']
                })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save global indices
    global_index = {
        'samples': all_samples,
        'file_indices': file_indices
    }
    
    with open(os.path.join(output_dir, 'global_index.json'), 'w') as f:
        json.dump(global_index, f)
    
    # Save cluster-based indices
    cluster_index = {str(cluster_id): samples for cluster_id, samples in cluster_samples.items()}
    
    with open(os.path.join(output_dir, 'cluster_index.json'), 'w') as f:
        json.dump(cluster_index, f)
    
    # Save file mapping
    file_mapping = {
        i: file_path for i, file_path in enumerate(parquet_files)
    }
    
    with open(os.path.join(output_dir, 'file_mapping.json'), 'w') as f:
        json.dump(file_mapping, f)
    
    # Generate summary
    summary = {
        'total_samples': len(all_samples),
        'total_clusters': len(cluster_samples),
        'total_files': len(parquet_files),
        'clusters': {cluster: len(samples) for cluster, samples in cluster_samples.items()}
    }
    
    with open(os.path.join(output_dir, 'index_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Created indices with {len(all_samples)} total samples across {len(cluster_samples)} clusters")
    print(f"Index files saved to {output_dir}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate index files for processed parquet files')
    parser.add_argument('--input_dir', default='/mnt/disk2/plinder/2024-06/v2/processed_parquet', help='Directory containing processed parquet files')
    parser.add_argument('--output_dir', default='/mnt/disk2/plinder/2024-06/v2/processed_parquet/indices', help='Directory to save index files (default: input_dir/indices)')
    
    args = parser.parse_args()
    
    # Set output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, 'indices')
    
    # Create indices
    create_indices(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()