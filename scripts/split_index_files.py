#!/usr/bin/env python3
"""
split_indices.py - Split existing index files into train/val/test based on filenames.

Usage:
    python split_indices.py --input_dir /path/to/indices --base_dir /path/to/processed_parquet
"""

import os
import json
import argparse
from collections import defaultdict

def split_indices(input_dir, base_dir):
    """
    Split existing index files into train/val/test based on filenames.
    
    Args:
        input_dir: Directory containing existing index files
        base_dir: Base directory containing train/val/test subdirectories
    """
    # Ensure the split directories exist
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(dir_path, 'indices'), exist_ok=True)
    
    # Load existing index files
    with open(os.path.join(input_dir, 'global_index.json'), 'r') as f:
        global_index = json.load(f)
    
    with open(os.path.join(input_dir, 'cluster_index.json'), 'r') as f:
        cluster_index = json.load(f)
    
    with open(os.path.join(input_dir, 'file_mapping.json'), 'r') as f:
        file_mapping = json.load(f)
    
    # Create reverse mapping for easier lookup
    file_paths = {i: path for i, path in file_mapping.items()}
    
    # Initialize split-based data structures
    train_samples = []
    val_samples = []
    test_samples = []
    
    train_file_indices = []
    val_file_indices = []
    test_file_indices = []
    
    train_global_samples = []
    val_global_samples = []
    test_global_samples = []
    
    train_cluster_index = defaultdict(list)
    val_cluster_index = defaultdict(list)
    test_cluster_index = defaultdict(list)
    
    train_file_mapping = {}
    val_file_mapping = {}
    test_file_mapping = {}
    
    # Process global index and create split-specific indices
    for i, sample in enumerate(global_index['samples']):
        file_idx = global_index['file_indices'][i]
        file_path = file_paths[str(file_idx)]
        
        # Determine which split this file belongs to
        if '/train/' in file_path or 'train_' in os.path.basename(file_path):
            split = 'train'
            train_global_samples.append(sample)
            train_file_indices.append(file_idx)
            train_file_mapping[file_idx] = file_path
            
            sample_info = {
                'file_idx': file_idx,
                'system_id': sample['system_id'],
                'cluster': sample['cluster']
            }
            train_samples.append(sample_info)
            
        elif '/val/' in file_path or 'val_' in os.path.basename(file_path):
            split = 'val'
            val_global_samples.append(sample)
            val_file_indices.append(file_idx)
            val_file_mapping[file_idx] = file_path
            
            sample_info = {
                'file_idx': file_idx,
                'system_id': sample['system_id'],
                'cluster': sample['cluster']
            }
            val_samples.append(sample_info)
            
        elif '/test/' in file_path or 'test_' in os.path.basename(file_path):
            split = 'test'
            test_global_samples.append(sample)
            test_file_indices.append(file_idx)
            test_file_mapping[file_idx] = file_path
            
            sample_info = {
                'file_idx': file_idx,
                'system_id': sample['system_id'],
                'cluster': sample['cluster']
            }
            test_samples.append(sample_info)
    
    # Process cluster indices and create split-specific cluster indices
    for cluster_id, samples in cluster_index.items():
        for sample in samples:
            file_idx = sample['file_idx']
            file_path = file_paths[str(file_idx)]
            
            # Determine which split this file belongs to
            if '/train/' in file_path or 'train_' in os.path.basename(file_path):
                train_cluster_index[cluster_id].append(sample)
            elif '/val/' in file_path or 'val_' in os.path.basename(file_path):
                val_cluster_index[cluster_id].append(sample)
            elif '/test/' in file_path or 'test_' in os.path.basename(file_path):
                test_cluster_index[cluster_id].append(sample)
    
    # Save train indices
    train_global_index = {
        'samples': train_global_samples,
        'file_indices': train_file_indices
    }
    
    with open(os.path.join(train_dir, 'indices', 'global_index.json'), 'w') as f:
        json.dump(train_global_index, f, indent=2)
    
    with open(os.path.join(train_dir, 'indices', 'index.json'), 'w') as f:
        json.dump(train_samples, f, indent=2)
    
    with open(os.path.join(train_dir, 'indices', 'cluster_index.json'), 'w') as f:
        json.dump(train_cluster_index, f, indent=2)
    
    with open(os.path.join(train_dir, 'indices', 'file_mapping.json'), 'w') as f:
        json.dump(train_file_mapping, f, indent=2)
    
    # Save val indices
    val_global_index = {
        'samples': val_global_samples,
        'file_indices': val_file_indices
    }
    
    with open(os.path.join(val_dir, 'indices', 'global_index.json'), 'w') as f:
        json.dump(val_global_index, f, indent=2)
    
    with open(os.path.join(val_dir, 'indices', 'index.json'), 'w') as f:
        json.dump(val_samples, f, indent=2)
    
    with open(os.path.join(val_dir, 'indices', 'cluster_index.json'), 'w') as f:
        json.dump(val_cluster_index, f, indent=2)
    
    with open(os.path.join(val_dir, 'indices', 'file_mapping.json'), 'w') as f:
        json.dump(val_file_mapping, f, indent=2)
    
    # Save test indices
    test_global_index = {
        'samples': test_global_samples,
        'file_indices': test_file_indices
    }
    
    with open(os.path.join(test_dir, 'indices', 'global_index.json'), 'w') as f:
        json.dump(test_global_index, f, indent=2)
    
    with open(os.path.join(test_dir, 'indices', 'index.json'), 'w') as f:
        json.dump(test_samples, f, indent=2)
    
    with open(os.path.join(test_dir, 'indices', 'cluster_index.json'), 'w') as f:
        json.dump(test_cluster_index, f, indent=2)
    
    with open(os.path.join(test_dir, 'indices', 'file_mapping.json'), 'w') as f:
        json.dump(test_file_mapping, f, indent=2)
    
    # Generate and save summaries
    train_summary = {
        'total_samples': len(train_global_samples),
        'total_clusters': len([k for k, v in train_cluster_index.items() if v]),
        'total_files': len(train_file_mapping),
        'clusters': {cluster: len(samples) for cluster, samples in train_cluster_index.items() if samples}
    }
    
    val_summary = {
        'total_samples': len(val_global_samples),
        'total_clusters': len([k for k, v in val_cluster_index.items() if v]),
        'total_files': len(val_file_mapping),
        'clusters': {cluster: len(samples) for cluster, samples in val_cluster_index.items() if samples}
    }
    
    test_summary = {
        'total_samples': len(test_global_samples),
        'total_clusters': len([k for k, v in test_cluster_index.items() if v]),
        'total_files': len(test_file_mapping),
        'clusters': {cluster: len(samples) for cluster, samples in test_cluster_index.items() if samples}
    }
    
    with open(os.path.join(train_dir, 'indices', 'summary.json'), 'w') as f:
        json.dump(train_summary, f, indent=2)
    
    with open(os.path.join(val_dir, 'indices', 'summary.json'), 'w') as f:
        json.dump(val_summary, f, indent=2)
    
    with open(os.path.join(test_dir, 'indices', 'summary.json'), 'w') as f:
        json.dump(test_summary, f, indent=2)
    
    # Generate overall summary
    overall_summary = {
        'total_samples': len(global_index['samples']),
        'train_samples': len(train_global_samples),
        'val_samples': len(val_global_samples),
        'test_samples': len(test_global_samples),
        'total_clusters': len(cluster_index),
        'total_files': len(file_mapping)
    }
    
    with open(os.path.join(base_dir, 'indices_summary.json'), 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    print(f"Created split indices with {overall_summary['total_samples']} total samples")
    print(f"Split distribution: Train: {overall_summary['train_samples']}, Val: {overall_summary['val_samples']}, Test: {overall_summary['test_samples']}")
    print(f"Index files saved to {train_dir}/indices, {val_dir}/indices, and {test_dir}/indices")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Split existing index files into train/val/test')
    parser.add_argument('--input_dir', default='/mnt/disk2/plinder/2024-06/v2/processed_parquet/indices', 
                        help='Directory containing existing index files')
    parser.add_argument('--base_dir', default='/mnt/disk2/plinder/2024-06/v2/processed_parquet', 
                        help='Base directory containing train/val/test subdirectories')
    
    args = parser.parse_args()
    
    # Split indices
    split_indices(args.input_dir, args.base_dir)

if __name__ == "__main__":
    main()