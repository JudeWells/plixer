import os
import glob
import pandas as pd
import numpy as np
import torch
import tempfile
import base64
import pickle
import json
import time
from tqdm import tqdm
import sys
import traceback
from collections import defaultdict

# Import necessary modules from your codebase
from src.data.poc2mol.datasets import ParquetDataset
from src.data.common.voxelization.config import Poc2MolDataConfig

# Set up configuration
config = Poc2MolDataConfig()
config.random_rotation = False
config.random_translation = 0.0
config.remove_hydrogens = True

# Define paths
input_dir = "/mnt/disk2/plinder/2024-06/v2/preprocessed_parquet"
output_dir = "/mnt/disk2/plinder/2024-06/v2/processed_parquet"
index_dir = os.path.join(output_dir, "indices")
error_log_dir = os.path.join(output_dir, "error_logs")

# Create output directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(index_dir, exist_ok=True)
os.makedirs(error_log_dir, exist_ok=True)

# Target file size in bytes (150MB)
TARGET_SIZE = 150 * 1024 * 1024

def validate_sample(row, dataset_instance):
    """
    Validate that a sample can be successfully processed by the dataset's __getitem__ method.
    
    Args:
        row: DataFrame row containing sample data
        dataset_instance: Instance of ParquetDataset
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Create a temporary sample and file_indices
        sample = {
            'system_id': row['system_id'],
            'smiles': row['smiles'],
            'weight': row.get('weight', 1.0),
            'cluster': row.get('cluster', '0'),  # Keep cluster as string
        }
        
        # Create a temporary dataframe with just this row
        temp_df = pd.DataFrame([row])
        
        # Mock the _get_dataframe method to return our temporary dataframe
        original_get_dataframe = dataset_instance._get_dataframe
        dataset_instance._get_dataframe = lambda _: temp_df
        
        # Set up a mock file_indices and samples
        original_file_indices = dataset_instance.file_indices
        original_samples = dataset_instance.samples
        dataset_instance.file_indices = [0]
        dataset_instance.samples = [sample]
        
        # Try to process the sample
        result = dataset_instance.__getitem__(0)
        
        # Check if protein and ligand voxels have valid data
        if result['ligand'] is None or (result['protein'] is None and config.has_protein):
            return False, "Missing protein or ligand voxel data"
        
        # Check if voxels have non-zero values (indicating presence of atoms)
        if torch.sum(result['ligand']) == 0 or (config.has_protein and torch.sum(result['protein']) == 0):
            return False, "Empty protein or ligand voxel (no atoms detected)"
        
        # Restore original methods and attributes
        dataset_instance._get_dataframe = original_get_dataframe
        dataset_instance.file_indices = original_file_indices
        dataset_instance.samples = original_samples
        
        return True, ""
    
    except Exception as e:
        # Restore original methods and attributes
        dataset_instance._get_dataframe = original_get_dataframe if 'original_get_dataframe' in locals() else dataset_instance._get_dataframe
        dataset_instance.file_indices = original_file_indices if 'original_file_indices' in locals() else dataset_instance.file_indices
        dataset_instance.samples = original_samples if 'original_samples' in locals() else dataset_instance.samples
        
        return False, str(e)

def process_parquet_file(file_path, output_dir, dataset_instance):
    """
    Process a single parquet file according to the requirements.
    
    Args:
        file_path: Path to the parquet file
        output_dir: Directory to save processed files
        dataset_instance: Instance of ParquetDataset for validation
    
    Returns:
        dict: Statistics about the processing
    """
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]
    
    print(f"Processing {file_path}...")
    
    # Read the parquet file
    df = pd.read_parquet(file_path)
    original_count = len(df)
    
    # Shuffle the rows
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Drop unnecessary columns if they exist
    columns_to_drop = []
    if 'pdb_content' in df.columns:
        columns_to_drop.append('pdb_content')
    if 'mol_block' in df.columns:
        columns_to_drop.append('mol_block')
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    
    # Validate each row
    valid_rows = []
    error_logs = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating rows"):
        is_valid, error_message = validate_sample(row, dataset_instance)
        
        if is_valid:
            valid_rows.append(idx)
        else:
            error_logs.append({
                'system_id': row['system_id'],
                'smiles': row['smiles'],
                'error': error_message
            })
    
    # Keep only valid rows
    df = df.iloc[valid_rows].reset_index(drop=True)
    
    # Write error log
    error_log_path = os.path.join(error_log_dir, f"{base_name}_errors.json")
    with open(error_log_path, 'w') as f:
        json.dump(error_logs, f, indent=2)
    
    # Check file size and split if necessary
    df_size = df.memory_usage(deep=True).sum()
    num_chunks = max(1, int(np.ceil(df_size / TARGET_SIZE)))
    
    chunk_size = len(df) // num_chunks
    if chunk_size == 0:
        chunk_size = 1
    
    # Create a list to store information about each chunk
    chunk_info = []
    
    # Split and save chunks
    for i in range(0, len(df), chunk_size):
        chunk_df = df.iloc[i:i+chunk_size]
        chunk_name = f"{base_name}_chunk{i//chunk_size}.parquet"
        chunk_path = os.path.join(output_dir, chunk_name)
        
        # Save the chunk
        chunk_df.to_parquet(chunk_path, index=False)
        
        # Store information about this chunk
        chunk_info.append({
            'file_name': chunk_name,
            'num_samples': len(chunk_df),
            'system_ids': chunk_df['system_id'].tolist(),
            'smiles': chunk_df['smiles'].tolist()
        })
    
    # Return statistics
    return {
        'original_file': file_name,
        'original_count': original_count,
        'valid_count': len(df),
        'invalid_count': original_count - len(df),
        'num_chunks': num_chunks,
        'chunks': chunk_info
    }

def create_indices(processed_files_info):
    """
    Create index files for faster dataset loading.
    
    Args:
        processed_files_info: List of dictionaries with information about processed files
    """
    # Create global indices
    all_samples = []
    file_indices = []
    
    # Create cluster-based indices
    cluster_samples = defaultdict(list)
    
    for file_idx, file_info in enumerate(processed_files_info):
        for chunk_info in file_info['chunks']:
            chunk_file = os.path.join(output_dir, chunk_info['file_name'])
            chunk_df = pd.read_parquet(chunk_file)
            
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
    
    # Save global indices
    global_index = {
        'samples': all_samples,
        'file_indices': file_indices
    }
    
    with open(os.path.join(index_dir, 'global_index.json'), 'w') as f:
        json.dump(global_index, f)
    
    # Save cluster-based indices
    cluster_index = {str(cluster_id): samples for cluster_id, samples in cluster_samples.items()}
    
    with open(os.path.join(index_dir, 'cluster_index.json'), 'w') as f:
        json.dump(cluster_index, f)
    
    # Save file mapping
    file_mapping = {
        i: os.path.join(output_dir, info['chunks'][0]['file_name']) 
        for i, info in enumerate(processed_files_info)
    }
    
    with open(os.path.join(index_dir, 'file_mapping.json'), 'w') as f:
        json.dump(file_mapping, f)
    
    print(f"Created indices with {len(all_samples)} total samples across {len(cluster_samples)} clusters")

def main():
    # Get all parquet files in the input directory
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files to process")
    
    # Create a dataset instance for validation
    # We'll use a temporary directory as data_path since we'll mock the _get_dataframe method
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a dummy parquet file in the temp directory
        dummy_df = pd.DataFrame({
            'system_id': ['dummy'],
            'smiles': ['C']
        })
        dummy_path = os.path.join(temp_dir, 'dummy.parquet')
        dummy_df.to_parquet(dummy_path)
        
        # Create dataset instance
        dataset_instance = ParquetDataset(
            config=config,
            data_path=temp_dir,
            translation=0.0,
            rotate=False
        )
        
        # Process each parquet file
        processed_files_info = []
        
        for file_path in parquet_files:
            try:
                file_info = process_parquet_file(file_path, output_dir, dataset_instance)
                processed_files_info.append(file_info)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                traceback.print_exc()
        
        # Create indices
        create_indices(processed_files_info)
    
    # Save processing summary
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input_directory': input_dir,
        'output_directory': output_dir,
        'files_processed': len(processed_files_info),
        'total_valid_samples': sum(info['valid_count'] for info in processed_files_info),
        'total_invalid_samples': sum(info['invalid_count'] for info in processed_files_info),
        'file_details': processed_files_info
    }
    
    with open(os.path.join(output_dir, 'processing_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Processing complete!")
    print(f"Total valid samples: {summary['total_valid_samples']}")
    print(f"Total invalid samples: {summary['total_invalid_samples']}")
    print(f"Summary saved to {os.path.join(output_dir, 'processing_summary.json')}")

if __name__ == "__main__":
    main()