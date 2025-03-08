import os
import argparse
import logging
import pickle
import base64
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from docktgrid.molecule import MolecularComplex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deserialize_molecular_data(serialized_data):
    """Deserialize MolecularData from base64 encoded bytes"""
    pickled = base64.b64decode(serialized_data)
    return pickle.loads(pickled)

def serialize_molecular_data(data):
    """Serialize MolecularData to base64 encoded bytes"""
    pickled = pickle.dumps(data)
    return base64.b64encode(pickled)

def prune_complex(protein_data, ligand_data, max_dist=32.0):
    """Create and prune a molecular complex"""

    
    # Calculate ligand center from original ligand data
    ligand_center = torch.mean(ligand_data.coords, 1).to(dtype=torch.float32)
    
    # Prune protein atoms
    if protein_data.coords is not None:
        prot_coords = protein_data.coords.T
        prot_dists = torch.linalg.vector_norm(prot_coords - ligand_center, dim=1)
        prot_mask = prot_dists < max_dist
        
        # Apply masking
        protein_data.coords = protein_data.coords[:, prot_mask]
        protein_data.element_symbols = protein_data.element_symbols[prot_mask.numpy()]
        if hasattr(protein_data, 'vdw_radii'):
            protein_data.vdw_radii = protein_data.vdw_radii[prot_mask]
    
    return protein_data, ligand_data

def process_parquet_file(input_path, output_path, max_dist=32.0):
    df = pd.read_parquet(input_path)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        try:
            # Deserialize original data
            protein_data = deserialize_molecular_data(row['protein_data_serialized'])
            ligand_data = deserialize_molecular_data(row['ligand_data_serialized'])
            
            # Prune the complex
            pruned_protein, pruned_ligand = prune_complex(protein_data, ligand_data, max_dist)
            
            # Update serialized data
            row['protein_data_serialized'] = serialize_molecular_data(pruned_protein)
            row['ligand_data_serialized'] = serialize_molecular_data(pruned_ligand)
            
            df.loc[idx] = row
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            continue
    
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved pruned data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Prune distant protein atoms from Plinder parquet files')
    parser.add_argument('--input_dir', default='/mnt/disk2/plinder/2024-06/v2/processed_parquet/train',
                      help='Directory containing parquet files')
    parser.add_argument('--output_dir', default='/mnt/disk2/plinder/2024-06/v2/processed_parquet/train_pruned',
                      help='Output directory for pruned parquet files')
    parser.add_argument('--max_dist', type=float, default=24.0,
                      help='Maximum distance from ligand center (Angstroms)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all parquet files
    parquet_files = [f for f in os.listdir(args.input_dir) if f.endswith('.parquet')]
    
    for pq_file in parquet_files:
        input_path = os.path.join(args.input_dir, pq_file)
        output_path = os.path.join(args.output_dir, pq_file)
        
        logger.info(f"Processing {input_path}")
        process_parquet_file(input_path, output_path, args.max_dist)
    
    logger.info("Processing completed")

if __name__ == "__main__":
    main()