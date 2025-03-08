import os
import argparse
import logging
import pickle
import base64
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from docktgrid.molparser import MolecularData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deserialize_molecular_data(serialized_data):
    """Deserialize MolecularData from base64 encoded bytes"""
    return pickle.loads(base64.b64decode(serialized_data))

def process_parquet_file(input_path, output_path):
    df = pd.read_parquet(input_path)
    
    # Process each row
    new_rows = []
    for _, row in df.iterrows():
        try:
            # Deserialize and extract data
            protein_data = deserialize_molecular_data(row['protein_data_serialized'])
            ligand_data = deserialize_molecular_data(row['ligand_data_serialized'])
            
            # Convert to numpy arrays
            new_row = {
                'system_id': row['system_id'],
                'smiles': row['smiles'],
                'protein_coords': protein_data.coords.numpy().astype(np.float16).flatten(),
                'protein_coords_shape': protein_data.coords.shape,
                'protein_element_symbols': protein_data.element_symbols.astype('U'),
                'ligand_coords': ligand_data.coords.numpy().astype(np.float16).flatten(),
                'ligand_coords_shape': ligand_data.coords.shape,
                'ligand_element_symbols': ligand_data.element_symbols.astype('U')
            }
            new_rows.append(new_row)
        except Exception as e:
            logger.error(f"Error processing row: {e}")
            continue
    
    # Create new dataframe and save
    new_df = pd.DataFrame(new_rows)
    new_df.to_parquet(output_path, index=False)
    

def main():
    parser = argparse.ArgumentParser(description='Convert Plinder parquet format')
    parser.add_argument('--input_dir', default="/mnt/disk2/plinder/2024-06/v2/processed_parquet/test",
                      help='Directory containing original parquet files')
    parser.add_argument('--output_dir', default="/mnt/disk2/plinder/2024-06/v2/processed_parquet/test_arrays",
                      help='Output directory for converted parquet files')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    parquet_files = [f for f in os.listdir(args.input_dir) if f.endswith('.parquet')]
    
    for pq_file in tqdm(parquet_files, desc="Processing files", total=len(parquet_files)):
        input_path = os.path.join(args.input_dir, pq_file)
        output_path = os.path.join(args.output_dir, pq_file)
        process_parquet_file(input_path, output_path)
    
    logger.info("Conversion completed")

if __name__ == "__main__":
    main() 