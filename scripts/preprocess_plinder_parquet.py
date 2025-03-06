"""
Preprocess Plinder parquet files by converting raw PDB and MOL data into MolecularData objects.
This avoids the need to create temporary files during training.
"""

import os
import pandas as pd
import numpy as np
import torch
import tempfile
import argparse
import logging
import pickle
import base64
from tqdm import tqdm
from biopandas import pdb, mol2
from docktgrid.molparser import MolecularData
from src.data.docktgrid_mods import MolecularParserWrapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def serialize_molecular_data(data):
    """Serialize MolecularData object to a base64 encoded string."""
    pickled = pickle.dumps(data)
    return base64.b64encode(pickled)

def process_parquet_file(parquet_file, output_file=None):
    """
    Process a parquet file by converting raw PDB and MOL data into MolecularData objects.
    
    Args:
        parquet_file: Path to the input parquet file
        output_file: Path to the output parquet file. If None, will overwrite the input file.
    
    Returns:
        Path to the processed parquet file
    """
    if output_file is None:
        output_file = parquet_file
    
    log.info(f"Processing {parquet_file}")
    df = pd.read_parquet(parquet_file)
    
    # Create a parser
    parser = MolecularParserWrapper()
    
    # Create temporary directory for files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process each row
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            try:
                # Write protein to PDB file
                protein_path = os.path.join(temp_dir, f"protein_{idx}.pdb")
                with open(protein_path, 'wb') as f:
                    f.write(row['pdb_content'])
                
                # Write ligand to MOL file
                ligand_path = os.path.join(temp_dir, f"ligand_{idx}.mol")
                with open(ligand_path, 'wb') as f:
                    f.write(row['mol_block'])
                
                # Parse protein
                protein_data = parser.parse_file(protein_path, '.pdb')
                
                # Parse ligand (MOL files need to be converted to PDB for biopandas)
                # We'll use RDKit to convert MOL to PDB
                from rdkit import Chem
                mol = Chem.MolFromMolBlock(row['mol_block'].decode())
                if mol is None:
                    log.warning(f"Failed to parse ligand for row {idx}, skipping")
                    continue
                
                # Write as PDB
                ligand_pdb_path = os.path.join(temp_dir, f"ligand_{idx}.pdb")
                Chem.MolToPDBFile(mol, ligand_pdb_path)
                
                # Parse ligand
                ligand_data = parser.parse_file(ligand_pdb_path, '.pdb')
                
                # Serialize the data
                protein_serialized = serialize_molecular_data(protein_data)
                ligand_serialized = serialize_molecular_data(ligand_data)
                
                # Store the serialized data
                df.at[idx, 'protein_data_serialized'] = protein_serialized
                df.at[idx, 'ligand_data_serialized'] = ligand_serialized
                
            except Exception as e:
                log.error(f"Error processing row {idx}: {e}")
    
    # Save the processed dataframe
    log.info(f"Saving processed dataframe to {output_file}")
    df.to_parquet(output_file)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Preprocess Plinder parquet files')
    parser.add_argument('--input_dir', type=str, default="/mnt/disk2/plinder/2024-06/v2/parquet", 
                        help='Directory containing Plinder parquet files')
    parser.add_argument('--output_dir', type=str, default="/mnt/disk2/plinder/2024-06/v2/preprocessed_parquet",
                        help='Directory to save processed parquet files. If None, will overwrite input files.')
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all parquet files
    parquet_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) 
                    if f.endswith('.parquet') and not f == 'index.parquet']
    
    # Process each file
    processed_files = []
    for parquet_file in parquet_files:
        if args.output_dir:
            output_file = os.path.join(args.output_dir, os.path.basename(parquet_file))
        else:
            output_file = None
        
        processed_file = process_parquet_file(parquet_file, output_file)
        processed_files.append(processed_file)
    
    # Create an index file if output_dir is specified
    if args.output_dir:
        index_df = pd.DataFrame({
            'parquet_file': processed_files,
            'split': [f.split('/')[-1].split('_')[0] for f in processed_files]
        })
        index_df.to_csv(os.path.join(args.output_dir, 'index.csv'), index=False)
    
    log.info(f"Processed {len(processed_files)} parquet files")

if __name__ == "__main__":
    main()