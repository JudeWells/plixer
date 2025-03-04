import os
import gzip
import glob
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

def mol2_to_mol(mol2_text):
    """Convert a mol2 text representation to an RDKit molecule."""
    mol = Chem.MolFromMol2Block(mol2_text)
    if mol is None:
        return None
    try:
        # Get a canonical SMILES string
        smiles = Chem.MolToSmiles(mol)
        # Ensure the molecule is valid by attempting to reconstruct it
        test_mol = Chem.MolFromSmiles(smiles)
        if test_mol is None:
            return None
        return mol
    except:
        return None

def process_mol2_file(file_path, max_mols_per_batch=1000):
    """Process a gzipped mol2 file and extract molecules."""
    try:
        with gzip.open(file_path, 'rt') as f:
            contents = f.read()
        
        # Split by @<TRIPOS>MOLECULE to get individual molecules
        mol_blocks = contents.split('@<TRIPOS>MOLECULE')
        mol_blocks = [block for block in mol_blocks if block.strip()]
        
        molecules = []
        for block in mol_blocks:
            # Reconstruct the mol2 block
            mol2_block = '@<TRIPOS>MOLECULE' + block
            mol = mol2_to_mol(mol2_block)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                molecules.append({
                    'smiles': smiles,
                    'mol': mol,
                    'source_file': str(file_path)
                })
            
            # If we've collected enough molecules, yield the batch
            if len(molecules) >= max_mols_per_batch:
                yield molecules
                molecules = []
        
        # Yield any remaining molecules
        if molecules:
            yield molecules
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def serialize_molecule(mol):
    """Serialize a molecule to a binary format."""
    return Chem.MolToMolBlock(mol).encode()

def save_molecules_to_parquet(molecules, output_dir, batch_idx):
    """Save a list of molecules to a parquet file."""
    if not molecules:
        return
    
    data = {
        'smiles': [],
        'mol_block': [],
        'source_file': []
    }
    
    for mol_data in molecules:
        data['smiles'].append(mol_data['smiles'])
        data['mol_block'].append(serialize_molecule(mol_data['mol']))
        data['source_file'].append(mol_data['source_file'])
    
    df = pd.DataFrame(data)
    output_file = os.path.join(output_dir, f'batch_{batch_idx:06d}.parquet')
    df.to_parquet(output_file, index=False)
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Process ZINC20 mol2 files into parquet format')
    parser.add_argument('--input_dir', type=str, 
                        default="/mnt/disk2/zinc20", 
                        help='Directory containing ZINC20 mol2.gz files')
    parser.add_argument('--output_dir', type=str, 
                        default="/mnt/disk2/zinc20_parquet", 
                        help='Directory to save parquet files')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of mol2.gz files to process')
    parser.add_argument('--max_mols_per_batch', type=int, default=10000, 
                        help='Maximum number of molecules per parquet file')
    parser.add_argument('--max_batches', type=int, default=None, 
                        help='Maximum number of parquet files to create')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all mol2.gz files
    mol2_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.mol2.gz'):
                mol2_files.append(os.path.join(root, file))
    
    if args.max_files:
        mol2_files = mol2_files[:args.max_files]
    
    print(f"Found {len(mol2_files)} mol2.gz files")
    
    # Process each file
    batch_idx = 0
    saved_files = []
    
    for file_path in tqdm(mol2_files):
        for molecule_batch in process_mol2_file(file_path, args.max_mols_per_batch):
            output_file = save_molecules_to_parquet(molecule_batch, args.output_dir, batch_idx)
            saved_files.append(output_file)
            batch_idx += 1
            
            if args.max_batches and batch_idx >= args.max_batches:
                print(f"Reached maximum number of batches ({args.max_batches})")
                break
        
        if args.max_batches and batch_idx >= args.max_batches:
            break
    
    print(f"Saved {batch_idx} parquet files with up to {args.max_mols_per_batch} molecules each")
    
    # Create an index file
    index_df = pd.DataFrame({'parquet_file': saved_files})
    index_df.to_csv(os.path.join(args.output_dir, 'index.csv'), index=False)
    
    print(f"Created index file with {len(saved_files)} parquet files")

if __name__ == "__main__":
    main()
