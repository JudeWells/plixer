import os
import glob
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
from tqdm import tqdm

def process_mol2_file(mol2_path):
    """Process a single mol2 file and return a dictionary with molecule data."""
    try:
        # Read mol2 file using RDKit
        mol = Chem.MolFromMol2File(mol2_path)
        if mol is None:
            print(f"Failed to read {mol2_path}")
            return None
        
        # Generate 3D coordinates if not present
        if mol.GetNumConformers() == 0:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            mol = Chem.RemoveHs(mol)
        
        # Get molecule ID from directory name
        mol_id = os.path.basename(os.path.dirname(mol2_path))
        
        # Extract atom positions
        conf = mol.GetConformer()
        positions = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        
        # Extract atom types
        atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
        
        # Create SMILES
        smiles = Chem.MolToSmiles(mol)
        
        return {
            'mol_id': mol_id,
            'smiles': smiles,
            'positions': positions,
            'atom_types': atom_types,
            'mol': mol  # Include the RDKit mol object for additional processing if needed
        }
    except Exception as e:
        print(f"Error processing {mol2_path}: {e}")
        return None

def save_to_parquet(molecules, output_file):
    """Save a list of molecule dictionaries to a parquet file."""
    # Convert to DataFrame
    data = []
    for mol_dict in molecules:
        if mol_dict is not None:
            # Convert positions and atom_types to serializable format
            positions_str = np.array2string(mol_dict['positions'], precision=6)
            atom_types_str = ','.join(mol_dict['atom_types'])
            
            data.append({
                'mol_id': mol_dict['mol_id'],
                'smiles': mol_dict['smiles'],
                'positions': positions_str,
                'atom_types': atom_types_str
            })
    
    # Create DataFrame and save to parquet
    df = pd.DataFrame(data)
    df.to_parquet(output_file, index=False)
    print(f"Saved {len(data)} molecules to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert PDBbind mol2 files to parquet format')
    parser.add_argument('--input_dir', type=str, 
                        default="/mnt/disk2/VoxelDiffOuter/PDBbind_v2020_refined-set", 
                        help='Input directory containing PDBbind data')
    parser.add_argument('--output_dir', type=str, default="/mnt/disk2/VoxelDiffOuter/PDBbind_v2020_refined-set/parquet", help='Output directory for parquet files')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of molecules per parquet file')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all mol2 files
    mol2_files = glob.glob(os.path.join(args.input_dir, '*/*_ligand.mol2'))
    print(f"Found {len(mol2_files)} mol2 files")
    
    # Process files in batches
    molecules = []
    file_count = 0
    
    for i, mol2_path in enumerate(tqdm(mol2_files)):
        mol_dict = process_mol2_file(mol2_path)
        if mol_dict is not None:
            molecules.append(mol_dict)
        
        # Save batch when it reaches the specified size
        if len(molecules) >= args.batch_size or i == len(mol2_files) - 1:
            if molecules:
                output_file = os.path.join(args.output_dir, f'pdbind_batch_{file_count}.parquet')
                save_to_parquet(molecules, output_file)
                file_count += 1
                molecules = []

    print(f"Completed processing. Created {file_count} parquet files.")

if __name__ == "__main__":
    main()