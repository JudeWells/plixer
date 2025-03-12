import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Function to generate RDKit molecule from SMILES
def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        return mol
    except:
        return None

# Function to visualize molecules
def visualize_molecules(mols, atom_size, output_path):
    mols_per_row = 10
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(500, 500),
        legends=[f"Mol {i+1}" for i in range(len(mols))],
        useSVG=False
    )
    img.save(output_path)
    print(f"Saved visualization for atom size {atom_size} to {output_path}")

# Main function
def main(input_csv, output_dir, atom_sizes, molecules_per_size=50):
    df = pd.read_csv(input_csv)
    df = df.sort_values(by='num_atoms')

    # Define atom size bins
    bins = defaultdict(list)
    for _, row in df.iterrows():
        num_atoms = row['num_atoms']
        smiles = row['smiles']
        bins[num_atoms].append(smiles)

    # Process each atom size
    for atom_size in atom_sizes:
        selected_smiles = []
        if atom_size == '200+':
            candidates = df[df['num_atoms'] >= 200]['smiles'].tolist()
        else:
            candidates = df[df['num_atoms'] == atom_size]['smiles'].tolist()

        unique_smiles = list(dict.fromkeys(candidates))  # Remove duplicates while preserving order
        mols = []
        idx = 0

        while len(mols) < molecules_per_size and idx < len(unique_smiles):
            mol = smiles_to_mol(unique_smiles[idx])
            if mol:
                mols.append(mol)
            idx += 1

        if len(mols) < molecules_per_size:
            print(f"Warning: Only {len(mols)} valid molecules found for atom size {atom_size}")

        if mols:
            output_path = f"{output_dir}/ligands_{atom_size}_atoms.png"
            visualize_molecules(mols, atom_size, output_path)

if __name__ == "__main__":
    input_csv = "large_ligands.csv"
    output_dir = "./ligand_images"
    atom_sizes = [30, 40, 50, 60, 80, 90, 100, 110, 120, 130, 150, '200+']

    import os
    os.makedirs(output_dir, exist_ok=True)

    main(input_csv, output_dir, atom_sizes)
