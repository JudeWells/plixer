#!/usr/bin/env python
"""
Script to generate SMILES strings from protein PDB files using the end-to-end model.
"""

import os
import argparse
import torch
import glob
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import matplotlib.pyplot as plt
import numpy as np

from src.models.poc2smiles import CombinedProteinToSmilesModel
from src.data.common.voxelization.config import Poc2MolDataConfig, Vox2SmilesDataConfig
from src.data.common.voxelization.molecule_utils import prepare_protein_ligand_complex, voxelize_complex
from src.utils.metrics import calculate_validity, calculate_uniqueness


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SMILES strings from protein PDB files")
    
    # Paths
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--protein_dir", type=str, required=True, help="Path to directory containing protein PDB files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    
    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate per protein")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = CombinedProteinToSmilesModel.load_from_checkpoint(args.checkpoint)
    model.eval()
    
    # Get protein PDB files
    protein_files = glob.glob(os.path.join(args.protein_dir, "**/*_protein.pdb"), recursive=True)
    print(f"Found {len(protein_files)} protein PDB files")
    
    # Create data configs
    poc2mol_config = Poc2MolDataConfig(
        vox_size=0.75,
        box_dims=[24.0, 24.0, 24.0],
        random_rotation=False,
        random_translation=0.0,
        max_atom_dist=32.0,
        batch_size=args.batch_size,
    )
    
    # Process each protein
    all_generated_smiles = []
    
    for protein_file in protein_files:
        print(f"Processing {protein_file}")
        
        # Get protein name
        protein_name = os.path.basename(os.path.dirname(protein_file))
        
        # Find ligand file
        ligand_file = os.path.join(os.path.dirname(protein_file), f"{protein_name}_ligand.mol2")
        if not os.path.exists(ligand_file):
            print(f"Ligand file not found for {protein_name}, skipping")
            continue
        
        # Voxelize protein-ligand complex
        try:
            protein_voxel, ligand_voxel, _ = voxelize_complex(protein_file, ligand_file, poc2mol_config)
        except Exception as e:
            print(f"Error voxelizing {protein_name}: {e}")
            continue
        
        # Generate SMILES strings
        protein_voxel = protein_voxel.unsqueeze(0)  # Add batch dimension
        
        generated_smiles_list = []
        
        with torch.no_grad():
            # Generate multiple samples
            for _ in range(args.num_samples):
                output = model(protein_voxel)
                ligand_voxels = output["ligand_voxels"]
                generated_smiles = model.vox2smiles_model.generate_smiles(
                    ligand_voxels, 
                    temperature=args.temperature
                )
                generated_smiles_list.extend(generated_smiles)
        
        all_generated_smiles.extend(generated_smiles_list)
        
        # Save generated SMILES
        output_file = os.path.join(args.output_dir, f"{protein_name}_smiles.txt")
        with open(output_file, "w") as f:
            for smiles in generated_smiles_list:
                f.write(f"{smiles}\n")
        
        # Visualize results
        visualize_results(
            protein_name,
            protein_voxel[0].cpu().numpy(),
            ligand_voxel.cpu().numpy(),
            output["ligand_voxels"][0].cpu().numpy(),
            generated_smiles_list,
            os.path.join(args.output_dir, f"{protein_name}_visualization.png")
        )
    
    # Calculate metrics
    validity = calculate_validity(all_generated_smiles)
    uniqueness = calculate_uniqueness(all_generated_smiles)
    
    print(f"Generated {len(all_generated_smiles)} SMILES strings")
    print(f"Validity: {validity:.4f}")
    print(f"Uniqueness: {uniqueness:.4f}")
    
    # Save metrics
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write(f"Generated SMILES: {len(all_generated_smiles)}\n")
        f.write(f"Validity: {validity:.4f}\n")
        f.write(f"Uniqueness: {uniqueness:.4f}\n")


def visualize_results(protein_name, protein_voxel, true_ligand_voxel, generated_ligand_voxel, generated_smiles, output_file):
    """
    Visualize the results of the generation.
    
    Args:
        protein_name: Name of the protein
        protein_voxel: Protein voxel
        true_ligand_voxel: True ligand voxel
        generated_ligand_voxel: Generated ligand voxel
        generated_smiles: List of generated SMILES strings
        output_file: Path to output file
    """
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Add title
    fig.suptitle(f"Results for {protein_name}", fontsize=16)
    
    # Plot protein voxel
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(np.max(protein_voxel[0], axis=0), cmap="viridis")
    ax1.set_title("Protein Voxel")
    ax1.axis("off")
    
    # Plot true ligand voxel
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(np.max(true_ligand_voxel[0], axis=0), cmap="viridis")
    ax2.set_title("True Ligand Voxel")
    ax2.axis("off")
    
    # Plot generated ligand voxel
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(np.max(generated_ligand_voxel[0], axis=0), cmap="viridis")
    ax3.set_title("Generated Ligand Voxel")
    ax3.axis("off")
    
    # Plot generated molecules
    valid_mols = []
    for smiles in generated_smiles[:3]:  # Show up to 3 molecules
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            AllChem.Compute2DCoords(mol)
            valid_mols.append((mol, smiles))
    
    for i, (mol, smiles) in enumerate(valid_mols):
        if i >= 3:
            break
        
        ax = fig.add_subplot(2, 3, 4 + i)
        img = Draw.MolToImage(mol, size=(300, 300))
        ax.imshow(img)
        ax.set_title(f"Generated Molecule {i+1}")
        ax.axis("off")
        
        # Add SMILES as text
        ax.text(0.5, -0.1, smiles, transform=ax.transAxes, ha="center", va="center", fontsize=8, wrap=True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main() 