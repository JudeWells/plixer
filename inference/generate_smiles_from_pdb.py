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
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.poc2smiles import CombinedProteinToSmilesModel
from src.data.common.voxelization.config import Poc2MolDataConfig
from src.data.common.voxelization.molecule_utils import voxelize_complex
from src.utils.utils import voxelize_protein
from src.utils.metrics import calculate_validity, calculate_uniqueness
from src.utils.utils import load_model, get_center_from_ligand
from src.evaluation.visual import visualize_2d_smiles_batch

def parse_args():
    parser = argparse.ArgumentParser(description="Generate SMILES strings from protein PDB files")
    
    # Paths
    parser.add_argument(
        "--vox2smiles_ckpt_path", 
        type=str, 
        default="checkpoints/combined_protein_to_smiles/epoch_000.ckpt", 
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--poc2mol_ckpt_path", 
        type=str, 
        default="checkpoints/poc_vox_to_mol_vox/epoch_173.ckpt",
        help="Path to model checkpoint"
    )
    parser.add_argument("--pdb_file", 
        type=str, 
        default="data/agonists/5-MeO-DMT_8fy8.pdb", 
        help="Path to the protein PDB file."
    )
    parser.add_argument("--ligand_file", 
        type=str,
        default="data/agonists/LSD_8fyt_E_7LD.mol2",
        help="Path to a ligand file (SDF/MOL2) only used to define the pocket center."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs/generate_smiles_8fy8", 
        help="Path to output directory"
    )
    
    # Generation parameters
    parser.add_argument("--center", type=float, nargs=3, help="Center coordinates (x y z) for the pocket can be used instead of --ligand_file to define the pocket center.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate per protein")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dtype", type=str, default="torch.float32", help="Data type for the model.")
    args = parser.parse_args()
    if isinstance(args.dtype, str):
        args.dtype = eval(args.dtype)
    return args


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    if not args.ligand_file and not args.center:
        raise ValueError("Either --ligand_file or --center must be provided.")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.vox2smiles_ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(
        args.vox2smiles_ckpt_path, 
        args.poc2mol_ckpt_path, 
        device, 
        dtype=args.dtype
    )
    model.eval()
    
    # Get protein PDB files
    protein_files = [args.pdb_file]
    print(f"Found {len(protein_files)} protein PDB files")
    
    # Create data configs
    complex_dataset_config = config.data.train_dataset.poc2mol_output_dataset.complex_dataset.config
    complex_dataset_config = {k:v for k,v in complex_dataset_config.items() if k != '_target_'}
    poc2mol_config = Poc2MolDataConfig(
        **complex_dataset_config
    )
    poc2mol_config.random_rotation = False
    poc2mol_config.random_translation = 0.0
    poc2mol_config.dtype = args.dtype
    
    # Process each protein
    all_generated_smiles = []
    
    print(f"Processing {args.pdb_file}")
    
    # Get protein name
    protein_name = os.path.basename(args.pdb_file)
    
    ligand_voxel = None
    if args.center:
        if args.ligand_file:
            print(f"--ligand_file provided with --center, ignoring ligand file and using --center")
        center = np.array(args.center)
        print(f"Using center: {center}")
        # Voxelize protein

    elif args.ligand_file:
        if not os.path.exists(args.ligand_file):
            print(f"Ligand file not found for {protein_name}, skipping")
            return
        center = get_center_from_ligand(args.ligand_file)
        # Voxelize protein-ligand complex
    else:
        raise ValueError("Either --ligand_file or --center must be provided to define the pocket center.")
    protein_voxel = voxelize_protein(args.pdb_file, center, poc2mol_config)

    
    generated_smiles_list = []
    
    with torch.no_grad():
        # Generate multiple samples
        output = model(protein_voxel, sample_smiles=False)
        for _ in range(args.num_samples):
            generated_smiles = model.vox2smiles_model.generate_smiles(
                output['predicted_ligand_voxels'], 
                do_sample=True,
                temperature=args.temperature,
                max_attempts=10
            )
            generated_smiles_list.extend(generated_smiles)
    
    all_generated_smiles.extend(generated_smiles_list)
    unique_generated_smiles = list(set(generated_smiles_list))
    print(f"Out of {len(generated_smiles_list)} generated SMILES, {len(unique_generated_smiles)} are unique")
    for smiles in generated_smiles_list:
        print(smiles)
    # Save generated SMILES
    output_file = os.path.join(args.output_dir, f"{protein_name}_smiles.txt")
    with open(output_file, "w") as f:
        for smiles in generated_smiles_list:
            f.write(f"{smiles}\n")
    
    # Visualize results
    visualize_2d_smiles_batch(
        generated_smiles_list,
        os.path.join(args.output_dir, f"{protein_name}_generated_smiles.png"),
        n_cols=3,
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



if __name__ == "__main__":
    main() 