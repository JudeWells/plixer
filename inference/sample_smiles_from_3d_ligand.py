"""
Inference script that takes a path to an SDF or MOL2 file containing a 3D
ligand structure, voxelizes the ligand, and samples SMILES from the
vox2smiles model.

Use this script if you want to sample analogs of a ligand for which you have 
3D structure cocrystalized with your target protein.

The temperature parameter may require some tuning: if all generated SMILES are
too similar: increase the temperature. If you get many invalid SMILES warnings: 
decrease the temperature.

This script mirrors the structure of `generate_smiles_from_pdb.py` but does
not load or use the Poc2Mol model. Instead, it directly voxelizes the ligand
and feeds it to the Vox2Smiles model for SMILES generation.
"""

import os
import argparse
import torch
import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from rdkit import Chem

from src.models.vox2smiles import VoxToSmilesModel
from src.data.common.voxelization.config import Vox2SmilesDataConfig
from src.data.common.voxelization.molecule_utils import voxelize_molecule
from src.utils.metrics import calculate_validity, calculate_uniqueness
from src.evaluation.visual import visualize_2d_smiles_batch
from src.utils.utils import get_config_from_cpt_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SMILES strings from a 3D ligand (SDF/MOL2) using Vox2Smiles"
    )

    parser.add_argument(
        "--ligand_file",
        type=str,
        default="data/crossdocked_test/ABL2_HUMAN_274_551_0/4xli_B_rec_4xli_1n1_lig_tt_min_0.sdf",
        help="Path to a ligand file (.sdf or .mol2) containing 3D coordinates.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/sample_smiles_from_ligand",
        help="Path to output directory",
    )

    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=1.5, help="Temperature for sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dtype", type=str, default="torch.float32", help="Data type for the model")
    parser.add_argument(
        "--vox2smiles_ckpt_path", 
        type=str, 
        default="checkpoints/combined_protein_to_smiles/epoch_000.ckpt", 
        help="Path to model checkpoint"
    )

    # Voxelization overrides (optional)
    parser.add_argument(
        "--random_rotation", action="store_true", help="Apply random rotation during voxelization"
    )
    parser.add_argument(
        "--random_translation", type=float, default=0.0, help="Max random translation during voxelization"
    )

    args = parser.parse_args()
    if isinstance(args.dtype, str):
        args.dtype = eval(args.dtype)
    return args


def load_ligand_from_file(path: str):
    if path.endswith(".sdf"):
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        mol = next(suppl)
    elif path.endswith(".mol2"):
        mol = Chem.MolFromMol2File(path, removeHs=False)
    else:
        raise ValueError("Ligand file must be .sdf or .mol2")
    if mol is None:
        raise ValueError(f"Could not read ligand file: {path}")
    return mol


def main():
    args = parse_args()

    # Set random seed and device
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model config and vox2smiles model
    print(f"Loading Vox2Smiles model from {args.vox2smiles_ckpt_path}")
    config = get_config_from_cpt_path(args.vox2smiles_ckpt_path)
    vox2smiles_model = VoxToSmilesModel.load_from_checkpoint(args.vox2smiles_ckpt_path)
    vox2smiles_model = vox2smiles_model.to(device=device, dtype=args.dtype)
    vox2smiles_model.eval()

    # Build a voxelization config mirroring dataset usage
    voxel_config = Vox2SmilesDataConfig()
    voxel_config.random_rotation = args.random_rotation
    voxel_config.random_translation = args.random_translation
    voxel_config.dtype = args.dtype
    voxel_config.batch_size = 1

    # Read ligand and optionally strip hydrogens
    ligand_mol = load_ligand_from_file(args.ligand_file)


    # Voxelize ligand
    ligand_voxel = voxelize_molecule(ligand_mol, voxel_config)
    if len(ligand_voxel.shape) == 4:
        # add a batch dimension
        ligand_voxel = ligand_voxel.unsqueeze(0)

    ligand_voxel = ligand_voxel.to(device)

    # Generate multiple samples
    generated_smiles_list = []
    with torch.no_grad():
        for _ in range(args.num_samples):
            smiles_batch = vox2smiles_model.generate_smiles(
                ligand_voxel, do_sample=True, temperature=args.temperature, max_attempts=10
            )
            generated_smiles_list.extend(smiles_batch)

    # Logging and outputs
    ligand_name = os.path.basename(args.ligand_file)
    unique_generated_smiles = list(set(generated_smiles_list))
    print(f"Out of {len(generated_smiles_list)} generated SMILES, {len(unique_generated_smiles)} are unique")
    for sm in generated_smiles_list:
        print(sm)

    # Save generated SMILES
    output_file = os.path.join(args.output_dir, f"{ligand_name}_smiles.txt")
    with open(output_file, "w") as f:
        for sm in generated_smiles_list:
            f.write(f"{sm}\n")

    # Visualize results
    visualize_2d_smiles_batch(
        generated_smiles_list,
        os.path.join(args.output_dir, f"{ligand_name}_generated_smiles.png"),
        n_cols=3,
    )

    # Calculate metrics
    validity = calculate_validity(generated_smiles_list)
    uniqueness = calculate_uniqueness(generated_smiles_list)

    print(f"Generated {len(generated_smiles_list)} SMILES strings")
    print(f"Validity: {validity:.4f}")
    print(f"Uniqueness: {uniqueness:.4f}")

    # Save metrics
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write(f"Generated SMILES: {len(generated_smiles_list)}\n")
        f.write(f"Validity: {validity:.4f}\n")
        f.write(f"Uniqueness: {uniqueness:.4f}\n")


if __name__ == "__main__":
    main()