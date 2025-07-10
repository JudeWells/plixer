"""
Creates visualisations which show the predicted voxels and the true voxels side-by-side
also shows the predicted voxels inside the protein voxel grid (in this case show the protein voxels in )
"""
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import tarfile
import shutil
from tqdm import tqdm
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from src.evaluation.visual import (
    show_3d_voxel_lig_only,
    show_3d_voxel_lig_in_protein,
    ALL_ANGLES,
)


ALL_ANGLES = [
            (0, 0),
            (0, 45),
            (0, 90),
            (45, 135),
            (45, 180),
            (45, -135),
            (90, -90),
            (90, -45),
            (90, 0),
            (135, 45),
            (135, 90),
            (135, 135),
            (180, 180),
            (180, -135),
            (180, -90),
            (-135, -45),
            (-135, 0),
            (-135, 45),
            (-90, 90),
            (-90, 135),
            (-90, 180),
            (-45, -135),
            (-45, -90),
            (-45, -45),
            (-45, 0),
        ]

def process_single_case(pred_vox_path, true_vox_path, protein_vox_path, output_dir, row_info):
    """Process a single case of voxel visualization"""
    # Extract identifier from file path
    identifier = pred_vox_path.split('/')[-1].replace('poc2mol_output_', '').replace('.npy', '')
    
    # Extract additional information from row_info for naming & visuals
    poc2mol_loss_val = row_info.get("poc2mol_loss", "NA")
    tanimoto_val = row_info.get("tanimoto_similarity", "NA")

    # Robust formatting for folder name (handle missing / non-numeric values)
    def _fmt(val, digits):
        try:
            return f"{float(val):.{digits}f}"
        except Exception:
            return str(val)

    poc2mol_loss_str = _fmt(poc2mol_loss_val, 4)
    tanimoto_str = _fmt(tanimoto_val, 3)

    # Create output directory for this case – include metrics in name
    case_dir_name = f"{identifier}_loss_{poc2mol_loss_str}_tan_{tanimoto_str}"
    if len(glob.glob(f"{case_dir_name}*/true_smiles.png")) > 0:
        print(f"Skipping {case_dir_name} because it already exists")
        return
    tar_path = f"{output_dir}/{case_dir_name}.tar.gz"
    if os.path.exists(tar_path):
        print(f"Skipping {case_dir_name} because it already exists")
        return
    case_dir = os.path.join(output_dir, case_dir_name)
    os.makedirs(case_dir, exist_ok=True)
    
    # Load voxel data
    pred_vox = np.load(pred_vox_path).squeeze()
    true_vox = np.load(true_vox_path).squeeze()
    protein_vox = np.load(protein_vox_path).squeeze()
    
    # Generate visualizations (3D voxel renders)
    # 1. Predicted ligand voxels
    show_3d_voxel_lig_only(
        pred_vox,
        angles=ALL_ANGLES,
        save_dir=case_dir,
        identifier='predicted_ligand'
    )
    
    # 2. True ligand voxels
    show_3d_voxel_lig_only(
        true_vox,
        angles=ALL_ANGLES,
        save_dir=case_dir,
        identifier='true_ligand'
    )
    
    # 3. Predicted ligand inside protein pocket
    show_3d_voxel_lig_in_protein(
        pred_vox,
        protein_vox,
        angles=ALL_ANGLES,
        save_dir=case_dir,
        identifier='predicted_ligand_in_protein'
    )
    
    # 4. True ligand inside protein pocket
    show_3d_voxel_lig_in_protein(
        true_vox,
        protein_vox,
        angles=ALL_ANGLES,
        save_dir=case_dir,
        identifier='true_ligand_in_protein'
    )
    
    # ------------------------------------------------------------------
    # 5. Create 2-D depictions of true and sampled SMILES using RDKit
    true_smiles = row_info.get("smiles")
    sampled_smiles = row_info.get("sampled_smiles")

    def _save_mol_image(smiles: str, out_path: str):
        if not smiles:
            return
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"[WARN] Could not parse SMILES: {smiles}")
            return
        Draw.MolToFile(mol, out_path)

    _save_mol_image(true_smiles, os.path.join(case_dir, "true_smiles.png"))
    _save_mol_image(sampled_smiles, os.path.join(case_dir, "sampled_smiles.png"))

    # ------------------------------------------------------------------
    # Create tar.gz archive of the case directory
    
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(case_dir, arcname=os.path.basename(case_dir))
    
    # Verify the tar.gz file was created successfully
    if os.path.exists(tar_path) and os.path.getsize(tar_path) > 0:
        # Remove the original directory
        shutil.rmtree(case_dir)
    else:
        print(f"Warning: Failed to create tar.gz for {identifier}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_idx", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument(
        "--dark_mode",
        action="store_true",
        help="Generate voxel visualisation images with a black background (dark mode)",
    )
    args = parser.parse_args()
    # Apply dark mode styling globally if requested
    if args.dark_mode:
        # The 'dark_background' style sets figure and axes facecolors to black and
        # switches default text/edge colours to lighter tones for contrast.
        plt.style.use("dark_background")
    # Define paths
    base_dir = "evaluation_results/CombinedHiQBindCkptFrmPrevCombined_2025-05-06_v3_member_zero_v3"
    output_dir = os.path.join(base_dir, "voxel_visualizations_all_angles_cluster_dark_mode")
    csv_path = glob.glob(os.path.join(base_dir, "*.csv"))[0]
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all voxel file paths
    pred_vox_paths = sorted(glob.glob(f"{base_dir}/voxels/poc2mol_output*.npy"))
    batch_size = (len(pred_vox_paths) // args.num_tasks) + 1
    this_batch = pred_vox_paths[args.task_idx * batch_size:(args.task_idx + 1) * batch_size]
    print(f"Processing batch {args.task_idx + 1} of {args.num_tasks} with {len(this_batch)} cases of {len(pred_vox_paths)} total")
    # Verify all paths match
    for pred_path in tqdm(this_batch):
        name = os.path.basename(pred_path).replace("poc2mol_output_", "").replace(".npy", "")[:-5]
        match = df[df["name"] == name]
        if len(match) == 0:
            print(f"No match found for {name}")
            continue
        row_info = match.iloc[0].to_dict()
        true_path = pred_path.replace("poc2mol_output", "true_ligand_voxels")
        protein_path = pred_path.replace("poc2mol_output", "protein_voxels")
        if not os.path.exists(true_path):
            print(f"True path does not exist: {true_path}")
            continue
        if not os.path.exists(protein_path):
            print(f"Protein path does not exist: {protein_path}")
            continue
        try:
            process_single_case(pred_path, true_path, protein_path, output_dir, row_info)
        except Exception as e:
            print(f"Error processing case: {e}")
    
    print(f"Visualizations completed and saved to {output_dir}")