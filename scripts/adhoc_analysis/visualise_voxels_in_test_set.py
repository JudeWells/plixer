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

from src.evaluation.visual import show_3d_voxel_lig_only, show_3d_voxel_lig_in_protein, ALL_ANGLES


def process_single_case(pred_vox_path, true_vox_path, protein_vox_path, output_dir):
    """Process a single case of voxel visualization"""
    # Extract identifier from file path
    identifier = pred_vox_path.split('/')[-1].replace('poc2mol_output_', '').replace('.npy', '')
    
    # Create output directory for this case
    case_dir = os.path.join(output_dir, identifier)
    os.makedirs(case_dir, exist_ok=True)
    
    # Load voxel data
    pred_vox = np.load(pred_vox_path).squeeze()
    true_vox = np.load(true_vox_path).squeeze()
    protein_vox = np.load(protein_vox_path).squeeze()
    
    # Generate visualizations
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
    
    # Create tar.gz archive of the case directory
    tar_path = f"{case_dir}.tar.gz"
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
    args = parser.parse_args()
    # Define paths
    base_dir = "evaluation_results/CombinedHiQBindCkptFrmPrevCombined_2025-05-06_v3_member_zero_v3"
    output_dir = os.path.join(base_dir, "voxel_visualizations_all_angles_cluster")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all voxel file paths
    pred_vox_paths = sorted(glob.glob(f"{base_dir}/voxels/poc2mol_output*.npy"))
    batch_size = (len(pred_vox_paths) // args.num_tasks) + 1
    this_batch = pred_vox_paths[args.task_idx * batch_size:(args.task_idx + 1) * batch_size]
    print(f"Processing batch {args.task_idx + 1} of {args.num_tasks} with {len(this_batch)} cases of {len(pred_vox_paths)} total")
    # Verify all paths match
    for pred_path in tqdm(this_batch):
        true_path = pred_path.replace("poc2mol_output", "true_ligand_voxels")
        protein_path = pred_path.replace("poc2mol_output", "protein_voxels")
        if not os.path.exists(true_path):
            print(f"True path does not exist: {true_path}")
            continue
        if not os.path.exists(protein_path):
            print(f"Protein path does not exist: {protein_path}")
            continue
        try:
            process_single_case(pred_path, true_path, protein_path, output_dir)
        except Exception as e:
            print(f"Error processing case: {e}")
    
    print(f"Visualizations completed and saved to {output_dir}")