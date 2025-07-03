import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import wandb
from rdkit import Chem
from rdkit.Chem import Draw
import os

ALL_ANGLES = [
            (0, 0),
            (0, 45),
            (0, 90),
            (0, 135),
            (0, 180),
            (0, -45),
            (0, -90),
            (0, -135), 
            (45, 0),
            (45, 45),
            (45, 90),
            (45, 135),
            (45, 180),
            (45, -45),
            (45, -90),
            (45, -135),
            # (90, 0),
            (90, 45),
            # (90, 90),
            # (90, 135),
            # (90, 180),
            # (90, -45),
            # (90, -90),
            # (90, -135),
            (135, 0),
            (135, 45),
            (135, 90),
            (135, 135),
            (135, 180),
            (135, -45),
            (135, -90),
            (135, -135),
            (180, 0),
            (180, 45),
            (180, 90),
            (180, 135),
            (180, 180),
            (180, -45),
            (180, -90),
            (180, -135),
            (-135, 0),
            (-135, 45),
            (-135, 90),
            (-135, 135),
            (-135, 180),
            (-135, -45),
            (-135, -90),
            (-135, -135),    
            (-45, 0),
            (-45, 45),
            (-45, 90),
            (-45, 135),
            (-45, 180),
            (-45, -45),
            (-45, -90),
            (-45, -135),
            (-90, 45)
            # (-90, 0),# these just rotate in perpendicular to plane of view
            # (-90, 90),
            # (-90, 135),
            # (-90, 180),
            # (-90, -45),
            # (-90, -90),
            # (-90, -135),
 
        ]

def show_3d_voxel_lig_only(vox, angles=None, save_dir=None, identifier='lig'):
    if not isinstance(vox, np.ndarray):
        vox = vox.detach().float().cpu().numpy()
    vox = (vox > 0.5).astype(int)
    # vox = vox[:5]  # only ligand channels
    colors = np.zeros((vox.shape[0], 4))
    colors[0] = mcolors.to_rgba('green')  # carbon_ligand
    # colors[1] = mcolors.to_rgba('white')  # hydrogen_ligand
    colors[1] = mcolors.to_rgba('red')  # oxygen_ligand
    colors[2] = mcolors.to_rgba('blue')  # nitrogen_ligand
    colors[3] = mcolors.to_rgba('yellow')  # sulfur_ligand
    colors[4] = mcolors.to_rgba('cyan')  # chlorine_ligand
    colors[5] = mcolors.to_rgba('darkcyan')  # fluorine_ligand
    colors[6] = mcolors.to_rgba('magenta')  # iodine_ligand
    colors[7] = mcolors.to_rgba('brown')  # bromine_ligand
    colors[8] = mcolors.to_rgba('purple')  # other_ligand

    # Set transparency for all colors
    colors[:, 3] = 0.2

    # Define the default viewing angles if not provided
    if angles is None:
        angles = [(45, 45), (15, 90), (0, 30), (45, -45)]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Iterate over each channel and plot the voxels with corresponding colors
    for channel in range(vox.shape[0]):
        ax.voxels(vox[channel], facecolors=colors[channel], edgecolors=colors[channel])
    # Iterate over each angle and save the image
    for i, angle in enumerate(angles):
        ax.view_init(elev=angle[0], azim=angle[1])
        ax.axis('off')
        ax.grid(False)  # Turn off the grid
        ax.set_xticks([])  # Turn off x ticks
        ax.set_yticks([])  # Turn off y ticks
        ax.set_zticks([])  #
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/{identifier}_angle_{i + 1}.png")
    plt.close()
    plt.close(fig)


def visualise_batch(lig, pred, names, angles=None, save_dir=None, batch='none', reuse_labels=True, log_wandb=True, limit_channels=None):
    """
    Visualise the label and the predictions for a batch of ligands
    show the predictions side by side in a single plot
    show 2 angles for each ligand (label and pred) in one row of the plot
    """
    color_names = ['green', 'red', 'blue', 'yellow', 'magenta', 'magenta', 'magenta', 'magenta', 'cyan']
    if not isinstance(lig, np.ndarray):
        lig = lig.detach().cpu().float().numpy()
    lig = (lig > 0.5).astype(int)

    if not isinstance(pred, np.ndarray):
        pred = pred.detach().cpu().float().numpy()
    pred = (pred > 0.5).astype(int)

    if angles is None:
        angles = [(45, 45)] # angles = [(45, 45), (45, -45)]
    if reuse_labels:
        label_save_dir = os.path.join(save_dir,  '..', '..', '..', 'label_images')
    else:
        label_save_dir = os.path.join(save_dir, f'batch_{batch}', 'label_images')
    pred_save_dir = os.path.join(save_dir, f'batch_{batch}', 'predictions')

    os.makedirs(label_save_dir, exist_ok=True)
    os.makedirs(pred_save_dir, exist_ok=True)
    # only visualise the main atom types:
    if limit_channels is not None:
        n_channels = min(limit_channels, lig.shape[1])
    else:
        n_channels = lig.shape[1]
    colors = np.zeros((n_channels, 4))
    for c in range(n_channels):
        colors[c] = mcolors.to_rgba(color_names[c])

    # Set transparency for all colors
    colors[:, 3] = 0.2
    fig, axs = plt.subplots(nrows=len(names), ncols=len(angles) * 2, figsize=(10, 15))

    for i, single_name in enumerate(names):
        for ang_idx, angle in enumerate(angles):
            # visualise the label
            label_save_path = os.path.join(label_save_dir, f"{single_name}_{ang_idx}.png")
            if not os.path.exists(label_save_path):
                # generate and save the label image
                fig_label = plt.figure(figsize=(5, 5))
                ax_label = fig_label.add_subplot(111, projection='3d')
                ax_label.grid(False)  # Turn off the grid
                ax_label.set_xticks([])  # Turn off x ticks
                ax_label.set_yticks([])  # Turn off y ticks
                ax_label.set_zticks([])  # Turn off z ticks
                for channel in range(n_channels):
                    ax_label.voxels(lig[i, channel], facecolors=colors[channel], edgecolors=colors[channel])
                ax_label.view_init(elev=angle[0], azim=angle[1])
                fig_label.savefig(label_save_path)
                plt.close(fig_label)

            # load the saved image for the label
            label_img = plt.imread(label_save_path)
            label_height, label_width, _ = label_img.shape
            lower = label_height // 4
            upper = label_height - lower
            if len(names) == 1:
                axs[ang_idx * 2].imshow(label_img)
                axs[ang_idx * 2].set_xlim(lower, upper)  # Updated cropping
                axs[ang_idx * 2].set_ylim(upper, lower)
                axs[ang_idx * 2].axis('off')
                axs[ang_idx * 2].set_title(f'Target {single_name}:{angle}')

            else:
                axs[i, ang_idx * 2].imshow(label_img)
                axs[i, ang_idx * 2].set_xlim(lower, upper)  # Updated cropping
                axs[i, ang_idx * 2].set_ylim(upper, lower)
                axs[i, ang_idx * 2].axis('off')
                axs[i, ang_idx * 2].set_title(f'Target {single_name}:{angle}')

            # visualise the prediction
            pred_save_path = os.path.join(pred_save_dir, f"{single_name}_{ang_idx}.png")
            # if not os.path.exists(pred_save_path):
            fig_pred = plt.figure(figsize=(8, 8))
            ax_pred = fig_pred.add_subplot(111, projection='3d')
            ax_pred.grid(False)  # Turn off the grid
            ax_pred.set_xticks([])  # Turn off x ticks
            ax_pred.set_yticks([])  # Turn off y ticks
            ax_pred.set_zticks([])  #
            for channel in range(n_channels):
                ax_pred.voxels(pred[i, channel], facecolors=colors[channel], edgecolors=colors[channel])
            ax_pred.view_init(elev=angle[0], azim=angle[1])
            fig_pred.savefig(pred_save_path)
            plt.close(fig_pred)

            # load the saved image for the prediction
            pred_img = plt.imread(pred_save_path)
            pred_height, pred_width, _ = pred_img.shape
            lower = pred_height // 4
            upper = pred_height - lower
            if len(names) == 1:
                axs[ang_idx * 2 + 1].imshow(pred_img)
                axs[ang_idx * 2 + 1].set_xlim(lower, upper)  # Updated cropping
                axs[ang_idx * 2 + 1].set_ylim(upper, lower)
                axs[ang_idx * 2 + 1].axis('off')
                axs[ang_idx * 2 + 1].set_title(f'Pred {single_name}:{angle}')
            else:
                axs[i, ang_idx * 2 + 1].imshow(pred_img)
                axs[i, ang_idx * 2 + 1].set_xlim(lower, upper)  # Updated cropping
                axs[i, ang_idx * 2 + 1].set_ylim(upper, lower)
                axs[i, ang_idx * 2 + 1].axis('off')
                axs[i, ang_idx * 2 + 1].set_title(f'Pred {single_name}:{angle}')
    shutil.rmtree(pred_save_dir)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    combined_image_path = os.path.join(save_dir, f'batch_{batch}_visualisation.png')
    plt.savefig(combined_image_path)
    if log_wandb:
        wandb.log({f'batch_{batch}_visualisation': wandb.Image(combined_image_path)})
    plt.close(fig)


if __name__=="__main__":
    import yaml
    from torch.utils.data import DataLoader
    from src.data.poc2mol.datasets import ParquetDataset
    from src.data.common.voxelization.config import Poc2MolDataConfig

    with open('configs/data/data.yaml', 'r') as file:
        yaml_config = yaml.safe_load(file)

    pdb_dir = "/mnt/disk2/VoxelDiffOuter/1b38"
    config = DataConfig(yaml_config['config'])
    class DictToClass:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                setattr(self, key, value)
    vox_config = DictToClass(yaml_config['config']['vox_config'])
    config.vox_config = vox_config
    dataset = ComplexDataset(
        config=config,
        pdb_dir=pdb_dir
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in loader:
        # visualise_batch(
        #     batch["ligand"],
        #     batch["protein"],
        #     batch["name"],
        #     angles=None,
        #     save_dir=pdb_dir,
        #     batch='none'
        # )
        show_3d_voxel_lig_only(
            batch["ligand"].squeeze(0),
            angles=None,
            save_dir=pdb_dir,
            identifier='1b38_pocket'
        )

        show_3d_voxel_lig_only(
            batch["protein"].squeeze(0),
            angles=None,
            save_dir=pdb_dir,
            identifier='1b38_ligand'
        )

        break


def visualize_2d_molecule_batch(batch_data, output_path, n_cols=3):
    """
    Visualizes a batch of molecules with their SMILES strings.
    
    Args:
        batch_data (list): List of dictionaries containing molecule data
        output_path (str): Path to save the output image
        n_cols (int): Number of columns in the grid (default=3)
    """
    n_mols = len(batch_data)
    n_rows = (n_mols + n_cols - 1) // n_cols  # Ceiling division
    
    # Create a figure with enough height to accommodate SMILES strings
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(6*n_cols, 5*n_rows))
    fig.suptitle('True vs Sampled Molecules', fontsize=16, y=0.95)
    
    # Flatten axes for easier indexing if there's only one row
    if n_rows == 1:
        axes = axes.reshape(2, -1)
    
    for idx, data in enumerate(batch_data):
        row = (idx // n_cols) * 2
        col = idx % n_cols
        
        # Draw true molecule
        true_img = Draw.MolToImage(data['true_mol'], size=(400, 400))
        axes[row, col].imshow(true_img)
        axes[row, col].axis('off')
        axes[row, col].set_title(f"True - {data['name']}", fontsize=12)
        
        # Add true SMILES string
        true_smiles = data['true_smiles']
        # add newline every 50 chars
        true_smiles = '\n'.join([true_smiles[i:i+50] for i in range(0, len(true_smiles), 50)])
        axes[row, col].text(0.5, -0.1, true_smiles, 
                          horizontalalignment='center',
                          verticalalignment='top',
                          transform=axes[row, col].transAxes,
                          fontsize=10,
                          wrap=True)
        
        # Draw sampled molecule
        sampled_img = Draw.MolToImage(data['sampled_mol'], size=(400, 400))
        axes[row + 1, col].imshow(sampled_img)
        axes[row + 1, col].axis('off')
        axes[row + 1, col].set_title(f"Sampled - {data['name']}", fontsize=12)
        
        # Add sampled SMILES string
        sampled_smiles = data['sampled_smiles']
        # add newline every 50 chars
        sampled_smiles = '\n'.join([sampled_smiles[i:i+50] for i in range(0, len(sampled_smiles), 50)])
        axes[row + 1, col].text(0.5, -0.1, sampled_smiles,
                               horizontalalignment='center',
                               verticalalignment='top',
                               transform=axes[row + 1, col].transAxes,
                               fontsize=10,
                               wrap=True)
    
    # Hide empty subplots
    for idx in range(n_mols, n_cols * n_rows):
        row = (idx // n_cols) * 2
        col = idx % n_cols
        axes[row, col].axis('off')
        axes[row + 1, col].axis('off')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save high-resolution figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def show_3d_voxel_lig_in_protein(lig_vox, protein_vox, angles=None, save_dir=None, identifier='lig_in_protein'):
    """
    Visualizes ligand voxels inside protein pocket.
    
    Args:
        lig_vox (np.ndarray): Ligand voxel grid of shape (n_channels, x, y, z)
        protein_vox (np.ndarray): Protein voxel grid of shape (n_channels, x, y, z)
        angles (list): List of (elevation, azimuth) angles to view from
        save_dir (str): Directory to save images
        identifier (str): Identifier for saved images
    """
    if not isinstance(lig_vox, np.ndarray):
        lig_vox = lig_vox.detach().cpu().numpy()
    if not isinstance(protein_vox, np.ndarray):
        protein_vox = protein_vox.detach().cpu().numpy()
        
    lig_vox = (lig_vox > 0.5).astype(int)
    protein_vox = (protein_vox > 0.5).astype(int)
    
    # Colors for ligand channels
    lig_colors = np.zeros((lig_vox.shape[0], 4))
    lig_colors[0] = mcolors.to_rgba('green')  # carbon_ligand
    lig_colors[1] = mcolors.to_rgba('red')  # oxygen_ligand
    lig_colors[2] = mcolors.to_rgba('blue')  # nitrogen_ligand
    lig_colors[3] = mcolors.to_rgba('yellow')  # sulfur_ligand
    lig_colors[4] = mcolors.to_rgba('cyan')  # chlorine_ligand
    lig_colors[5] = mcolors.to_rgba('darkcyan')  # fluorine_ligand
    lig_colors[6] = mcolors.to_rgba('magenta')  # iodine_ligand
    lig_colors[7] = mcolors.to_rgba('brown')  # bromine_ligand
    lig_colors[8] = mcolors.to_rgba('purple')  # other_ligand
    
    # Set transparency for ligand colors
    lig_colors[:, 3] = 0.3
    
    # Color for protein (semi-transparent gray)
    protein_color = np.array([0.7, 0.7, 0.7, 0.05])  # Gray with low opacity
    protein_color[:4] = mcolors.to_rgba('pink')
    protein_color[3] = 0.05
    if angles is None:
        # cover all angles in 45 degree increments
        angles = ALL_ANGLES
        
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot protein voxels first (as background)
    for channel in range(protein_vox.shape[0]):
        ax.voxels(protein_vox[channel], facecolors=protein_color, edgecolors=protein_color)
    
    # Plot ligand voxels on top
    for channel in range(lig_vox.shape[0]):
        ax.voxels(lig_vox[channel], facecolors=lig_colors[channel], edgecolors=lig_colors[channel])
    
    # Save images from different angles
    for i, angle in enumerate(angles):
        ax.view_init(elev=angle[0], azim=angle[1])
        ax.axis('off')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/{identifier}_angle_{i + 1}.png", dpi=300, bbox_inches='tight')
            
    plt.close(fig)