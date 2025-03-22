import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import wandb


def show_3d_voxel_lig_only(vox, angles=None, save_dir=None, identifier='lig'):
    if not isinstance(vox, np.ndarray):
        vox = vox.detach().cpu().numpy()
    vox = (vox > 0.5).astype(int)
    # vox = vox[:5]  # only ligand channels
    colors = np.zeros((vox.shape[0], 4))
    colors[0] = mcolors.to_rgba('green')  # carbon_ligand
    # colors[1] = mcolors.to_rgba('white')  # hydrogen_ligand
    colors[1] = mcolors.to_rgba('red')  # oxygen_ligand
    colors[2] = mcolors.to_rgba('blue')  # nitrogen_ligand
    colors[3] = mcolors.to_rgba('yellow')  # sulfur_ligand
    colors[4] = mcolors.to_rgba('magenta')  # chlorine_ligand
    colors[5] = mcolors.to_rgba('magenta')  # fluorine_ligand
    colors[6] = mcolors.to_rgba('magenta')  # iodine_ligand
    colors[7] = mcolors.to_rgba('magenta')  # bromine_ligand
    colors[8] = mcolors.to_rgba('cyan')  # other_ligand

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
        lig = lig.detach().cpu().numpy()
    lig = (lig > 0.5).astype(int)

    if not isinstance(pred, np.ndarray):
        pred = pred.detach().cpu().numpy()
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
    from src.data.poc2mol.datasets import PlinderParquetDataset
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