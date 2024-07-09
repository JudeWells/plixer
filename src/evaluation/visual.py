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
    vox = vox[:6]  # only ligand channels
    # Define colors for each atom type
    colors = np.zeros((vox.shape[0], 4))
    colors[0] = mcolors.to_rgba('green')  # carbon_ligand
    colors[1] = mcolors.to_rgba('white')  # hydrogen_ligand
    colors[2] = mcolors.to_rgba('red')  # oxygen_ligand
    colors[3] = mcolors.to_rgba('blue')  # nitrogen_ligand
    colors[4] = mcolors.to_rgba('yellow')  # sulfur_ligand
    colors[5] = mcolors.to_rgba('magenta')  # other_ligand

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
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/{identifier}_angle_{i + 1}.png")
    plt.close()
    plt.close(fig)


def visualise_batch(lig, pred, names, angles=None, save_dir=None, batch='none'):
    """
    Visualise the label and the predictions for a batch of ligands
    show the predictions side by side in a single plot
    show 2 angles for each ligand (label and pred) in one row of the plot
    """
    if not isinstance(lig, np.ndarray):
        lig = lig.detach().cpu().numpy()
    lig = (lig > 0.5).astype(int)

    if not isinstance(pred, np.ndarray):
        pred = pred.detach().cpu().numpy()
    pred = (pred > 0.5).astype(int)

    if angles is None:
        angles = [(45, 45), (45, -45)]
    label_save_dir = os.path.join(save_dir,  '..', '..', '..', 'label_images')
    pred_save_dir = os.path.join(save_dir, f'batch_{batch}', 'predictions')

    os.makedirs(label_save_dir, exist_ok=True)
    os.makedirs(pred_save_dir, exist_ok=True)
    # Define colors for each atom type
    colors = np.zeros((lig.shape[1], 4))
    colors[0] = mcolors.to_rgba('green')  # carbon_ligand
    colors[1] = mcolors.to_rgba('white')  # hydrogen_ligand
    colors[2] = mcolors.to_rgba('red')  # oxygen_ligand
    colors[3] = mcolors.to_rgba('blue')  # nitrogen_ligand
    colors[4] = mcolors.to_rgba('yellow')  # sulfur_ligand
    colors[5] = mcolors.to_rgba('magenta')  # other_ligand
    colors[6] = mcolors.to_rgba('magenta')  # other_ligand
    colors[7] = mcolors.to_rgba('magenta')  # other_ligand
    colors[8] = mcolors.to_rgba('magenta')  # other_ligand
    colors[9] = mcolors.to_rgba('cyan')  # other_ligand

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
                for channel in range(lig.shape[1]):
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
            for channel in range(pred.shape[1]):
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
    wandb.log({f'batch_{batch}_visualisation': wandb.Image(combined_image_path)})
    plt.close(fig)