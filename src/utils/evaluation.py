import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
from matplotlib.animation import FuncAnimation

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

    fig = plt.figure(figsize=(8, 8))
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
