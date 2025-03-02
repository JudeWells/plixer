"""
Loads the pretrained structural autoencoder.
function 1: generates a few decoded samples and saves them as images
function 2: uses the encode to generate features used to train an
amino acid classifier
"""

import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, random_split
from lightning import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

# Imports from this codebase
from src.data.poc2mol_datasets import StructuralPretrainDataset, ProteinComplex, MolecularParserWrapper
from src.models.structural_vae import StructuralVariationalAutoEncoder
from src.evaluation.visual import show_3d_voxel_lig_only, visualise_batch
from src.data.poc2mol_data_module import DataConfig
from src.constants import _3_to_1, _3_to_num

# A minimal vox_config-like object for example
class VoxConfigMock:
    def __init__(self):
        self.vox_size = 0.75
        self.box_dims = [24.0, 24.0, 24.0]
        self.random_rotation = False
        self.random_translation = 0.0


# --------------------------------------------------------------------------------
# Utility function to generate and save reconstructions of a few samples
# --------------------------------------------------------------------------------
def generate_and_visualize_samples(model, dataset, num_samples=10, save_dir="outputs/visual_samples_OVERFIT_v2"):
    """
    Select a few samples, pass through autoencoder, and visualize.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    indices = random.sample(range(len(dataset)), k=min(num_samples, len(dataset)))
    device = next(model.parameters()).device

    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            inp = sample["input"].unsqueeze(0).to(device)
            name = sample["name"]
            recon, _, _ = model(inp)

            # Move to CPU for visualization
            inp_cpu = inp.squeeze(0).cpu()
            recon_cpu = recon.squeeze(0).cpu()

            # Save side-by-side visual
            fig_save_dir = os.path.join(save_dir, f"sample_{i+1}_{name}")
            os.makedirs(fig_save_dir, exist_ok=True)

            colors = np.zeros((inp_cpu.shape[0], 4))
            colors[0] = mcolors.to_rgba('green')  # carbon_ligand
            colors[1] = mcolors.to_rgba('blue')  # nitrogen_ligand
            colors[2] = mcolors.to_rgba('red')  # oxygen_ligand
            colors[3] = mcolors.to_rgba('yellow')  # sulfur_ligand
            colors[4] = mcolors.to_rgba('magenta')  # phosphorus_ligand
            colors[5] = mcolors.to_rgba('cyan')  # halogen_ligand
            colors[6] = mcolors.to_rgba('grey')  # metal_ligand
            # Using show_3d_voxel_lig_only for quick per-voxel-channel visualization
            show_3d_voxel_lig_only(inp_cpu, angles=None, save_dir=fig_save_dir, identifier="input", colors=colors)
            show_3d_voxel_lig_only(recon_cpu, angles=None, save_dir=fig_save_dir, identifier="recon", colors=colors)

# --------------------------------------------------------------------------------
# Main script
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    seed_everything(42)
    cfg_yaml_path = "logs/vae_pretrain/runs/2025-01-27_19-10-37/.hydra/config.yaml"
    # Path to the pretrained structural autoencoder checkpoint
    ckpt_path = "logs/vae_pretrain/runs/2025-01-27_19-10-37/checkpoints/last.ckpt"

    # Path to a directory with PDB files
    pdb_dir = "/mnt/disk2/plinder/2024-06/v2/systems/6zzs__1__1.B__1.I_1.J"

    # We will create a config for the StructuralPretrainDataset
    # limiting the total number of samples to 6000 (5000 train, 500 val, 500 test).
    config = DataConfig(
        batch_size=16,
        dtype=torch.float16,
        max_atom_dist=24.0,
        coord_indices=[100, 300, 500, 700, 900],

        vox_config=VoxConfigMock(),
    )

    # mock_vox_config = VoxConfigMock()
    # config.vox_config = mock_vox_config

    # Create dataset that also returns amino acid label at the center
    full_dataset = StructuralPretrainDataset(config, pdb_dir)


    # --------------------------------------------------------------------------------
    # 1) Load the pretrained autoencoder
    # --------------------------------------------------------------------------------
    # Make sure to call load_from_checkpoint on the class directly, not on an instance
    model_ae = StructuralVariationalAutoEncoder.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        in_channels=8,      # These should match the AE's original hyperparams
        f_maps=64,
        latent_dim=512,
        num_levels=5,
        layer_order='gcr',
        num_groups=8,
        conv_padding=1,
        dropout_prob=0.1,
        lr=1e-4,
        weight_decay=1.0e-05,
        loss=None,
        half_decoder_layers=False,
        half_decoder_channels=False,
        beta=0.05,
    )
    model_ae.eval()

    # --------------------------------------------------------------------------------
    # 2) Generate a few decoded samples (visualization)
    # --------------------------------------------------------------------------------
    # Just for visual: store 10 samples as images
    print("Generating a few decoded samples...")
    generate_and_visualize_samples(
        model_ae, 
        full_dataset, 
        num_samples=10, 
        save_dir="outputs/visual_samples_overfit_beta0")
