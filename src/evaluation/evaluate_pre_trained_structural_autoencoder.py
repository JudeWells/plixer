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
from src.models.structural_autoencoder import StructuralAutoEncoder
from src.evaluation.visual import show_3d_voxel_lig_only, visualise_batch
from src.data.poc2mol_data_module import DataConfig
from src.constants import _3_to_1, _3_to_num

# A minimal vox_config-like object for example
class VoxConfigMock:
    def __init__(self):
        self.vox_size = 0.75
        self.box_dims = [24.0, 24.0, 24.0]
        self.random_rotation = True
        self.random_translation = 0.0


# --------------------------------------------------------------------------------
# A simple MLP classifier to classify the amino acid at the center from the latent embedding
# --------------------------------------------------------------------------------
class AminoAcidClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=20, hidden_dim=256):
        """
        A simple feed-forward network for amino acid classification.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------------------------------------
# A custom dataset that returns the voxel plus the center amino acid label
# --------------------------------------------------------------------------------
class AminoAcidDataset(StructuralPretrainDataset):
    """
    Extends StructuralPretrainDataset to also return the amino acid type
    at the center residue. For simplicity, assume we have 20 standard amino acids.
    """
    def __init__(self, config, pdb_dir, rotate=True, max_samples=None):
        """
        max_samples: limit the total number of samples if desired
        """
        super().__init__(config, pdb_dir, rotate=rotate, use_ca=True)
        self.max_samples = max_samples

    def __getitem__(self, idx):
        # The original data:
        data = super().__getitem__(idx)
        # data has "input" (voxel) and "name" (filename)

        # For demonstration, we will pretend all centers are in the "standard 20 AAs".
        # In a real scenario, you'd need to parse the center residue's amino acid type.
        # Here, we just store a dummy label for demonstration:
        # Let's assume each sample's label is random among 20 classes:
        try:
            label = _3_to_num[data["aa_type"]]
        except KeyError:
            label = 20

        return {
            "input": data["input"],
            "name": data["name"],
            "label": label,
        }

    def __len__(self):
        # Optionally limit total samples
        full_length = super().__len__()
        if self.max_samples is not None:
            return min(full_length, self.max_samples)
        return full_length

# --------------------------------------------------------------------------------
# Utility function to generate and save reconstructions of a few samples
# --------------------------------------------------------------------------------
def generate_and_visualize_samples(model, dataset, num_samples=10, save_dir="outputs/visual_samples"):
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
            recon, _ = model(inp)

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
    cfg_yaml_path = "logs/poc2mol/runs/2025-01-08_20-24-57/config.yaml"
    # Path to the pretrained structural autoencoder checkpoint
    ckpt_path = "logs/poc2mol/runs/2025-01-08_20-24-57/checkpoints/epoch_020.ckpt"

    # Path to a directory with PDB files
    pdb_dir = "/mnt/disk2/pinder/pinder/2024-02/pdbs"

    # We will create a config for the StructuralPretrainDataset
    # limiting the total number of samples to 6000 (5000 train, 500 val, 500 test).
    config = DataConfig(
        vox_config = None,  # We'll define a minimal structure for demonstration
        batch_size = 32,
        dtype=torch.float32,  # for demonstration
        max_atom_dist=24.0
    )

    mock_vox_config = VoxConfigMock()
    config.vox_config = mock_vox_config

    # Create dataset that also returns amino acid label at the center
    full_dataset = AminoAcidDataset(
        config=config,
        pdb_dir=pdb_dir,
        rotate=True,
        max_samples=6000  # total
    )

    # Split into train(5000), val(500), test(500)
    lengths = [5000, 500, 500]
    train_data, val_data, test_data = random_split(full_dataset, lengths)

    # --------------------------------------------------------------------------------
    # 1) Load the pretrained autoencoder
    # --------------------------------------------------------------------------------
    # Make sure to call load_from_checkpoint on the class directly, not on an instance
    model_ae = StructuralAutoEncoder.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        in_channels=8,      # These should match the AE's original hyperparams
        f_maps=64,
        latent_dim=512,
        num_levels=5,
        layer_order='gcr',
        num_groups=8,
        conv_padding=1,
        dropout_prob=0.1,
        lr=1e-3,
        weight_decay=1.0e-05,
        loss=None,
    )
    model_ae.eval()

    # --------------------------------------------------------------------------------
    # 2) Generate a few decoded samples (visualization)
    # --------------------------------------------------------------------------------
    # Just for visual: store 10 samples as images
    print("Generating a few decoded samples...")
    # generate_and_visualize_samples(model_ae, full_dataset, num_samples=1, save_dir="outputs/visual_samples")

    # --------------------------------------------------------------------------------
    # 3) Create feature dataset by encoding samples
    # --------------------------------------------------------------------------------
    print("Encoding data for amino acid classification (this may take a while)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ae.to(device)
    model_ae.eval()

    def encode_dataset(dataset_obj):
        loader = DataLoader(dataset_obj, batch_size=32, shuffle=False, num_workers=2)
        embeddings, labels = [], []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                print(f"Processing batch {i+1} of {len(loader)}")
                inp = batch["input"].to(device)
                z = model_ae.encode(inp)  # shape: [B, latent_dim, D/32, H/32, W/32] typically
                # Flatten
                z = torch.mean(z, dim=(2, 3, 4))  # simple global pooling for demonstration
                embeddings.append(z.cpu())
                labels.append(batch["label"])  # shape [B]
        return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)
    embded_dir = "outputs/aa_classifier_data"
    os.makedirs(embded_dir, exist_ok=True)
    if not os.path.exists(os.path.join(embded_dir, "train_embeddings.pt")):
        train_embeddings, train_labels = encode_dataset(train_data)
        val_embeddings, val_labels = encode_dataset(val_data)
        test_embeddings, test_labels = encode_dataset(test_data)


        torch.save(train_embeddings, os.path.join(embded_dir, "train_embeddings.pt"))
        torch.save(train_labels, os.path.join(embded_dir, "train_labels.pt"))
        torch.save(val_embeddings, os.path.join(embded_dir, "val_embeddings.pt"))
        torch.save(val_labels, os.path.join(embded_dir, "val_labels.pt"))
        torch.save(test_embeddings, os.path.join(embded_dir, "test_embeddings.pt"))
        torch.save(test_labels, os.path.join(embded_dir, "test_labels.pt"))
    else:
        train_embeddings = torch.load(os.path.join(embded_dir, "train_embeddings.pt"))
        train_labels = torch.load(os.path.join(embded_dir, "train_labels.pt"))
        val_embeddings = torch.load(os.path.join(embded_dir, "val_embeddings.pt"))
        val_labels = torch.load(os.path.join(embded_dir, "val_labels.pt"))
        test_embeddings = torch.load(os.path.join(embded_dir, "test_embeddings.pt"))
        test_labels = torch.load(os.path.join(embded_dir, "test_labels.pt"))

    # --------------------------------------------------------------------------------
    # 4) Train a simple classifier (manual training loop)
    # --------------------------------------------------------------------------------
    print("Training a simple amino acid classifier...")

    input_dim = train_embeddings.shape[1]
    num_classes = 21
    epochs = 500

    classifier = AminoAcidClassifier(input_dim=input_dim, num_classes=num_classes, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()

    # Remove usage of EarlyStopping callback in a manual loop.
    # We'll implement a very simple manual early stopping logic instead.
    best_val_loss = float("inf")
    no_improve_count = 0
    patience = 30

    wandb_logger = WandbLogger(project="amino_acid_classification_demo", name="struct_ae_eval")

    for epoch in range(1, epochs + 1):
        classifier.train()
        perm = torch.randperm(train_embeddings.size(0))
        train_embeddings_shuf = train_embeddings[perm]
        train_labels_shuf = train_labels[perm]

        # Mini-batch training
        batch_size = 128
        num_batches = (len(train_embeddings_shuf) + batch_size - 1) // batch_size
        running_loss = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            xb = train_embeddings_shuf[start:end].to(device)
            yb = train_labels_shuf[start:end].to(device)

            optimizer.zero_grad()
            preds = classifier(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / num_batches

        # Validation
        classifier.eval()
        with torch.no_grad():
            preds_val = classifier(val_embeddings.to(device))
            val_loss = criterion(preds_val, val_labels.to(device)).item()

        print(f"Epoch {epoch}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        wandb_logger.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        # Manual early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(classifier.state_dict(), "best_classifier.pt")
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print("Early stopping triggered.")
                break

    # Load best weights
    classifier.load_state_dict(torch.load("best_classifier.pt"))

    # --------------------------------------------------------------------------------
    # 5) Evaluate on test set
    # --------------------------------------------------------------------------------
    classifier.eval()
    with torch.no_grad():
        preds_test = classifier(test_embeddings.to(device))
        test_loss = criterion(preds_test, test_labels.to(device)).item()
        correct = (preds_test.argmax(dim=1) == test_labels.to(device)).sum().item()
        accuracy = correct / test_labels.size(0)

        preds_train = classifier(train_embeddings.to(device))
        correct = (preds_train.argmax(dim=1) == train_labels.to(device)).sum().item()
        train_accuracy = correct / train_labels.size(0)
    print(f"Train Accuracy = {train_accuracy:.4f}")
    print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {accuracy:.4f}")
    wandb_logger.log_metrics({"test_loss": test_loss, "test_accuracy": accuracy})
    print("Done evaluating the pre-trained structural autoencoder.")