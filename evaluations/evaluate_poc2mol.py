import os
import torch
import yaml
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from src.models.poc2mol import Poc2Mol
from src.evaluation.visual import visualise_batch, show_3d_voxel_lig_only
from src.utils.utils import get_config_from_cpt_path
from src.data.poc2mol.datasets import ComplexDataset
from src.data.common.voxelization.config import Poc2MolDataConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Poc2Mol model")
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="logs/vox2smilesZincAndPoc2MolOutputs/runs/2025-03-22_21-18-58/checkpoints/last.ckpt",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="configs/data/poc2mol_PDBbind_data.yaml",
        help="Path to the data configuration file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for evaluation"
    )
    return parser.parse_args()


def load_model(ckpt_path):
    print(f"Loading model from {ckpt_path}")
    model = Poc2Mol.load_from_checkpoint(ckpt_path)
    model = model.to(torch.float16)
    model.eval()
    return model


def load_dataset(config):
    dataset_config = Poc2MolDataConfig(
        batch_size=config['config']['batch_size'],
        ligand_channel_indices=config['config']['ligand_channel_indices'],
        protein_channel_indices=config['config']['protein_channel_indices'],
        vox_size=config['config']['vox_size'],
        box_dims=config['config']['box_dims'],
        random_rotation=False,
        random_translation=6.0,
        has_protein=config['config']['has_protein'],
        ligand_channel_names=config['config']['ligand_channel_names'],
        protein_channel_names=config['config']['protein_channel_names'],
        protein_channels=config['config']['protein_channels']
    )
    dataset = ComplexDataset(
        config=dataset_config,
        pdb_dir=config['pdb_dir'],
        translation=6.0,
        rotate=False
    )
    return dataset


def evaluate_model(model, dataset, output_dir, num_samples=10, batch_size=4):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    sample_count = 0
    batch_idx = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to the same device as model
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Generate predictions
            outputs = model(batch["protein"])
            
            # Visualize the results
            batch_save_dir = os.path.join(output_dir, f"batch_{batch_idx}")
            os.makedirs(batch_save_dir, exist_ok=True)
            
            # Get the names, ligands, and predictions for visualization
            names = batch["name"][:batch_size]
            ligands = batch["ligand"][:batch_size]
            predictions = outputs[:batch_size]
            
            # Visualize the batch
            visualise_batch(
                ligands,
                predictions,
                names,
                save_dir=batch_save_dir,
                batch=str(batch_idx),
                reuse_labels=False,
                log_wandb=False
            )
    
    print(f"Evaluation complete. Results saved to {output_dir}")



def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config = get_config_from_cpt_path(args.ckpt_path)
    
    dataset = load_dataset(config['data'])
    
    # Load model
    model = load_model(args.ckpt_path)

    
    # Evaluate model
    evaluate_model(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size
    )
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()