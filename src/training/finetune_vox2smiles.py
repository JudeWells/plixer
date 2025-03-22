import os
import sys
import argparse
import torch
import multiprocessing
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import hydra
from omegaconf import OmegaConf

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
# This must be done at the beginning of the program
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.models.poc2mol import Poc2Mol, ResUnetConfig
from src.models.vox2smiles import VoxToSmilesModel
from src.data.poc2mol.data_module import ComplexDataModule
from src.data.vox2smiles.data_module import Vox2SmilesDataModule
from src.data.vox2smiles.datasets import Poc2MolOutputDataset, CombinedDataset
from src.data.common.voxelization.config import Poc2MolDataConfig, Vox2SmilesDataConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Vox2Smiles on Poc2Mol outputs")
    
    # Paths
    parser.add_argument("--poc2mol_checkpoint", type=str, required=True, help="Path to Poc2Mol checkpoint")
    parser.add_argument("--vox2smiles_checkpoint", type=str, required=True, help="Path to Vox2Smiles checkpoint")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Path to PDB directory")
    parser.add_argument("--val_pdb_dir", type=str, required=True, help="Path to validation PDB directory")
    parser.add_argument("--vox2smiles_data_path", type=str, required=True, help="Path to Vox2Smiles data")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--ratio", type=float, default=0.5, help="Ratio of Poc2Mol outputs to Vox2Smiles data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Poc2Mol model with proper configuration
    # First, load the default configuration from the config file
    config_path = os.path.join(project_root, "configs/model/poc2mol.yaml")
    poc2mol_config = OmegaConf.load(config_path)
    
    # Load the model with the configuration
    poc2mol_model = Poc2Mol.load_from_checkpoint(
        args.poc2mol_checkpoint,
        config=ResUnetConfig(
            in_channels=poc2mol_config.config.in_channels,
            out_channels=poc2mol_config.config.out_channels,
            final_sigmoid=poc2mol_config.config.final_sigmoid,
            f_maps=poc2mol_config.config.f_maps,
            layer_order=poc2mol_config.config.layer_order,
            num_groups=poc2mol_config.config.num_groups,
            num_levels=poc2mol_config.config.num_levels,
            conv_padding=poc2mol_config.config.conv_padding,
            conv_upscale=poc2mol_config.config.conv_upscale,
            upsample=poc2mol_config.config.upsample,
            dropout_prob=poc2mol_config.config.dropout_prob,
            basic_module=hydra.utils.get_class(poc2mol_config.config.basic_module),
            loss=poc2mol_config.config.loss
        )
    )
    poc2mol_model.eval()
    poc2mol_model.freeze()
    
    # Load Vox2Smiles model with its saved configuration
    # This will use the configuration stored in the checkpoint
    vox2smiles_model = VoxToSmilesModel.load_from_checkpoint(
        args.vox2smiles_checkpoint,
        # strict=True
    )
    
    # Load the vox2smiles configuration from the config file for data module setup
    vox2smiles_config_path = os.path.join(project_root, "configs/model/poc2smiles.yaml")
    vox2smiles_full_config = OmegaConf.load(vox2smiles_config_path)
    
    # Create data modules
    poc2mol_config = Poc2MolDataConfig(
        vox_size=0.75,
        box_dims=[24.0, 24.0, 24.0],
        random_rotation=True,
        random_translation=6.0,
        max_atom_dist=32.0,
        batch_size=args.batch_size,
    )
    
    vox2smiles_config = Vox2SmilesDataConfig(
        vox_size=0.75,
        box_dims=[24.0, 24.0, 24.0],
        random_rotation=True,
        random_translation=6.0,
        max_atom_dist=32.0,
        batch_size=args.batch_size,
        max_smiles_len=200,
    )
    
    # Create data modules
    poc2mol_data_module = ComplexDataModule(
        config=poc2mol_config,
        pdb_dir=args.pdb_dir,
        val_pdb_dir=args.val_pdb_dir,
    )
    
    vox2smiles_data_module = Vox2SmilesDataModule(
        config=vox2smiles_config,
        data_path=args.vox2smiles_data_path,
    )
    
    # Set up data modules
    poc2mol_data_module.setup()
    vox2smiles_data_module.setup()
    
    # Create datasets
    poc2mol_output_dataset = Poc2MolOutputDataset(
        poc2mol_model=poc2mol_model,
        complex_dataset=poc2mol_data_module.train_dataset,
        tokenizer=vox2smiles_data_module.tokenizer,
        max_smiles_len=vox2smiles_config.max_smiles_len,
    )
    
    combined_dataset = CombinedDataset(
        poc2mol_output_dataset=poc2mol_output_dataset,
        vox2smiles_dataset=vox2smiles_data_module.train_dataset,
        ratio=args.ratio,
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=vox2smiles_data_module.collate_fn,
        pin_memory=False,
        persistent_workers=False
    )
    
    val_loader = vox2smiles_data_module.val_dataloader()
    
    # Set up logger
    logger = WandbLogger(
        project="vox2smiles-finetune",
        name=f"finetune-ratio-{args.ratio}-lr-{args.lr}",
        save_dir=args.output_dir,
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="vox2smiles-finetune-{epoch:02d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Set up trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator="auto",
        devices=1,
    )
    
    # Fine-tune Vox2Smiles model
    trainer.fit(
        model=vox2smiles_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    # Save final model
    trainer.save_checkpoint(os.path.join(args.output_dir, "vox2smiles-finetune-final.ckpt"))


if __name__ == "__main__":
    main() 