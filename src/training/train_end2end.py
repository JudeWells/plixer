import os
import argparse
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from src.models.poc2mol import Poc2Mol
from src.models.vox2smiles import VoxToSmilesModel
from src.models.poc2smiles import CombinedProteinToSmilesModel
from src.data.poc2smiles.data_module import ProteinLigandSmilesDataModule
from src.data.common.voxelization.config import Poc2MolDataConfig, Vox2SmilesDataConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train end-to-end Poc2Smiles model")
    
    # Paths
    parser.add_argument("--poc2mol_checkpoint", type=str, required=True, help="Path to Poc2Mol checkpoint")
    parser.add_argument("--vox2smiles_checkpoint", type=str, required=True, help="Path to Vox2Smiles checkpoint")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Path to PDB directory")
    parser.add_argument("--val_pdb_dir", type=str, required=True, help="Path to validation PDB directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--freeze_poc2mol", action="store_true", help="Freeze Poc2Mol model")
    parser.add_argument("--freeze_vox2smiles", action="store_true", help="Freeze Vox2Smiles model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Poc2Mol model
    poc2mol_model = Poc2Mol.load_from_checkpoint(args.poc2mol_checkpoint)
    if args.freeze_poc2mol:
        poc2mol_model.freeze()
    
    # Load Vox2Smiles model
    vox2smiles_model = VoxToSmilesModel.load_from_checkpoint(args.vox2smiles_checkpoint)
    if args.freeze_vox2smiles:
        vox2smiles_model.freeze()
    
    # Create combined model
    combined_model = CombinedProteinToSmilesModel(
        poc2mol_model=poc2mol_model,
        vox2smiles_model=vox2smiles_model,
        config={
            "lr": args.lr,
            "weight_decay": 0.0,
            "step_size": 100,
            "gamma": 0.99,
            "img_save_dir": args.output_dir,
        },
    )
    
    # Create data configs
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
    
    # Create data module
    data_module = ProteinLigandSmilesDataModule(
        poc2mol_config=poc2mol_config,
        vox2smiles_config=vox2smiles_config,
        pdb_dir=args.pdb_dir,
        val_pdb_dir=args.val_pdb_dir,
    )
    
    # Set up logger
    logger = WandbLogger(
        project="poc2smiles-end2end",
        name=f"end2end-lr-{args.lr}-freeze_poc2mol-{args.freeze_poc2mol}-freeze_vox2smiles-{args.freeze_vox2smiles}",
        save_dir=args.output_dir,
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="poc2smiles-end2end-{epoch:02d}-{val_loss:.4f}",
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
    
    # Train combined model
    trainer.fit(
        model=combined_model,
        datamodule=data_module,
    )
    
    # Save final model
    trainer.save_checkpoint(os.path.join(args.output_dir, "poc2smiles-end2end-final.ckpt"))


if __name__ == "__main__":
    main() 