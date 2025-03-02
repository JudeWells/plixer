import os
import argparse
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from src.models.poc2mol import Poc2Mol
from src.models.vox2smiles import VoxToSmilesModel
from src.data.poc2mol.data_module import ComplexDataModule
from src.data.vox2smiles.data_module import VoxMilesDataModule
from src.data.vox2smiles.datasets import Poc2MolOutputDataset, CombinedDataset
from src.data.common.voxelization.config import Poc2MolDataConfig, Vox2SmilesDataConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Vox2Smiles on Poc2Mol outputs")
    
    # Paths
    parser.add_argument("--poc2mol_checkpoint", type=str, required=True, help="Path to Poc2Mol checkpoint")
    parser.add_argument("--vox2smiles_checkpoint", type=str, required=True, help="Path to Vox2Smiles checkpoint")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Path to PDB directory")
    parser.add_argument("--val_pdb_dir", type=str, required=True, help="Path to validation PDB directory")
    parser.add_argument("--voxmiles_data_path", type=str, required=True, help="Path to VoxMiles data")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--ratio", type=float, default=0.5, help="Ratio of Poc2Mol outputs to VoxMiles data")
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
    poc2mol_model.eval()
    poc2mol_model.freeze()
    
    # Load Vox2Smiles model
    vox2smiles_model = VoxToSmilesModel.load_from_checkpoint(args.vox2smiles_checkpoint)
    
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
    
    vox2smiles_data_module = VoxMilesDataModule(
        config=vox2smiles_config,
        data_path=args.voxmiles_data_path,
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
        voxmiles_dataset=vox2smiles_data_module.train_dataset,
        ratio=args.ratio,
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=vox2smiles_data_module.collate_fn,
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