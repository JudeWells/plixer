import os
import argparse
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning import Trainer

from src.models.poc2mol import Poc2Mol
from src.models.vox2smiles import VoxToSmilesModel
from src.models.poc2smiles import CombinedProteinToSmilesModel
from src.data.poc2smiles.data_module import ProteinLigandSmilesDataModule
from src.utils.metrics import calculate_metrics


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for running the end-to-end model.
    
    Args:
        cfg: Hydra configuration
    """
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seed
    if "seed" in cfg:
        torch.manual_seed(cfg.seed)
    
    # Create output directory
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    
    if cfg.mode == "train_end2end":
        train_end2end(cfg)
    elif cfg.mode == "finetune_vox2smiles":
        finetune_vox2smiles(cfg)
    elif cfg.mode == "generate":
        generate(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


def train_end2end(cfg: DictConfig):
    """
    Train the end-to-end model.
    
    Args:
        cfg: Hydra configuration
    """
    # Load Poc2Mol model
    poc2mol_model = Poc2Mol.load_from_checkpoint(cfg.training.poc2mol_checkpoint)
    if cfg.training.freeze_poc2mol:
        poc2mol_model.freeze()
    
    # Load Vox2Smiles model
    vox2smiles_model = VoxToSmilesModel.load_from_checkpoint(cfg.training.vox2smiles_checkpoint)
    if cfg.training.freeze_vox2smiles:
        vox2smiles_model.freeze()
    
    # Create combined model
    combined_model = CombinedProteinToSmilesModel(
        poc2mol_model=poc2mol_model,
        vox2smiles_model=vox2smiles_model,
        config={
            "lr": cfg.training.lr,
            "weight_decay": 0.0,
            "step_size": 100,
            "gamma": 0.99,
            "img_save_dir": cfg.training.output_dir,
        },
    )
    
    # Create data module
    data_module = hydra.utils.instantiate(cfg.data.poc2smiles_data)
    
    # Create trainer
    trainer = hydra.utils.instantiate(
        cfg.training.trainer,
        logger=hydra.utils.instantiate(cfg.training.logger),
        callbacks=[
            hydra.utils.instantiate(cfg.training.callbacks.model_checkpoint),
            hydra.utils.instantiate(cfg.training.callbacks.lr_monitor),
            hydra.utils.instantiate(cfg.training.callbacks.early_stopping),
        ],
    )
    
    # Train combined model
    trainer.fit(
        model=combined_model,
        datamodule=data_module,
    )
    
    # Save final model
    trainer.save_checkpoint(os.path.join(cfg.training.output_dir, "poc2smiles-end2end-final.ckpt"))


def finetune_vox2smiles(cfg: DictConfig):
    """
    Fine-tune the Vox2Smiles model on Poc2Mol outputs.
    
    Args:
        cfg: Hydra configuration
    """
    from src.data.vox2smiles.datasets import Poc2MolOutputDataset, CombinedDataset
    
    # Load Poc2Mol model
    poc2mol_model = Poc2Mol.load_from_checkpoint(cfg.training.poc2mol_checkpoint)
    poc2mol_model.eval()
    poc2mol_model.freeze()
    
    # Load Vox2Smiles model
    vox2smiles_model = VoxToSmilesModel.load_from_checkpoint(cfg.training.vox2smiles_checkpoint)
    
    # Create data modules
    poc2mol_data_module = hydra.utils.instantiate(cfg.data.poc2mol_data)
    vox2smiles_data_module = hydra.utils.instantiate(cfg.data.vox2smiles_data)
    
    # Set up data modules
    poc2mol_data_module.setup()
    vox2smiles_data_module.setup()
    
    # Create datasets
    poc2mol_output_dataset = Poc2MolOutputDataset(
        poc2mol_model=poc2mol_model,
        complex_dataset=poc2mol_data_module.train_dataset,
        tokenizer=vox2smiles_data_module.tokenizer,
        max_smiles_len=cfg.data.vox2smiles_data.config.max_smiles_len,
    )
    
    combined_dataset = CombinedDataset(
        poc2mol_output_dataset=poc2mol_output_dataset,
        voxmiles_dataset=vox2smiles_data_module.train_dataset,
        ratio=cfg.training.ratio,
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=vox2smiles_data_module.collate_fn,
    )
    
    val_loader = vox2smiles_data_module.val_dataloader()
    
    # Create trainer
    trainer = hydra.utils.instantiate(
        cfg.training.trainer,
        logger=hydra.utils.instantiate(cfg.training.logger),
        callbacks=[
            hydra.utils.instantiate(cfg.training.callbacks.model_checkpoint),
            hydra.utils.instantiate(cfg.training.callbacks.lr_monitor),
            hydra.utils.instantiate(cfg.training.callbacks.early_stopping),
        ],
    )
    
    # Fine-tune Vox2Smiles model
    trainer.fit(
        model=vox2smiles_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    # Save final model
    trainer.save_checkpoint(os.path.join(cfg.training.output_dir, "vox2smiles-finetune-final.ckpt"))


def generate(cfg: DictConfig):
    """
    Generate SMILES strings from protein voxels using the end-to-end model.
    
    Args:
        cfg: Hydra configuration
    """
    # Load combined model
    combined_model = CombinedProteinToSmilesModel.load_from_checkpoint(cfg.generate.checkpoint_path)
    combined_model.eval()
    
    # Create data module
    data_module = hydra.utils.instantiate(cfg.data.poc2smiles_data)
    data_module.setup("test")
    
    # Create test loader
    test_loader = data_module.test_dataloader()
    
    # Generate SMILES strings
    all_generated_smiles = []
    all_true_smiles = []
    
    for batch in test_loader:
        protein_voxels = batch["protein"]
        true_smiles = batch.get("smiles_str", [])
        
        # Generate SMILES strings
        with torch.no_grad():
            output = combined_model(protein_voxels)
            ligand_voxels = output["ligand_voxels"]
            generated_smiles = combined_model.vox2smiles_model.generate_smiles(ligand_voxels)
        
        all_generated_smiles.extend(generated_smiles)
        all_true_smiles.extend(true_smiles)
    
    # Calculate metrics
    metrics = calculate_metrics(all_generated_smiles, all_true_smiles)
    
    # Print metrics
    print("Generation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Save generated SMILES
    os.makedirs(cfg.generate.output_dir, exist_ok=True)
    with open(os.path.join(cfg.generate.output_dir, "generated_smiles.txt"), "w") as f:
        for smiles in all_generated_smiles:
            f.write(f"{smiles}\n")
    
    # Save true SMILES
    with open(os.path.join(cfg.generate.output_dir, "true_smiles.txt"), "w") as f:
        for smiles in all_true_smiles:
            f.write(f"{smiles}\n")
    
    # Save metrics
    with open(os.path.join(cfg.generate.output_dir, "metrics.txt"), "w") as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")


if __name__ == "__main__":
    main() 