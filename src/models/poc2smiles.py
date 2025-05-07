import os
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torch.optim.lr_scheduler import StepLR
from transformers.optimization import get_scheduler
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from src.models.poc2mol import Poc2Mol
from src.models.vox2smiles import VoxToSmilesModel
from src.utils.metrics import calculate_validity, calculate_novelty, calculate_uniqueness


class CombinedProteinToSmilesModel(L.LightningModule):
    """
    Combined model that takes protein voxels as input and outputs SMILES strings.
    It uses Poc2Mol to generate ligand voxels from protein voxels, and then
    Vox2Smiles to generate SMILES strings from ligand voxels.
    """
    
    def __init__(
            self, 
            poc2mol_model, 
            vox2smiles_model, 
            config, 
            override_optimizer_on_load: bool = False
        ):
        """
        Initialize the combined model.
        
        Args:
            poc2mol_model: Poc2Mol model
            vox2smiles_model: Vox2Smiles model
            config: Configuration dictionary
        """
        super().__init__()
        self.poc2mol_model = poc2mol_model
        self.vox2smiles_model = vox2smiles_model
        self.config = config
        self.save_hyperparameters(ignore=["poc2mol_model", "vox2smiles_model"])
        
        # Disable automatic optimization to handle the two models separately
        self.automatic_optimization = False
        
        # Create directory for saving images
        if self.config.get("img_save_dir"):
            os.makedirs(self.config["img_save_dir"], exist_ok=True)
        
        self.override_optimizer_on_load = override_optimizer_on_load
    
    def forward(self, protein_voxels, labels=None, decoy_labels=None, ligand_voxels=None):
        """
        Forward pass through the combined model.
        
        Args:
            protein_voxels: Protein voxels [batch_size, channels, x, y, z]
            
        Returns:
            Dictionary containing:
                - logits: Logits for SMILES tokens [batch_size, seq_len, vocab_size]
                - ligand_voxels: Generated ligand voxels [batch_size, channels, x, y, z]
        """
        # Generate ligand voxels from protein voxels
        poc2mol_output = self.poc2mol_model(protein_voxels, labels=ligand_voxels)
        if isinstance(poc2mol_output, dict):
            pred_vox = poc2mol_output['pred_vox']
        else:
            pred_vox = poc2mol_output
        pred_vox = torch.sigmoid(pred_vox)
        
        vox2smiles_output = self.vox2smiles_model(pred_vox, labels=labels)
        sampled_smiles = self.vox2smiles_model.generate_smiles(pred_vox)
        result = {
            "logits": vox2smiles_output["logits"],
            "predicted_ligand_voxels": pred_vox,
            "sampled_smiles": sampled_smiles,
            "loss": vox2smiles_output['loss']
        }
        if labels is not None:
            result['poc2mol_bce'] = poc2mol_output['bce'].mean().item()
            result['poc2mol_dice'] = poc2mol_output['dice'].mean().item()
            result['poc2mol_loss'] = result['poc2mol_bce'] + result['poc2mol_dice']
        if decoy_labels is not None:
            vox2smiles_output_decoy_loss = self.vox2smiles_model(pred_vox, labels=decoy_labels)['loss']
            result['decoy_loss'] = vox2smiles_output_decoy_loss

        return result
    
    def training_step(self, batch, batch_idx):
        """
        Training step for the combined model.
        
        Args:
            batch: Batch containing protein voxels, ligand voxels, and SMILES tokens
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        # Get optimizers

        opt = self.optimizers()
        
        # Get protein voxels and SMILES tokens
        protein_voxels = batch["protein_voxels"]
        smiles_tokens = batch["smiles_tokens"]
        
        # Forward pass
        output = self(protein_voxels)
        logits = output["logits"]
        
        # Calculate loss
        loss = self.vox2smiles_model.calculate_loss(logits, smiles_tokens)
        
        # Backward pass and optimization
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        
        # Update learning rate
        self.lr_schedulers().step()
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the combined model.
        
        Args:
            batch: Batch containing protein voxels, ligand voxels, and SMILES tokens
            batch_idx: Batch index
            
        Returns:
            Dictionary containing loss and generated SMILES
        """
        # Get protein voxels and SMILES tokens
        protein_voxels = batch["protein_voxels"]
        smiles_tokens = batch["smiles_tokens"]
        true_smiles = batch.get("smiles_strings", [])
        
        # Forward pass
        output = self(protein_voxels)
        logits = output["logits"]
        ligand_voxels = output["ligand_voxels"]
        
        # Calculate loss
        loss = self.vox2smiles_model.calculate_loss(logits, smiles_tokens)
        
        # Generate SMILES
        generated_smiles = self.vox2smiles_model.generate_smiles(ligand_voxels)
        
        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Save some examples
        if batch_idx == 0 and self.config.get("img_save_dir"):
            self._save_examples(protein_voxels, ligand_voxels, true_smiles, generated_smiles)
        
        # Calculate chemical metrics
        if len(generated_smiles) > 0:
            validity = calculate_validity(generated_smiles)
            self.log("val/validity", validity, on_step=False, on_epoch=True)
            
            if len(true_smiles) > 0:
                novelty = calculate_novelty(generated_smiles, true_smiles)
                uniqueness = calculate_uniqueness(generated_smiles)
                self.log("val/novelty", novelty, on_step=False, on_epoch=True)
                self.log("val/uniqueness", uniqueness, on_step=False, on_epoch=True)
        
        return {
            "loss": loss,
            "generated_smiles": generated_smiles,
            "true_smiles": true_smiles,
        }
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary containing optimizer and scheduler
        """
        # Create optimizer
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.get("lr", 1e-4),
            weight_decay=self.config.get("weight_decay", 0.0),
        )

        # Scheduler configuration (nested style)
        scheduler_config = self.config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "step")

        if scheduler_type == "step":
            # Fallback to legacy root‑level step_size/gamma if not provided inside scheduler block
            step_size = scheduler_config.get("step_size", self.config.get("step_size", 100))
            gamma = scheduler_config.get("gamma", self.config.get("gamma", 0.99))
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            # Use transformers' schedulers for advanced cases
            num_training_steps = self.trainer.estimated_stepping_batches
            num_warmup_steps = scheduler_config.get("num_warmup_steps", 0)

            if isinstance(num_warmup_steps, float) and 0 <= num_warmup_steps < 1:
                num_warmup_steps = int(num_training_steps * num_warmup_steps)

            scheduler_specific_kwargs = {}

            if scheduler_type == "cosine_with_restarts":
                scheduler_specific_kwargs["num_cycles"] = scheduler_config.get("num_cycles", 1)
            elif scheduler_type == "cosine_with_min_lr":
                scheduler_specific_kwargs["min_lr_rate"] = scheduler_config.get("min_lr_rate", 0.1)

            scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=scheduler_specific_kwargs,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": scheduler_config.get("interval", "step"),
                    "frequency": scheduler_config.get("frequency", 1),
                },
            }

    def on_load_checkpoint(self, checkpoint):
        """Handle checkpoint loading, optionally overriding optimizer and scheduler states.

        If override_optimizer_on_load is True, we'll remove the optimizer and
        lr_scheduler states from the checkpoint, forcing Lightning to create new ones
        based on the current config hyperparameters.
        """
        if self.override_optimizer_on_load:
            if "optimizer_states" in checkpoint:
                print(
                    "Overriding optimizer state from checkpoint with current config values"
                )
                del checkpoint["optimizer_states"]

            if "lr_schedulers" in checkpoint:
                print(
                    "Overriding lr scheduler state from checkpoint with current config values"
                )
                del checkpoint["lr_schedulers"]

            # Set a flag to tell Lightning not to expect optimizer states
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []



    def _save_examples(self, protein_voxels, ligand_voxels, true_smiles, generated_smiles, num_examples=4):
        """
        Save examples of generated molecules.
        
        Args:
            protein_voxels: Protein voxels [batch_size, channels, x, y, z]
            ligand_voxels: Generated ligand voxels [batch_size, channels, x, y, z]
            true_smiles: True SMILES strings
            generated_smiles: Generated SMILES strings
            num_examples: Number of examples to save
        """
        num_examples = min(num_examples, len(generated_smiles))
        
        for i in range(num_examples):
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot protein voxels
            protein_vox = protein_voxels[i, 0].detach().cpu().numpy()
            axes[0].imshow(np.max(protein_vox, axis=0), cmap="viridis")
            axes[0].set_title("Protein Voxels")
            axes[0].axis("off")
            
            # Plot ligand voxels
            ligand_vox = ligand_voxels[i, 0].detach().cpu().numpy()
            axes[1].imshow(np.max(ligand_vox, axis=0), cmap="viridis")
            axes[1].set_title("Generated Ligand Voxels")
            axes[1].axis("off")
            
            # Plot molecule
            try:
                mol = Chem.MolFromSmiles(generated_smiles[i])
                if mol is not None:
                    AllChem.Compute2DCoords(mol)
                    img = Draw.MolToImage(mol, size=(300, 300))
                    axes[2].imshow(img)
                    axes[2].set_title(f"Generated: {generated_smiles[i]}")
                else:
                    axes[2].text(0.5, 0.5, f"Invalid SMILES: {generated_smiles[i]}", 
                                ha="center", va="center")
            except Exception as e:
                axes[2].text(0.5, 0.5, f"Error: {str(e)}", ha="center", va="center")
            
            axes[2].axis("off")
            
            # Add true SMILES if available
            if i < len(true_smiles):
                fig.suptitle(f"True SMILES: {true_smiles[i]}")
            
            # Save figure
            epoch = self.current_epoch
            plt.tight_layout()
            plt.savefig(os.path.join(self.config["img_save_dir"], f"example_{epoch}_{i}.png"))
            plt.close(fig)