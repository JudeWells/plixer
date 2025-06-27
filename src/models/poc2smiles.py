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
import warnings

from src.models.poc2mol import Poc2Mol
from src.models.vox2smiles import VoxToSmilesModel
from src.utils.metrics import calculate_validity, calculate_novelty, calculate_uniqueness, accuracy_from_outputs


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
    
    def forward(
            self, 
            protein_voxels, 
            labels=None, 
            decoy_labels=None, 
            ligand_voxels=None, 
            sample_smiles=True,
            predicted_ligand_voxels=None,
            temperature=1.0,
            ):
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
        if predicted_ligand_voxels is None:
            poc2mol_output = self.poc2mol_model(protein_voxels, labels=ligand_voxels)
        if isinstance(poc2mol_output, dict):
            predicted_ligand_voxels = poc2mol_output['pred_vox']
        else:
            predicted_ligand_voxels = poc2mol_output
        predicted_ligand_voxels = torch.sigmoid(predicted_ligand_voxels)
        result = {
            "predicted_ligand_voxels": predicted_ligand_voxels,
        }
        if labels is not None:
            masked_labels = labels.clone()
            masked_labels[masked_labels == self.vox2smiles_model.tokenizer.pad_token_id] = -100
            vox2smiles_output = self.vox2smiles_model(predicted_ligand_voxels, labels=masked_labels)
            result['logits'] = vox2smiles_output["logits"]
            result['loss'] = vox2smiles_output['loss']
            result['poc2mol_bce'] = poc2mol_output['bce'].mean().item()
            result['poc2mol_dice'] = poc2mol_output['dice'].mean().item()
            result['poc2mol_loss'] = result['poc2mol_bce'] + result['poc2mol_dice']
            result['smiles_teacher_forced_accuracy'] = accuracy_from_outputs(
                vox2smiles_output, masked_labels, start_ix=1, ignore_index=-100
            )
            if decoy_labels is not None:  # todo remove this bloc
                masked_decoy_labels = decoy_labels.clone()
                masked_decoy_labels[masked_decoy_labels == self.vox2smiles_model.tokenizer.pad_token_id] = -100
                vox2smiles_output_decoy_loss = self.vox2smiles_model(predicted_ligand_voxels, labels=decoy_labels)['loss']
                result['decoy_loss'] = vox2smiles_output_decoy_loss
                result['decoy_smiles_true_label_teacher_forced_accuracy'] = accuracy_from_outputs(
                    vox2smiles_output, masked_decoy_labels, start_ix=1, ignore_index=-100
                )
                vox2smiles_decoy_output = self.vox2smiles_model(predicted_ligand_voxels, labels=decoy_labels)
                result['decoy_smiles_decoy_label_teacher_forced_accuracy'] = accuracy_from_outputs(
                    vox2smiles_decoy_output, masked_decoy_labels, start_ix=1, ignore_index=-100
                )
    
        if sample_smiles:
            sampled_smiles = self.vox2smiles_model.generate_smiles(predicted_ligand_voxels, temperature=temperature)
        else:
            sampled_smiles = None
        result['sampled_smiles'] = sampled_smiles

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

    def generate_smiles_from_protein_voxels(self, protein_voxels, num_samples=1):
        """
        Generates SMILES strings from protein voxels.
        """
        poc2mol_output = self.poc2mol_model(protein_voxels)
        if isinstance(poc2mol_output, dict):
            pred_vox = poc2mol_output['pred_vox']
        else:
            pred_vox = poc2mol_output
        pred_vox = torch.sigmoid(pred_vox)
        
        if num_samples > 1:
            # Repeat tensor for multiple samples
            pred_vox = pred_vox.repeat_interleave(num_samples, dim=0)

        generated_smiles = self.vox2smiles_model.generate_smiles(pred_vox)
        return generated_smiles

    def score_smiles(self, protein_voxels, smiles_list, tokenizer, max_smiles_len, batch_size: int = 250):
        """
        Scores a list of SMILES strings against a protein structure.
        Returns a list of log-likelihood scores.
        """
        # 1. Get predicted ligand voxels. This is done only once.
        self.poc2mol_model.eval()
        self.vox2smiles_model.eval()
        poc2mol_output = self.poc2mol_model(protein_voxels)
        if isinstance(poc2mol_output, dict):
            pred_vox = poc2mol_output['pred_vox']
        else:
            pred_vox = poc2mol_output
        pred_vox = torch.sigmoid(pred_vox)

        # 2. Canonicalize SMILES strings when possible
        processed_smiles = []
        for s in smiles_list:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    raise ValueError("RDKit failed to parse SMILES")
                canonical = Chem.MolToSmiles(mol, canonical=True)
                print("Original:", s, "Canonical:", canonical)
            except Exception as e:
                warnings.warn(
                    f"Could not canonicalize SMILES '{s}'. Using original string. Error: {str(e)}"
                )
                canonical = s
            processed_smiles.append(canonical)

        scores = []

        # Fallback to original (loss-based) method for batch_size == 1. This avoids
        # unnecessary overhead and guarantees identical behaviour to the legacy
        # implementation, which can be useful for detailed debugging or single-SMILES
        # evaluations.
        if batch_size == 1:
            for smi in processed_smiles:
                # ensure BOS/EOS tokens
                if not smi.startswith(tokenizer.bos_token):
                    smi_tok = tokenizer.bos_token + smi
                else:
                    smi_tok = smi
                if not smi_tok.endswith(tokenizer.eos_token):
                    smi_tok = smi_tok + tokenizer.eos_token

                tokenized_single = tokenizer(
                    [smi_tok],
                    padding='max_length',
                    max_length=max_smiles_len,
                    truncation=True,
                    return_tensors="pt",
                )

                labels_single = tokenized_single['input_ids'].to(self.device)

                vox2smiles_output = self.vox2smiles_model(pred_vox, labels=labels_single)
                loss_val = vox2smiles_output['loss'].item()
                scores.append(-loss_val)

            return scores

        # 3. Iterate through SMILES in batches (modern vectorised method)
        for start_idx in range(0, len(processed_smiles), batch_size):
            batch_smiles = processed_smiles[start_idx : start_idx + batch_size]

            # Add BOS/EOS tokens
            batch_smiles_with_tokens = []
            for smi in batch_smiles:
                if not smi.startswith(tokenizer.bos_token):
                    smi = tokenizer.bos_token + smi
                if not smi.endswith(tokenizer.eos_token):
                    smi = smi + tokenizer.eos_token
                batch_smiles_with_tokens.append(smi)
            batch_max_length = max(len(smi) for smi in batch_smiles_with_tokens)
            tokenized = tokenizer(
                batch_smiles_with_tokens,
                padding='max_length',
                max_length=batch_max_length,
                truncation=True,
                return_tensors="pt",
            )

            labels = tokenized['input_ids'].to(self.device)

            # Repeat predicted voxel for each SMILES in the batch
            pred_vox_batch = pred_vox.repeat(labels.size(0), 1, 1, 1, 1)

            # Forward pass through Vox2Smiles model to obtain logits
            vox2smiles_output = self.vox2smiles_model(pred_vox_batch, labels=labels)
            logits = vox2smiles_output.logits  # (B, seq_len, vocab)

            # Compute per-sample average log-likelihood
            vocab_size = logits.size(-1)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            masked_labels = labels.clone()
            pad_token_id = self.vox2smiles_model.tokenizer.pad_token_id
            masked_labels[masked_labels == pad_token_id] = -100

            # Gather log-probs corresponding to the true tokens
            gather_idx = masked_labels.clone()
            gather_idx[gather_idx == -100] = 0  # placeholder index (will be masked later)
            token_log_probs = log_probs.gather(-1, gather_idx.unsqueeze(-1)).squeeze(-1)

            # Zero-out padding positions
            valid_mask = (masked_labels != -100)
            token_log_probs = token_log_probs * valid_mask

            # Compute mean log-likelihood per sample (same scale as −CE)
            seq_lengths = valid_mask.sum(dim=1)
            # To avoid division by zero (shouldn't happen) we clamp minimum to 1
            seq_lengths = seq_lengths.clamp(min=1)
            sample_log_likelihood = token_log_probs.sum(dim=1) / seq_lengths

            scores.extend(sample_log_likelihood.detach().cpu().tolist())

        return scores