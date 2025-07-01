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
from sklearn.metrics import roc_auc_score

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
            override_optimizer_on_load: bool = False,
            decoy_smiles_list: list = None,
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
        # Decoy SMILES list for validation-time rescoring
        if isinstance(decoy_smiles_list, str) and os.path.exists(decoy_smiles_list):
            # Assume CSV with `smiles` column
            try:
                import pandas as _pd
                self.decoy_smiles_list = _pd.read_csv(decoy_smiles_list).smiles.tolist()
            except Exception as _e:
                raise ValueError(f"Could not load decoy SMILES from '{decoy_smiles_list}': {_e}")
        else:
            self.decoy_smiles_list = decoy_smiles_list
    
    def forward(
        self,
        protein_voxels: torch.Tensor = None,
        labels: torch.Tensor = None,
        ligand_voxels: torch.Tensor = None,
        sample_smiles: bool = True,
        temperature: float = 1.0,
    ):
        """End-to-end forward pass.

        Exactly one source of ligand voxels must ultimately be provided:
        1. *ligand_voxels* – explicit ground-truth voxels (e.g. during
           supervised training).
        2. *protein_voxels* – if *ligand_voxels* are **not** given we will run
           *poc2mol_model* on the provided protein voxels to obtain the
           predictions on-the-fly.

        The method is therefore flexible enough for both training and inference
        scenarios.  A dictionary with the following (important) keys is
        returned:
            ``predicted_ligand_voxels`` – Sigmoid-activated ligand voxels fed
            into the Vox2Smiles model.
            ``logits`` / ``loss`` (optional) – Outputs of the Vox2Smiles model
            when *labels* are supplied.
        """

        result = {}

        # ------------------------------------------------------------------
        # 1) Obtain (or use provided) ligand voxels
        # ------------------------------------------------------------------
        poc2mol_output = None  # for bookkeeping – may stay *None*

        assert ligand_voxels is not None or protein_voxels is not None, (
            "Either ligand_voxels or protein_voxels must be provided."
        )

        if ligand_voxels is not None:
            # User explicitly provided ground-truth ligand voxels – bypass
            # Poc2Mol for the forward pass but **optionally** compute the
            # Poc2Mol loss if protein voxels are also given.
            final_ligand_voxels = ligand_voxels
            if protein_voxels is not None:
                poc2mol_output = self.poc2mol_model(protein_voxels, labels=ligand_voxels)
        else:
            # Need to run Poc2Mol to obtain predictions.
            poc2mol_output = self.poc2mol_model(protein_voxels, labels=ligand_voxels)
            final_ligand_voxels = poc2mol_output["predicted_ligand_voxels"]

        # Store Poc2Mol outputs (if any) for downstream logging/analysis.
        if poc2mol_output is not None and isinstance(poc2mol_output, dict):
            result.update({
                "poc2mol_bce": poc2mol_output.get("bce", torch.tensor(0.0)).mean().item() if "bce" in poc2mol_output else None,
                "poc2mol_dice": poc2mol_output.get("dice", torch.tensor(0.0)).mean().item() if "dice" in poc2mol_output else None,
                "poc2mol_loss": poc2mol_output.get("loss", torch.tensor(0.0)).item() if "loss" in poc2mol_output else None,
            })

        # ------------------------------------------------------------------
        # 2) Vox2Smiles – teacher forcing when labels are provided
        # ------------------------------------------------------------------
        if labels is not None:
            vox2smiles_result = self.compute_smiles_metrics(
                ligand_voxels=final_ligand_voxels,
                labels=labels,
            )
            result.update(vox2smiles_result)

        # ------------------------------------------------------------------
        # 3) Autoregressive generation (optional)
        # ------------------------------------------------------------------
        if sample_smiles:
            sampled_smiles = self.vox2smiles_model.generate_smiles(final_ligand_voxels, temperature=temperature)
        else:
            sampled_smiles = None

        # ------------------------------------------------------------------
        # 4) Assemble final result dict
        # ------------------------------------------------------------------
        result.update({
            "predicted_ligand_voxels": final_ligand_voxels,
            # Legacy alias to avoid breaking older code
            "ligand_voxels": final_ligand_voxels,
            "sampled_smiles": sampled_smiles,
        })

        return result

    def compute_smiles_metrics(
        self,
        ligand_voxels: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute loss / logits / accuracy for given *ligand_voxels* and *labels*.

        This helper avoids re-running Poc2Mol when the ligand voxels are already
        available (e.g. during rescoring of many SMILES against the same
        pocket-specific ligand voxels).
        """
        result: dict = {}

        # Ensure tensors are on the same device
        device = ligand_voxels.device
        labels = labels.to(device)
        pad_id = self.vox2smiles_model.tokenizer.pad_token_id

        masked_labels = labels.clone()
        masked_labels[masked_labels == pad_id] = -100

        vox2smiles_output = self.vox2smiles_model(ligand_voxels, labels=masked_labels)

        result.update({
            "logits": vox2smiles_output["logits"],
            "loss": vox2smiles_output["loss"],
            "smiles_teacher_forced_accuracy": accuracy_from_outputs(
                vox2smiles_output,
                masked_labels,
                start_ix=1,
                ignore_index=-100,
            ),
        })


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
        
        # Forward pass (teacher-forcing with labels)
        output = self(
            protein_voxels,
            labels=smiles_tokens,
            ligand_voxels=batch.get("ligand_voxels", None),
            sample_smiles=False,
        )

        loss = output["loss"]
        
        # Backward pass and optimisation
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
        true_smiles = batch.get("smiles_strings", batch.get("smiles_str", []))
        
        # Forward pass with labels to obtain loss + logits
        output = self(
            protein_voxels,
            labels=smiles_tokens,
            ligand_voxels=batch.get("ligand_voxels", None),
            sample_smiles=False,
        )

        logits = output["logits"]
        ligand_voxels = output["predicted_ligand_voxels"]
        loss = output["loss"]

        # Generate SMILES (greedy)
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
        
        # --------------------------------------------------------------
        #  Decoy-based likelihood AUC ROC evaluation (optional)
        # --------------------------------------------------------------
        decoy_smiles_source = batch.get("decoy_smiles", getattr(self, "decoy_smiles_list", None))
        if decoy_smiles_source is not None and len(decoy_smiles_source) > 1 and len(true_smiles) > 0:
            for idx_in_batch, tgt_smi in enumerate(true_smiles):
                if tgt_smi is None or len(tgt_smi) == 0:
                    continue
                decoys = [s for s in decoy_smiles_source if s != tgt_smi]
                if len(decoys) == 0:
                    continue

                smiles_candidates = [tgt_smi] + decoys
                ll_scores = self._log_likelihoods_for_smiles_list(
                    ligand_voxels[idx_in_batch : idx_in_batch + 1],  # keep tensor dims
                    smiles_candidates,
                    batch_size=250,
                )

                labels_auc = [1] + [0] * len(decoys)
                try:
                    auc_val = roc_auc_score(labels_auc, ll_scores)
                    self.log(
                        "val/decoy_roc_auc",
                        auc_val,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                    )

                    sorted_indices = sorted(
                        range(len(ll_scores)), key=lambda i: ll_scores[i], reverse=True
                    )
                    rank_of_hit = sorted_indices.index(0)
                    self.log(
                        "val/hit_rank_among_decoys",
                        rank_of_hit,
                        on_step=False,
                        on_epoch=True,
                    )
                except ValueError:
                    pass
        
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
        pred_vox = poc2mol_output["predicted_ligand_voxels"]
        
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
        pred_vox = poc2mol_output["predicted_ligand_voxels"]

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

    def _log_likelihoods_for_smiles_list(
        self,
        ligand_voxels: torch.Tensor,
        smiles_list: list,
        batch_size: int = 250,
    ):
        """Compute average log-likelihood (−CE) for every SMILES in *smiles_list*.

        The method reuses the **provided** *ligand_voxels* tensor (shape 1×C×X×Y×Z) and
        therefore avoids additional Poc2Mol forward passes.  The voxels are repeated
        as needed to match the mini-batch size.
        """

        if len(smiles_list) == 0:
            return []

        device = ligand_voxels.device
        tokenizer = self.vox2smiles_model.tokenizer

        processed_smiles = []
        for smi in smiles_list:
            # Ensure BOS/EOS tokens
            if not smi.startswith(tokenizer.bos_token):
                smi_tok = tokenizer.bos_token + smi
            else:
                smi_tok = smi
            if not smi_tok.endswith(tokenizer.eos_token):
                smi_tok = smi_tok + tokenizer.eos_token
            processed_smiles.append(smi_tok)

        scores = []

        # Vectorised evaluation in chunks to avoid OOM
        for start in range(0, len(processed_smiles), batch_size):
            batch_smiles = processed_smiles[start : start + batch_size]
            max_len = max(len(s) for s in batch_smiles)
            tokenized = tokenizer(
                batch_smiles,
                padding='max_length',
                max_length=max_len,
                truncation=True,
                return_tensors='pt',
            )
            labels = tokenized['input_ids'].to(device)

            # Repeat voxels for every SMILES in this chunk
            vox_batch = ligand_voxels.repeat(labels.size(0), 1, 1, 1, 1)

            # Forward pass (teacher forced)
            outputs = self.vox2smiles_model(vox_batch, labels=labels)

            # Convert CE loss to log-likelihood per sample
            logits = outputs.logits  # (B,L,V)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            pad_id = tokenizer.pad_token_id
            masked_labels = labels.clone()
            masked_labels[masked_labels == pad_id] = -100

            gather_idx = masked_labels.clone()
            gather_idx[gather_idx == -100] = 0
            token_log_probs = log_probs.gather(-1, gather_idx.unsqueeze(-1)).squeeze(-1)

            # Mask padding
            valid_mask = masked_labels != -100
            token_log_probs = token_log_probs * valid_mask

            seq_lengths = valid_mask.sum(dim=1).clamp(min=1)
            sample_ll = token_log_probs.sum(dim=1) / seq_lengths  # average LL per token

            scores.extend(sample_ll.detach().cpu().tolist())

        return scores