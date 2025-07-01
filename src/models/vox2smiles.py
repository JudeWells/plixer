import torch
import numpy as np
import copy
from lightning import LightningModule
from torchmetrics import MeanMetric
from transformers import (VisionEncoderDecoderModel,
                          VisionEncoderDecoderConfig,
                          ViTConfig,
                          GPT2Config,
                          get_scheduler,
                          SchedulerType)
import wandb
from rdkit import Chem
from rdkit.Chem import Draw
import io
from PIL import Image, ImageDraw, ImageFont
from src.models.modeling_vit_3d import ViTModel3D
from src.data.common.tokenizers.smiles_tokenizer import build_smiles_tokenizer
from src.utils.metrics import accuracy_from_outputs, calculate_validity, calculate_novelty, calculate_uniqueness


class VoxToSmilesModel(LightningModule):
    def __init__(
        self,
        config,
        override_optimizer_on_load: bool = False,
        visualise_val: bool = True,
        n_samples_for_validity_testing: int = 20,
    ) -> None:
        super().__init__()
        if "torch_dtype" not in config:
            config.torch_dtype = torch.bfloat16
        self.save_hyperparameters(logger=False)

        self.tokenizer = build_smiles_tokenizer()

        vit_config = ViTConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            layer_norm_eps=config.layer_norm_eps,
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            qkv_bias=config.qkv_bias,
            encoder_stride=config.encoder_stride,
            torch_dtype=config.torch_dtype,
        )

        gpt2_config = GPT2Config(
            bos_token_id=self.tokenizer.cls_token_id,
            eos_token_id=self.tokenizer.sep_token_id,
            vocab_size=len(self.tokenizer),
            pad_token_id=self.tokenizer.pad_token_id
        )

        encoder = ViTModel3D(vit_config)
        
        encoder_decoder_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=vit_config,
            decoder_config=gpt2_config,
            decoder_start_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        self.model = VisionEncoderDecoderModel(config=encoder_decoder_config, encoder=encoder)
        if vit_config.torch_dtype is not None:
            if not isinstance(vit_config.torch_dtype, torch.dtype):
                raise ValueError(f"Unsupported torch_dtype: {vit_config.torch_dtype}")
            else:
                self.model = self.model.to(vit_config.torch_dtype)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_acc = MeanMetric()
        self.val_loss_poc2mol_output = MeanMetric()
        self.val_acc_poc2mol_output = MeanMetric()
        self.test_loss = MeanMetric()
        self.test_acc = MeanMetric()

        self.override_optimizer_on_load = override_optimizer_on_load
        self.train_sample_counter = 0
        self.train_poc2mol_sample_counter = 0
        self.val_validity = MeanMetric()
        self.val_validity_poc2mol_output = MeanMetric()
        self.visualise_val = visualise_val
        self.n_samples_for_validity_testing = n_samples_for_validity_testing

    def forward(self, pixel_values, labels=None):
        if labels is None:
            raise ValueError("Labels are required for Vox2Smiles forward method.")

        decoder_attention_mask = (labels != self.tokenizer.pad_token_id).long()

        masked_labels = labels.clone()
        masked_labels[masked_labels == self.tokenizer.pad_token_id] = -100

        return self.model(
            pixel_values=pixel_values,
            labels=masked_labels,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["input_ids"]
        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/batch_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.train_sample_counter += len(batch['pixel_values'])
        self.log("train/n_samples", self.train_sample_counter, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch['pixel_values']))
        if 'poc2mol_loss' in batch and batch['poc2mol_loss'] is not None:
            elements_with_loss_mask = batch['poc2mol_loss'] > 0
            self.train_poc2mol_sample_counter += elements_with_loss_mask.int().sum()
            self.log("train/proportion_from_poc2mol", self.train_poc2mol_sample_counter / self.train_sample_counter, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch['pixel_values']))
            masked_labels = labels.clone()
            masked_labels[masked_labels == self.tokenizer.pad_token_id] = -100
            accuracy = accuracy_from_outputs(outputs, masked_labels, start_ix=1, ignore_index=-100)
            if elements_with_loss_mask.int().sum() == 0:
                self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(labels))
            elif elements_with_loss_mask.int().sum() == len(labels):
                self.log("train/poc2mol_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(labels))
            self.log("train/n_poc2mol_samples", self.train_poc2mol_sample_counter, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch['pixel_values']))
            if elements_with_loss_mask.any():
                self.log("train/poc2mol_loss", batch['poc2mol_loss'][elements_with_loss_mask].mean(), on_step=True, on_epoch=True, prog_bar=True, batch_size=elements_with_loss_mask.int().sum())
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pixel_values = batch["pixel_values"]
        labels = batch["input_ids"]
        masked_labels = labels.clone()
        masked_labels[masked_labels == self.tokenizer.pad_token_id] = -100
        outputs = self(pixel_values, labels=masked_labels)
        loss = outputs.loss
        accuracy = accuracy_from_outputs(outputs, masked_labels, start_ix=1, ignore_index=-100)
        if dataloader_idx == 0:
            self.val_loss(loss)
            self.log(f"val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            self.val_acc(accuracy)
            self.log(f"val/accuracy", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        else:
            self.val_loss_poc2mol_output(loss)
            self.log(f"val/poc2mol_output/loss", self.val_loss_poc2mol_output, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            self.val_acc_poc2mol_output(accuracy)
            self.log(f"val/poc2mol_output/accuracy", self.val_acc_poc2mol_output, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        if batch_idx == 0 or batch_idx * len(batch['pixel_values']) < self.n_samples_for_validity_testing:
            generated_smiles = self.generate_smiles(pixel_values[:self.n_samples_for_validity_testing], max_attempts=1)
            if len(generated_smiles) > 0:
                validity = calculate_validity(generated_smiles)
                if dataloader_idx == 0:
                    logging_key = "val/validity"
                    self.val_validity(validity)

                else:
                    logging_key = "val/poc2mol_output/validity"
                    self.val_validity_poc2mol_output(validity)
                self.log(
                    logging_key,
                    self.val_validity_poc2mol_output,
                    on_step=False,
                    on_epoch=True,
                    add_dataloader_idx=False,
                    batch_size=len(generated_smiles)
                )
        if batch_idx < 3 and self.visualise_val:
            try:
                if dataloader_idx == 0:
                    sample_str = ""
                else:
                    sample_str = "poc2mol_output "
                self.visualize_smiles(batch, sample_str=sample_str)
            except Exception as e:
                print("Error visualizing smiles: ", e)
        return loss

    def test_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["input_ids"]
        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        masked_labels = labels.clone()
        masked_labels[masked_labels == self.tokenizer.pad_token_id] = -100
        accuracy = accuracy_from_outputs(outputs, masked_labels, start_ix=1, ignore_index=-100)
        self.test_acc(accuracy)
        self.log("test/accuracy", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.config.lr)
        
        # Get scheduler configuration from config
        scheduler_config = getattr(self.hparams.config, "scheduler", {})
        scheduler_type = scheduler_config.get("type", "step")
        
        if scheduler_type == "step":
            # Default StepLR scheduler
            step_size = scheduler_config.get("step_size", 100)
            gamma = scheduler_config.get("gamma", 0.997)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            # Use transformers' get_scheduler for other scheduler types
            num_training_steps = scheduler_config.get("num_training_steps", self.trainer.estimated_stepping_batches)
            num_warmup_steps = scheduler_config.get("num_warmup_steps", 0)
            
            if isinstance(num_warmup_steps, float) and 0 <= num_warmup_steps < 1:
                # If warmup_steps is a fraction, calculate the absolute number
                num_warmup_steps = int(num_training_steps * num_warmup_steps)
            
            scheduler_specific_kwargs = {}
            
            # Handle specific scheduler parameters
            if scheduler_type == "cosine_with_restarts":
                scheduler_specific_kwargs["num_cycles"] = scheduler_config.get("num_cycles", 1)
            elif scheduler_type == "cosine_with_min_lr":
                scheduler_specific_kwargs["min_lr_rate"] = scheduler_config.get("min_lr_rate", 0.1)
            elif scheduler_type == "warmup_stable_decay":
                scheduler_specific_kwargs["num_stable_steps"] = scheduler_config["num_stable_steps"]
                scheduler_specific_kwargs["num_decay_steps"] = scheduler_config["num_decay_steps"]
                scheduler_specific_kwargs["min_lr_ratio"] = scheduler_config.get("min_lr_ratio", 0.1)
            
            scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=scheduler_specific_kwargs
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
    

    def is_valid_smiles(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return True
            else:
                return False
        except:
            return False
    
    def repetition_count(self, smiles):
        max_reps = 0
        current_reps = 0
        for sm in smiles:
            sm = sm.replace(' ', '')
            for i in range(len(sm)):
                for j in range(i+1, len(sm)):
                    if sm[i] == sm[j]:
                        current_reps += 1
                        max_reps = max(max_reps, current_reps)
                    else:
                        current_reps = 0
        return max_reps

    def generate_smiles(
            self, 
            pixel_values, 
            max_length=200, 
            max_attempts=6, 
            max_token_repeats=10, 
            do_sample=False, 
            temperature=1.0
        ):
        assert max_attempts > 0, "max_attempts must be greater than 0"
        batch_size = pixel_values.shape[0]
        results = [None] * batch_size
        best_results = [None] * batch_size
        min_repetition_scores = [float('inf')] * batch_size
        need_generation = [True] * batch_size
        attempts = [0] * batch_size
        
        while any(need_generation) and max(attempts) < max_attempts:
            indices_to_generate = [i for i, need_gen in enumerate(need_generation) if need_gen]
            if len(indices_to_generate) < batch_size:
                current_pixel_values = pixel_values[indices_to_generate]
            else:
                current_pixel_values = pixel_values
            
            current_do_sample = do_sample or max(attempts) > 0  # Use sampling after first attempt
            tokens = self.model.generate(current_pixel_values, max_length=max_length, do_sample=current_do_sample, temperature=temperature)
            predicted_smiles = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
            predicted_smiles = [sm.replace(' ', '') for sm in predicted_smiles]
            
            for idx, gen_idx in enumerate(indices_to_generate):
                attempts[gen_idx] += 1
                current_smiles = predicted_smiles[idx]
                
                # Track this attempt if it's a valid SMILES string
                if len(current_smiles) > 0 and self.is_valid_smiles(current_smiles):
                    # Check repetition score
                    rep_count = self.repetition_count([current_smiles])
                    
                    # Update best result if this has a lower repetition score
                    if rep_count < min_repetition_scores[gen_idx]:
                        best_results[gen_idx] = current_smiles
                        min_repetition_scores[gen_idx] = rep_count
                    
                    # If under threshold, consider this a success
                    if rep_count <= max_token_repeats:
                        results[gen_idx] = current_smiles
                        need_generation[gen_idx] = False
                    

                else:
                    if best_results[gen_idx] is None:
                        best_results[gen_idx] = current_smiles
    
        for i in range(batch_size):
            if results[i] is None:
                results[i] = best_results[i]
        
        return results

    def visualize_smiles(self, batch, sample_str=""):
        actual_smiles = batch["smiles_str"]
        actual_smiles = [sm.replace("[BOS]", '').replace("[EOS]", "") for sm in actual_smiles]
        predicted_smiles = self.generate_smiles(batch["pixel_values"])

        images_to_log = []

        for i, (pred, actual) in enumerate(zip(predicted_smiles, actual_smiles)):
            pred_img = self.smiles_to_image(pred, f"Predicted: {pred}")
            if pred_img:
                images_to_log.append(wandb.Image(pred_img, caption=f"Sample {i} {sample_str}Predicted"))

            actual_img = self.smiles_to_image(actual, f"Actual: {actual}")
            if actual_img:
                images_to_log.append(wandb.Image(actual_img, caption=f"Sample {i} {sample_str}Actual"))

        if images_to_log:
            wandb.log({"SMILES Comparison": images_to_log})

    def smiles_to_image(self, smiles, label):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                img = Draw.MolToImage(mol, size=(300, 300))

                # Add SMILES string as text to the image
                draw = ImageDraw.Draw(img)
                font = ImageFont.load_default()
                draw.text((10, 0), label, font=font, fill=(0, 0, 0))

                return img  # Return PIL Image object directly
            else:
                return None
        except:
            return None