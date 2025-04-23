import torch
import numpy as np
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
        compile: bool = False,
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
        self.test_loss = MeanMetric()
        self.test_acc = MeanMetric()

        self.override_optimizer_on_load = override_optimizer_on_load
        self.train_sample_counter = 0
        self.train_poc2mol_sample_counter = 0
        self.val_validity = MeanMetric()
    def forward(self, pixel_values, labels=None):
        if labels is not None:
            return self.model(pixel_values=pixel_values, labels=labels, return_dict=True)
        else:
            return self.model(pixel_values=pixel_values, return_dict=False)

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["input_ids"]
        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/batch_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        if 'poc2mol_loss' in batch and batch['poc2mol_loss'] is not None:
            elements_with_loss_mask = batch['poc2mol_loss'] > 0
            self.train_poc2mol_sample_counter += elements_with_loss_mask.int().sum()
            self.log("train/proportion_from_poc2mol", self.train_poc2mol_sample_counter / self.train_sample_counter, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch['pixel_values']))
            self.log("train/n_samples", self.train_sample_counter, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch['pixel_values']))
            self.log("train/n_poc2mol_samples", self.train_poc2mol_sample_counter, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch['pixel_values']))
            if elements_with_loss_mask.any():
                self.log("train/poc2mol_loss", batch['poc2mol_loss'][elements_with_loss_mask].mean(), on_step=True, on_epoch=True, prog_bar=True, batch_size=elements_with_loss_mask.int().sum())
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["input_ids"]
        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        accuracy = accuracy_from_outputs(outputs, labels, start_ix=1, ignore_index=0)
        self.val_acc(accuracy)
        self.log("val/accuracy", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx % 200 == 0:
            generated_smiles = self.generate_smiles(pixel_values)
            if len(generated_smiles) > 0:
                validity = calculate_validity(generated_smiles)
                self.val_validity(validity)
                self.log("val/validity", self.val_validity, on_step=False, on_epoch=True)
        if batch_idx < 3:
            self.visualize_smiles(batch, outputs)
        return loss

    def test_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["input_ids"]
        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        accuracy = accuracy_from_outputs(outputs, labels, start_ix=1, ignore_index=0)
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
    
    def generate_smiles(self, pixel_values, max_length=200):
        tokens = self.model.generate(pixel_values, max_length=max_length)
        predicted_smiles = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        return [sm.replace(' ', '') for sm in predicted_smiles]

    def visualize_smiles(self, batch, outputs):
        actual_smiles = batch["smiles_str"]
        actual_smiles = [sm.replace("[BOS]", '').replace("[EOS]", "") for sm in actual_smiles]
        predicted_smiles = self.generate_smiles(batch["pixel_values"])

        images_to_log = []

        for i, (pred, actual) in enumerate(zip(predicted_smiles, actual_smiles)):
            pred_img = self.smiles_to_image(pred, f"Predicted: {pred}")
            if pred_img:
                images_to_log.append(wandb.Image(pred_img, caption=f"Sample {i} - Predicted"))

            actual_img = self.smiles_to_image(actual, f"Actual: {actual}")
            if actual_img:
                images_to_log.append(wandb.Image(actual_img, caption=f"Sample {i} - Actual"))

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