import torch
import numpy as np
from lightning import LightningModule
from torchmetrics import MeanMetric
from transformers import (VisionEncoderDecoderModel,
                          VisionEncoderDecoderConfig,
                          ViTConfig,
                          GPT2Config)
import wandb
from rdkit import Chem
from rdkit.Chem import Draw
import io
from PIL import Image, ImageDraw, ImageFont
from src.models.modeling_vit_3d import ViTModel3D
from src.data.tokenizers.smiles_tokenizer import build_smiles_tokenizer
from src.utils.metrics import accuracy_from_outputs


class VoxToSmilesModel(LightningModule):
    def __init__(
        self,
        config,
        compile: bool = False,
    ) -> None:
        super().__init__()
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

        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_acc = MeanMetric()
        self.test_loss = MeanMetric()
        self.test_acc = MeanMetric()

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["input_ids"]
        outputs = self(pixel_values, labels=labels)
        loss = outputs.loss
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/batch_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def visualize_smiles(self, batch, outputs):
        actual_smiles = batch["smiles_str"]
        actual_smiles = [sm.replace("[BOS]", '').replace("[EOS]", "") for sm in actual_smiles]

        # Generate predictions
        gen_output = self.model.generate(batch["pixel_values"], max_length=200)

        # Decode predictions
        predicted_smiles = self.tokenizer.batch_decode(gen_output, skip_special_tokens=True)
        predicted_smiles = [sm.replace(' ', '') for sm in predicted_smiles]

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