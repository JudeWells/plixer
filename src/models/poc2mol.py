from typing import Any, Dict, Optional

import torch
import numpy as np
from lightning import LightningModule

from src.models.pytorch3dunet import ResidualUNetSE3D
from src.models.pytorch3dunet_lib.unet3d.buildingblocks import ResNetBlockSE, ResNetBlock
from transformers.optimization import get_scheduler

class Poc2MolConfig:
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        final_sigmoid: bool = True,
        f_maps: int = 64,
        layer_order: str = 'gcr',
        num_groups: int = 8,
        num_levels: int = 5,
        is_segmentation: bool = True,
        conv_padding: int = 1,
        conv_upscale: int = 2,
        upsample: str = 'default',
        dropout_prob: float = 0.1,
        basic_module: ResNetBlock = ResNetBlockSE,

    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_sigmoid = final_sigmoid
        self.f_maps = f_maps
        self.layer_order = layer_order
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.is_segmentation = is_segmentation
        self.conv_padding = conv_padding
        self.conv_upscale = conv_upscale
        self.upsample = upsample
        self.dropout_prob = dropout_prob
        self.basic_module = basic_module





class Poc2Mol(LightningModule):
    def __init__(
        self,
        config: Poc2MolConfig,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        scheduler_name: str = None,
        num_warmup_steps: int = 0,
        num_training_steps: Optional[int] = 100000,
        num_decay_steps: int = 0,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = ResidualUNetSE3D(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            final_sigmoid=config.final_sigmoid,
            f_maps=config.f_maps,
            layer_order=config.layer_order,
            num_groups=config.num_groups,
            num_levels=config.num_levels,
            is_segmentation=config.is_segmentation,
            conv_padding=config.conv_padding,
            conv_upscale=config.conv_upscale,
            upsample=config.upsample,
            dropout_prob=config.dropout_prob,
            basic_module=config.basic_module,
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name
        self.num_warmup_steps = num_warmup_steps
        self.num_decay_steps = num_decay_steps
        self.num_training_steps = num_training_steps

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
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-5,
        )
        optim_dict = {"optimizer": optimizer}
        if self.scheduler_name is not None:
            scheduler = get_scheduler(
                self.scheduler_name,
                optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps,
            )
            optim_dict["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
            }
        return optim_dict