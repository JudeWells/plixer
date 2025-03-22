import os
from typing import Any, Dict, Optional

import torch
import numpy as np
from lightning import LightningModule

from src.models.pytorch3dunet import ResidualUNetSE3D
from src.models.pytorch3dunet_lib.unet3d.buildingblocks import ResNetBlockSE, ResNetBlock
from transformers.optimization import get_scheduler

from src.evaluation.visual import show_3d_voxel_lig_only, visualise_batch

class ResUnetConfig:
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
        loss: Dict = None,

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
        self.loss = loss

class Poc2Mol(LightningModule):
    def __init__(
        self,
        config: ResUnetConfig,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        scheduler_name: str = None,
        num_training_steps: Optional[int] = 100000,
        num_warmup_steps: int = 0,
        num_decay_steps: int = 0,
        img_save_dir: str = None,
        scheduler_kwargs: Dict[str, Any] = None,
        matmul_precision: str = 'high',
        compile: bool = False,
        override_optimizer_on_load: bool = False,
        visualise_val: bool = True,
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
            loss=config.loss,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs
        self.num_decay_steps = num_decay_steps
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.img_save_dir = img_save_dir
        torch.set_float32_matmul_precision(matmul_precision)
        self.override_optimizer_on_load = override_optimizer_on_load
        self.visualise_val = visualise_val


    def forward(self, prot_vox, labels=None):
        return self.model(x=prot_vox, labels=labels)

    def training_step(self, batch, batch_idx):
        if "load_time" in batch:
            self.log("train/load_time", batch["load_time"].mean(), on_step=True, on_epoch=False, prog_bar=True)
        outputs, loss = self(batch["protein"], labels=batch["ligand"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/batch_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, loss = self(batch["protein"], labels=batch["ligand"])
        save_dir = f"{self.img_save_dir}/val"
        if self.visualise_val and batch_idx in [0, 50, 100]:
            lig, pred, names = batch["ligand"][:4], outputs[:4], batch["name"][:4]

            visualise_batch(
                lig,
                pred,
                names,
                save_dir=save_dir,
                batch=str(batch_idx)
            )

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        outputs, loss = self(batch["protein"], labels=batch["ligand"])
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-5,
        )
        if self.scheduler_name is not None:
            scheduler = get_scheduler(
                self.scheduler_name,
                optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps,
                scheduler_specific_kwargs=self.scheduler_kwargs,
            )

        return [optimizer], [scheduler]
    
    
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