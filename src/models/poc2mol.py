import os
from typing import Any, Dict, Optional

import torch
import numpy as np
from lightning import LightningModule

from src.models.pytorch3dunet import ResidualUNetSE3D
from src.models.pytorch3dunet_lib.unet3d.buildingblocks import ResNetBlockSE, ResNetBlock
from transformers.optimization import get_scheduler
from torch.optim.lr_scheduler import StepLR

from src.evaluation.visual import show_3d_voxel_lig_only, visualise_batch

from src.models.pytorch3dunet_lib.unet3d.losses import get_loss_criterion

class ResUnetConfig:
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        final_sigmoid: bool = False,
        f_maps: int = 64,
        layer_order: str = 'gcr',
        num_groups: int = 8,
        num_levels: int = 5,
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
        self.conv_padding = conv_padding
        self.conv_upscale = conv_upscale
        self.upsample = upsample
        self.dropout_prob = dropout_prob
        self.basic_module = basic_module

class Poc2Mol(LightningModule):
    def __init__(
        self,
        config: ResUnetConfig,
        loss="BCEDiceLoss",
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        scheduler: Optional[Dict[str, Any]] = None,
        scheduler_name: str = None,  # legacy support
        num_training_steps: Optional[int] = 100000,
        num_warmup_steps: int = 0,
        num_decay_steps: int = 0,
        img_save_dir: str = None,
        scheduler_kwargs: Dict[str, Any] = None,
        matmul_precision: str = 'high',
        compile: bool = False,
        override_optimizer_on_load: bool = False,
        visualise_train: bool = True,
        visualise_val: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        assert config.final_sigmoid == False, "final_sigmoid must be False"
        self.model = ResidualUNetSE3D(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            final_sigmoid=config.final_sigmoid,
            f_maps=config.f_maps,
            layer_order=config.layer_order,
            num_groups=config.num_groups,
            num_levels=config.num_levels,
            conv_padding=config.conv_padding,
            conv_upscale=config.conv_upscale,
            upsample=config.upsample,
            dropout_prob=config.dropout_prob,
            basic_module=config.basic_module,
        )
        self.loss = get_loss_criterion(loss, with_logits=True)
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler or {}
        self.scheduler_name = scheduler_name  # legacy
        self.scheduler_kwargs = scheduler_kwargs
        self.num_decay_steps = num_decay_steps
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.img_save_dir = img_save_dir
        torch.set_float32_matmul_precision(matmul_precision)
        self.override_optimizer_on_load = override_optimizer_on_load
        self.visualise_val = visualise_val
        self.visualise_train = visualise_train
        self.residual_unet_config = config


    def forward(self, prot_vox, labels=None):
        """Run the UNet and (optionally) compute loss.

        Parameters
        ----------
        prot_vox : torch.Tensor
            Protein voxel tensor of shape ``[B, C, X, Y, Z]``.
        labels : torch.Tensor | None, optional
            Ground-truth ligand voxels.  If supplied, the method will also
            compute the configured loss criterion.

        Returns
        -------
        Dict[str, torch.Tensor]
            The returned dictionary always contains at least the following keys

            ``predicted_ligand_logits``
                The raw UNet output (logits).

            ``predicted_ligand_voxels``
                Sigmoid-activated version of the logits which can be treated as
                probabilities/occupancies.

            ``loss`` *(optional)*
                Total loss (only present when *labels* is given).

            In training mode (i.e. when *labels* are provided) the dictionary
            additionally contains one entry per individual loss component (e.g.
            ``bce`` and ``dice``).
        """
        # ------------------------------------------------------------------
        # Forward through the UNet
        # ------------------------------------------------------------------
        pred_logits = self.model(x=prot_vox)  # raw network output

        # Always compute the sigmoid once so that we have a consistent
        # probability representation available.
        pred_vox = torch.sigmoid(pred_logits)

        # If no labels are provided we are in inference mode – simply return
        # the predictions.
        if labels is None:
            return {
                "predicted_ligand_logits": pred_logits,
                "predicted_ligand_voxels": pred_vox,
            }

        # ------------------------------------------------------------------
        # Compute the per-component losses
        # ------------------------------------------------------------------
        result = self.loss(pred_logits, labels)

        # Aggregate to a single scalar so that Lightning knows what to optimise
        total_loss = None
        if isinstance(result, dict):
            total_loss = sum(result.values())

        result["loss"] = total_loss
        # Raw logits and sigmoid activations
        result["predicted_ligand_logits"] = pred_logits
        result["predicted_ligand_voxels"] = pred_vox


        return result

    def training_step(self, batch, batch_idx):
        if "load_time" in batch:
            self.log("train/load_time", batch["load_time"].mean(), on_step=True, on_epoch=False, prog_bar=True)

        outputs = self(batch["protein"], labels=batch["ligand"])
        # ``outputs`` is a dictionary – extract what we need
        pred_vox = outputs["predicted_ligand_voxels"]
        loss = outputs["loss"]

        # Log each individual loss component (except the prediction tensor)
        for k, v in outputs.items():
            if k in {"predicted_ligand_voxels", "predicted_ligand_logits"}:
                continue
            # Per-batch logging
            self.log(f"train/batch_{k}", v, on_step=True, on_epoch=False, prog_bar=(k=="loss"))
            # Per-epoch logging
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=(k=="loss"))

        # Channel statistics
        self.log_channel_means(batch, pred_vox)

        # Visualisation
        if batch_idx == 0 and self.visualise_train:
            outputs_for_viz = pred_vox.float().detach().cpu().numpy()
            visualise_batch(batch["ligand"], outputs_for_viz, batch["name"], save_dir=self.img_save_dir, batch=str(batch_idx))

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["protein"], labels=batch["ligand"])
        pred_vox = outputs["predicted_ligand_voxels"]
        loss = outputs["loss"]

        # Log each component
        for k, v in outputs.items():
            if k in {"predicted_ligand_voxels", "predicted_ligand_logits"}:
                continue
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=(k=="loss"))

        # Optional visualisation
        if self.visualise_val and batch_idx in [0, 50, 100]:
            save_dir = f"{self.img_save_dir}/val" if self.img_save_dir else None
            outputs_for_viz = pred_vox.float().detach().cpu().numpy()
            lig, pred, names = batch["ligand"][:4], outputs_for_viz[:4], batch["name"][:4]
        
            visualise_batch(
                lig,
                pred,
                names,
                save_dir=save_dir,
                batch=str(batch_idx)
            )

        
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(batch["protein"], labels=batch["ligand"])
        return outputs["loss"]

    def configure_optimizers(self) -> Dict[str, Any]:

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-5,
        )

        # Resolve scheduler configuration
        scheduler_config = self.scheduler_config.copy()

        # Fallback to legacy single-name arguments if new config not provided
        if not scheduler_config and self.scheduler_name is not None:
            scheduler_config = {
                "type": self.scheduler_name,
                "num_warmup_steps": self.num_warmup_steps,
            }

        if not scheduler_config:
            # No scheduler requested – return optimizer only
            return {"optimizer": optimizer}

        scheduler_type = scheduler_config.get("type", "step")

        if scheduler_type == "step":
            step_size = scheduler_config.get("step_size", 100)
            gamma = scheduler_config.get("gamma", 0.997)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": scheduler_config.get("interval", "step"),
                    "frequency": scheduler_config.get("frequency", 1),
                },
            }
        else:
            # Use HuggingFace get_scheduler for advanced schedulers
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
    
    def log_channel_means(self, batch, pred_vox):
        """Log mean values for each channel of protein, true ligand and prediction."""
        n_lig_channels = batch['ligand'].shape[1]
        self.log_dict({
            f"channel_mean/ligand_{channel}": batch['ligand'][:, channel, ...].mean().detach().item()
            for channel in range(n_lig_channels)
        })
        self.log_dict({
            f"channel_mean/pred_ligand_{channel}": pred_vox[:, channel, ...].mean().detach().item()
            for channel in range(n_lig_channels)
        })
        n_prot_channels = batch['protein'].shape[1]
        self.log_dict({
            f"channel_mean/protein_{channel}": batch['protein'][:, channel, ...].mean().detach().item()
            for channel in range(n_prot_channels)
        })