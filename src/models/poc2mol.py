import torch
import numpy as np
from lightning import LightningModule
from torchmetrics import MeanMetric
from pytorch3dunet import ResidualUNetSE3D


class VoxToSmilesModel(LightningModule):
    def __init__(
        self,
        config,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = ResidualUNetSE3D(
            in_channels,
            out_channels,
            final_sigmoid=True,
            f_maps=64,
            layer_order='gcr',
            num_groups=8,
            num_levels=5,
            is_segmentation=True,
            conv_padding=1,
            conv_upscale=2,
            upsample='default',
            dropout_prob=0.1
        )

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
