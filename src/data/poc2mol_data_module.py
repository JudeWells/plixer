import os

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Optional
from src.data.datasets import DockstringTestDataset, ComplexDataset
import torch

class DataConfig:
    def __init__(
            self,
            pdb_dir: str,
            vox_config,
            batch_size: int = 32,
            dtype=torch.float16,
            ligand_channels: list[int] = [0, 1, 2, 3, 4, 5],
            protein_channels: list[int] = [6, 7, 8, 9, 10, 11],
            fnames: Optional[list] = None,
            max_atom_dist: Optional[float] = 24.0,
    ):
        self.pdb_dir = pdb_dir
        self.vox_config = vox_config
        self.batch_size = batch_size
        self.dtype = dtype
        self.fnames = fnames
        self.ligand_channels = ligand_channels
        self.protein_channels = protein_channels
        self.max_atom_dist = max_atom_dist


class ComplexDataModule(LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ComplexDataset(self.config)
        self.val_dataset = ComplexDataset(self.config)
        self.test_dataset = ComplexDataset(self.config)



    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
