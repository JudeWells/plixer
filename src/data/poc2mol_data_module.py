import os

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Optional
from src.data.poc2mol_datasets import DockstringTestDataset, ComplexDataset
import torch

class DataConfig:
    def __init__(
            self,
            vox_config,
            batch_size: int = 32,
            dtype=torch.float16,
            ligand_channels: list[int] = [0, 1, 2, 3, 4, 5],
            protein_channels: list[int] = [6, 7, 8, 9, 10, 11],
            fnames: Optional[list] = None,
            max_atom_dist: Optional[float] = 24.0,
    ):
        self.vox_config = vox_config
        self.batch_size = batch_size
        self.dtype = dtype
        self.fnames = fnames
        self.ligand_channels = ligand_channels
        self.protein_channels = protein_channels
        self.max_atom_dist = max_atom_dist


class ComplexDataModule(LightningDataModule):
    def __init__(self, config: DataConfig, pdb_dir: str, val_pdb_dir: str):
        super().__init__()
        self.config = config
        self.pdb_dir = pdb_dir
        self.val_pdb_dir = val_pdb_dir

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ComplexDataset(
            self.config,
            pdb_dir=self.pdb_dir,
            translation=self.config.vox_config.random_translation,
            rotate=self.config.vox_config.random_rotation,
        )
        self.val_dataset = ComplexDataset(
            self.config,
            pdb_dir=self.val_pdb_dir,
            translation=0.0,
            rotate=False
        )
        self.test_dataset = ComplexDataset(
            self.config,
            pdb_dir=self.val_pdb_dir,
            translation=0.0,
            rotate=False
        )



    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=4,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=4,
            shuffle=False,
        )
