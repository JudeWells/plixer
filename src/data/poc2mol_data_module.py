import os

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Optional
from src.data.poc2mol_datasets import DockstringTestDataset, ComplexDataset, StructuralPretrainDataset
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
            coord_indices: Optional[list[int]] = None,
    ):
        self.vox_config = vox_config
        self.batch_size = batch_size
        self.dtype = dtype
        self.fnames = fnames
        self.ligand_channels = ligand_channels
        self.protein_channels = protein_channels
        self.max_atom_dist = max_atom_dist
        self.coord_indices = coord_indices

class ComplexDataModule(LightningDataModule):
    def __init__(self, config: DataConfig, pdb_dir: str, val_pdb_dir: str):
        super().__init__()
        self.config = config
        self.pdb_dir = pdb_dir
        self.val_pdb_dir = val_pdb_dir
        self.save_hyperparameters(ignore=['config'])

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
            pin_memory=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=4,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=4,
            shuffle=False,
        )


class StructuralPretrainDataModule(LightningDataModule):
    """
    DataModule for pre-training structural autoencoder.
    """
    def __init__(self, config: DataConfig, train_pdb_dir: str, val_pdb_dir: str, num_workers: int = 4):
        super().__init__()
        self.config = config
        self.train_pdb_dir = train_pdb_dir
        self.val_pdb_dir = val_pdb_dir
        self.batch_size = config.batch_size
        self.num_workers = num_workers
        self.save_hyperparameters(ignore=['config'])
    
    def setup(self, stage: Optional[str] = None):
        # Create training dataset with rotations
        self.train_dataset = StructuralPretrainDataset(
            self.config,
            pdb_dir=self.train_pdb_dir,
            rotate=True,
        )
        
        # Create validation dataset without rotations
        self.val_dataset = StructuralPretrainDataset(
            self.config,
            pdb_dir=self.val_pdb_dir,
            rotate=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )