from lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Optional
from src.data.datasets import DockstringTestDataset, ComplexDataset
import torch

class DataConfig:
    def __init__(
            self,
            pdb_dir: str,
            rotate: bool = False,
            translation: float = 0.0,
            vox_size: float = 0.75,
            box_dims: list = [24, 24, 24],
            batch_size: int = 32,
            num_workers: int = 0,
            dtype=torch.float16,
            ligand_channels: list[int] = [0, 1, 2, 3, 4, 5],
            protein_channels: list[int] = [6, 7, 8, 9, 10, 11],
            fnames: Optional[list] = None,
    ):
        self.pdb_dir = pdb_dir
        self.rotate = rotate
        self.translation = translation
        self.vox_size = vox_size
        self.box_dims = box_dims
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dtype = dtype
        self.fnames = fnames
        self.ligand_channels = ligand_channels
        self.protein_channels = protein_channels


class ComplexDataModule(LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ComplexDataset(self.config)
        self.val_dataset = None
        self.test_dataset = None



    def train_dataloader(self):
        return DataLoader(self.train_dataset)

    def val_dataloader(self):
        return DataLoader(self.train_dataset)

    def test_dataloader(self):
        return DataLoader(self.train_dataset)
