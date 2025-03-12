from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.common.voxelization.config import Poc2MolDataConfig
from src.data.poc2mol.datasets import ComplexDataset, DockstringTestDataset
from torch.utils.data import Dataset

class ComplexDataModule(LightningDataModule):
    """
    Lightning data module for protein-ligand complexes.
    """
    def __init__(
        self,
        config: Poc2MolDataConfig,
        pdb_dir: str,
        val_pdb_dir: str,
        test_pdb_dir: Optional[str] = None,
        num_workers: int = 0,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ):
        super().__init__()
        self.config = config
        self.pdb_dir = pdb_dir
        self.val_pdb_dir = val_pdb_dir
        self.test_pdb_dir = test_pdb_dir or val_pdb_dir  # Use val_pdb_dir as default for test
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for each stage."""
        if stage == 'fit' or stage is None:
            if self.train_dataset is None:
                self.train_dataset = ComplexDataset(
                    self.config,
                    pdb_dir=self.pdb_dir,
                    translation=self.config.random_translation,
                    rotate=self.config.random_rotation,
                )
            else:
                self.train_dataset = self.train_dataset
            
            if self.val_dataset is None:
                self.val_dataset = ComplexDataset(
                    self.config,
                    pdb_dir=self.val_pdb_dir,
                    translation=0.0,  # No translation for validation
                    rotate=False,     # No rotation for validation
                )
            else:
                self.val_dataset = self.val_dataset
        
        if stage == 'test' or stage is None:
            if self.test_dataset is None:
                self.test_dataset = ComplexDataset(
                    self.config,
                    pdb_dir=self.test_pdb_dir,
                    translation=0.0,  # No translation for testing
                    rotate=False,     # No rotation for testing
                )
            else:
                self.test_dataset = self.test_dataset

    def train_dataloader(self):
        """Get the training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        """Get the validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=min(4, self.config.batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers= False, #True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        """Get the test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=min(4, self.config.batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        ) 