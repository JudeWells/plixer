from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.common.voxelization.config import Vox2SmilesDataConfig
from src.data.common.tokenizers.smiles_tokenizer import build_smiles_tokenizer
from src.data.vox2smiles.datasets import VoxMilesDataset, get_collate_function


class VoxMilesDataModule(LightningDataModule):
    """
    Lightning data module for voxelized molecules with SMILES strings.
    """
    def __init__(
        self,
        config: Vox2SmilesDataConfig,
        data_path: str,
        val_split: float = 0.1,
        test_split: float = 0.1
    ):
        super().__init__()
        self.config = config
        self.data_path = data_path
        self.val_split = val_split
        self.test_split = test_split
        
        # Build the tokenizer
        self.tokenizer = build_smiles_tokenizer()
        
        # Create the collate function
        self.collate_fn = get_collate_function(self.tokenizer)

    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for each stage."""
        if stage == 'fit' or stage is None:
            self.train_dataset = VoxMilesDataset(
                data_path=f"{self.data_path}/train",
                tokenizer=self.tokenizer,
                config=self.config,
                random_rotation=self.config.random_rotation,
                random_translation=self.config.random_translation
            )
            
            self.val_dataset = VoxMilesDataset(
                data_path=f"{self.data_path}/val",
                tokenizer=self.tokenizer,
                config=self.config,
                random_rotation=False,  # No rotation for validation
                random_translation=0.0   # No translation for validation
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = VoxMilesDataset(
                data_path=f"{self.data_path}/test",
                tokenizer=self.tokenizer,
                config=self.config,
                random_rotation=False,  # No rotation for testing
                random_translation=0.0   # No translation for testing
            )

    def train_dataloader(self):
        """Get the training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Get the validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Get the test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        ) 