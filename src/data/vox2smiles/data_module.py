from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.common.voxelization.config import Vox2SmilesDataConfig
from src.data.common.tokenizers.smiles_tokenizer import build_smiles_tokenizer
from src.data.vox2smiles.datasets import Vox2SmilesDataset, get_collate_function


class Vox2SmilesDataModule(LightningDataModule):
    """
    Lightning data module for voxelized molecules with SMILES strings.
    """
    def __init__(
        self,
        config: Vox2SmilesDataConfig,
        data_path: str,
        val_split: float = 0.1,
        test_split: float = 0.1,
        train_dataset = None,
        val_datasets = None,  # New unified mapping of validation datasets
        test_dataset = None,
        num_workers = 0
    ):
        super().__init__()
        self.config = config
        self.data_path = data_path
        self.val_split = val_split
        self.test_split = test_split
        
        self.train_dataset_provided = train_dataset
        self.val_datasets_provided = {}

        if val_datasets is not None:
            self.val_datasets_provided.update(val_datasets)

        # Handle legacy single val_dataset & secondary_val_dataset if passed via kwargs
        legacy_val_dataset = locals().get('val_dataset', None)
        if legacy_val_dataset is not None:
            self.val_datasets_provided['val'] = legacy_val_dataset
        legacy_secondary = locals().get('secondary_val_dataset', None)
        if legacy_secondary is not None:
            self.val_datasets_provided['secondary'] = legacy_secondary

        self.test_dataset_provided = test_dataset
        self.tokenizer = build_smiles_tokenizer()
        self.collate_fn = get_collate_function(self.tokenizer)
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for each stage."""
        if stage == 'fit' or stage is None:
            # Use provided datasets if available, otherwise create default ones
            if self.train_dataset_provided is not None:
                self.train_dataset = self.train_dataset_provided
            else:
                self.train_dataset = Vox2SmilesDataset(
                    data_path=f"{self.data_path}/train",

                    config=self.config,
                    random_rotation=self.config.random_rotation,
                    random_translation=self.config.random_translation
                )
            
            # ---------------- Validation datasets ----------------
            if self.val_datasets_provided:
                # User supplied mapping of datasets; use directly
                self.val_datasets = self.val_datasets_provided
            else:
                # Legacy behaviour – single default validation dataset
                default_val_ds = Vox2SmilesDataset(
                    data_path=f"{self.data_path}/val_5k",
                    config=self.config,
                    random_rotation=False,
                    random_translation=0.0,
                )
                self.val_datasets = {"default_val": default_val_ds}
        
        if stage == 'test' or stage is None:
            if self.test_dataset_provided is not None:
                self.test_dataset = self.test_dataset_provided
            else:
                self.test_dataset = Vox2SmilesDataset(
                    data_path=f"{self.data_path}/test",
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
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        """Get the validation data loader."""
        loaders = []
        for name, dataset in self.val_datasets.items():
            # Determine batch size – prefer dataset.config.val_batch_size if present
            batch_size = getattr(self.config, 'val_batch_size', getattr(self.config, 'batch_size', 32))
            # Allow dataset to override via attribute `batch_size`
            if hasattr(dataset, 'config') and hasattr(dataset.config, 'val_batch_size'):
                batch_size = dataset.config.val_batch_size
            loaders.append(
                DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    collate_fn=self.collate_fn,
                    pin_memory=False,
                    persistent_workers=True if self.num_workers > 0 else False,
                )
            )
        return loaders

    def test_dataloader(self):
        """Get the test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
        ) 