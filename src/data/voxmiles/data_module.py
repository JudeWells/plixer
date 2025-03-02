import bisect
import glob
import hashlib
import itertools
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

from src.data.voxmiles_dataset import VoxMilesDataset, get_collate_function
from src.data.tokenizers.smiles_tokenizer import build_smiles_tokenizer


# TODO: in future we might actually want standalone dataset class for
# more flexible customisation (e.g. mapping uniprot ids via db)
@dataclass
class ProteinDatasetConfig:
    data_path_pattern: str
    name: str
    keep_gaps: bool = False
    keep_insertions: bool = False
    to_upper: bool = False
    # https://huggingface.co/docs/datasets-server/en/parquet
    is_parquet: bool = False


class StringObject:
    """
    Custom class to allow for
    non-tensor elements in batch
    """

    text: List[str]

    def to(self, device):
        return self


class CustomDataCollator:
    """
    Wraps DataCollatorForLanguageModeling
    allows us to include elements which are not
    tensors with seq_len dimension, eg. smiles strings
    """

    def __init__(self, tokenizer, mlm=False):
        self.base_collator = DataCollatorForLanguageModeling(tokenizer, mlm=mlm)

    def __call__(self, examples):
        has_ds_name = "ds_name" in examples[0]
        has_doc_hash = "doc_hash" in examples[0]
        if has_ds_name or has_doc_hash:
            if has_ds_name:
                ds_names = [example.pop("ds_name") for example in examples]
            if has_doc_hash:
                doc_hashes = [example.pop("doc_hash") for example in examples]
            batch = self.base_collator(examples)
            if has_ds_name:
                ds_names_obj = StringObject()
                ds_names_obj.text = ds_names
                batch["ds_name"] = ds_names_obj
            if has_doc_hash:
                doc_hash_obj = StringObject()
                doc_hash_obj.text = doc_hashes
                batch["doc_hash"] = doc_hash_obj
        else:
            batch = self.base_collator(examples)
        return batch

class VoxMilesDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        vox_size: float,
        box_dims: list,
        max_smiles_len: int = 256,
        batch_size: int = 32,
        num_workers: int = 0,
        random_rotation: bool = True,
        random_translation: float = 6.0,
    ):
        super().__init__()
        self.data_path = data_path
        self.vox_size = vox_size
        self.box_dims = box_dims
        self.max_smiles_len = max_smiles_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_rotation = random_rotation
        self.random_translation = random_translation

        self.tokenizer = build_smiles_tokenizer()
        self.collate_fn = get_collate_function(self.tokenizer)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = VoxMilesDataset(
                data_path=f"{self.data_path}/train",
                tokenizer=self.tokenizer,
                vox_size=self.vox_size,
                box_dims=self.box_dims,
                max_smiles_len=self.max_smiles_len,
                random_rotation=self.random_rotation,
                random_translation=self.random_translation,
            )
            self.val_dataset = VoxMilesDataset(
                data_path=f"{self.data_path}/val",
                tokenizer=self.tokenizer,
                vox_size=self.vox_size,
                box_dims=self.box_dims,
                max_smiles_len=self.max_smiles_len,
                random_rotation=self.random_rotation,
                random_translation=self.random_translation,
            )

        if stage == "test" or stage is None:
            self.test_dataset = VoxMilesDataset(
                data_path=f"{self.data_path}/test",
                tokenizer=self.tokenizer,
                vox_size=self.vox_size,
                box_dims=self.box_dims,
                max_smiles_len=self.max_smiles_len,
                random_rotation=True,
                random_translation=self.random_translation,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )