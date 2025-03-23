from typing import Optional
import os
import glob
from rdkit import Chem
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.common.voxelization.config import Poc2MolDataConfig, Vox2SmilesDataConfig
from src.data.common.tokenizers.smiles_tokenizer import build_smiles_tokenizer
from src.data.poc2mol.datasets import ComplexDataset



def collate_fn(batch):
    # Collate protein and ligand voxels
    protein_voxels = torch.stack([item["protein"] for item in batch])
    ligand_voxels = torch.stack([item["ligand"] for item in batch])
    
    # Collate SMILES data
    smiles = torch.stack([item["smiles"] for item in batch])
    smiles_attention_mask = torch.stack([item["smiles_attention_mask"] for item in batch])
    smiles_str = [item["smiles_str"] for item in batch]
    
    # Collate names
    names = [item["name"] for item in batch]
    
    return {
        "protein": protein_voxels,
        "ligand": ligand_voxels,
        "smiles": smiles,
        "smiles_attention_mask": smiles_attention_mask,
        "smiles_str": smiles_str,
        "name": names
    }

class ProteinLigandSmilesDataset(Dataset):
    """
    Dataset for protein-ligand complexes with SMILES strings.
    This dataset extends the ComplexDataset to include SMILES strings for the ligands.
    """
    def __init__(
        self,
        config: Poc2MolDataConfig,
        pdb_dir: str,
        translation: float = None,
        rotate: bool = None,
        max_smiles_len: int = 200
    ):
        # Initialize the base ComplexDataset
        self.complex_dataset = ComplexDataset(
            config=config,
            pdb_dir=pdb_dir,
            translation=translation,
            rotate=rotate
        )
        
        # Initialize the tokenizer
        self.tokenizer = build_smiles_tokenizer()
        self.max_smiles_len = max_smiles_len
        
        # Cache SMILES strings for each complex
        self.smiles_cache = {}
        self._load_smiles()

    def _load_smiles(self):
        """
        Load SMILES strings for each ligand.
        This is a placeholder implementation. In a real implementation, you would
        load the SMILES strings from your data source.
        """
        # In a real implementation, you would load the SMILES strings from your data source
        # For now, we'll just use a placeholder
        for i, path in enumerate(self.complex_dataset.struct_paths):
            # Try to find a SMILES file in the same directory as the ligand
            smiles_path = os.path.join(path, '*_ligand.smi')
            smiles_files = glob.glob(smiles_path)
            
            if smiles_files:
                # If a SMILES file exists, read it
                with open(smiles_files[0], 'r') as f:
                    smiles = f.read().strip()
            else:
                # Otherwise, use a placeholder
                smiles = "C1=CC=CC=C1"  # Benzene as a placeholder
            
            self.smiles_cache[i] = smiles

    def __len__(self):
        return len(self.complex_dataset)

    def __getitem__(self, idx):
        """
        Get a protein-ligand complex with SMILES string.
        """
        # Get the protein-ligand complex
        complex_data = self.complex_dataset[idx]
        
        # Get the SMILES string
        smiles_str = self.tokenizer.bos_token + self.smiles_cache[idx] + self.tokenizer.eos_token
        
        # Tokenize the SMILES string
        smiles = self.tokenizer(
            smiles_str,
            padding='max_length',
            max_length=self.max_smiles_len,
            truncation=True,
            return_tensors="pt"
        )
        
        # Add the SMILES data to the complex data
        complex_data["smiles"] = smiles["input_ids"].squeeze()
        complex_data["smiles_attention_mask"] = smiles["attention_mask"].squeeze()
        complex_data["smiles_str"] = smiles_str
        
        return complex_data


class ProteinLigandSmilesDataModule(LightningDataModule):
    """
    Lightning data module for protein-ligand complexes with SMILES strings.
    This data module is used for the combined Poc2Smiles model.
    """
    def __init__(
        self,
        poc2mol_config: Poc2MolDataConfig,
        vox2smiles_config: Vox2SmilesDataConfig,
        pdb_dir: str,
        val_pdb_dir: str,
        test_pdb_dir: Optional[str] = None,
        num_workers: int = 4
    ):
        super().__init__()
        self.poc2mol_config = poc2mol_config
        self.vox2smiles_config = vox2smiles_config
        self.pdb_dir = pdb_dir
        self.val_pdb_dir = val_pdb_dir
        self.test_pdb_dir = test_pdb_dir or val_pdb_dir  # Use val_pdb_dir as default for test
        self.num_workers = num_workers
        # Build the tokenizer
        self.tokenizer = build_smiles_tokenizer()
        
        # Create the collate function
        self.collate_fn = collate_fn

    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for each stage."""
        if stage == 'fit' or stage is None:
            self.train_dataset = ProteinLigandSmilesDataset(
                config=self.poc2mol_config,
                pdb_dir=self.pdb_dir,
                translation=self.poc2mol_config.random_translation,
                rotate=self.poc2mol_config.random_rotation,
                max_smiles_len=self.vox2smiles_config.max_smiles_len
            )
            
            self.val_dataset = ProteinLigandSmilesDataset(
                config=self.poc2mol_config,
                pdb_dir=self.val_pdb_dir,
                translation=0.0,  # No translation for validation
                rotate=False,     # No rotation for validation
                max_smiles_len=self.vox2smiles_config.max_smiles_len
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = ProteinLigandSmilesDataset(
                config=self.poc2mol_config,
                pdb_dir=self.test_pdb_dir,
                translation=0.0,  # No translation for testing
                rotate=False,     # No rotation for testing
                max_smiles_len=self.vox2smiles_config.max_smiles_len
            )

    def train_dataloader(self):
        """Get the training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.poc2mol_config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        """Get the validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=min(4, self.poc2mol_config.batch_size),  # Smaller batch size for validation
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        """Get the test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=min(4, self.poc2mol_config.batch_size),  # Smaller batch size for testing
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )