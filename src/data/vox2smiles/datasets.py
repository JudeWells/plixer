import os
import glob
import pickle
import numpy as np
from rdkit import Chem
import torch
from torch.utils.data import Dataset

from src.data.common.voxelization.config import Vox2SmilesDataConfig
from src.data.common.voxelization.molecule_utils import (
    load_mol_from_pickle,
    prepare_rdkit_molecule,
    voxelize_molecule
)


def get_collate_function(tokenizer):
    """
    Create a collate function for the Vox2Smiles dataset.
    This function handles batching of voxelized molecules and tokenized SMILES strings.
    """
    from transformers import DataCollatorWithPadding
    
    smiles_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def collate_fn(batch, smiles_collator=smiles_collator):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        text_batch = {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch])
        }
        text_batch = smiles_collator(text_batch)
        smiles_str = [item['smiles_str'] for item in batch]
        return {
            'pixel_values': pixel_values,
            'input_ids': text_batch['input_ids'],
            'attention_mask': text_batch['attention_mask'],
            'smiles_str': smiles_str
        }
    
    return collate_fn


class VoxMilesDataset(Dataset):
    """
    Dataset for voxelized molecules with SMILES strings.
    Loads RDKit molecules from pickle files, voxelizes them, and pairs them with SMILES strings.
    """
    def __init__(
        self,
        data_path: str,
        tokenizer,
        config: Vox2SmilesDataConfig,
        random_rotation: bool = None,
        random_translation: float = None
    ):
        self.data_path = data_path
        self.data = glob.glob(f"{data_path}/*.pickle")
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.pad_token
        self.max_smiles_len = config.max_smiles_len
        
        # Override config values if provided
        self.random_rotation = random_rotation if random_rotation is not None else config.random_rotation
        self.random_translation = random_translation if random_translation is not None else config.random_translation
        
        # Store the config
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Load a pickled RDKit molecule, voxelize it, and pair it with its SMILES string.
        """
        # Load the molecule
        path = self.data[idx]
        with open(path, "rb") as f:
            mol_data = pickle.load(f)
        
        # If there are multiple conformers, randomly select one
        if "conformers" in mol_data:
            conformer_idx = np.random.randint(0, len(mol_data["conformers"]))
            mol = mol_data["conformers"][conformer_idx]["rd_mol"]
        else:
            mol = mol_data["rd_mol"]
        
        # Create a temporary config with the current instance's settings
        temp_config = Vox2SmilesDataConfig(
            vox_size=self.config.vox_size,
            box_dims=self.config.box_dims,
            random_rotation=self.random_rotation,
            random_translation=self.random_translation,
            has_protein=False,  # Vox2Smiles doesn't use protein channels
            ligand_channel_names=self.config.ligand_channel_names,
            protein_channel_names=self.config.protein_channel_names,
            protein_channels=self.config.protein_channels,
            ligand_channels=self.config.ligand_channels,
            max_atom_dist=self.config.max_atom_dist,
            dtype=self.config.dtype
        )
        
        # Voxelize the molecule
        voxel = voxelize_molecule(mol, temp_config)
        
        # Get the SMILES string
        smiles_str = self.tokenizer.bos_token + Chem.MolToSmiles(mol) + self.tokenizer.eos_token
        
        # Tokenize the SMILES string
        smiles = self.tokenizer(
            smiles_str,
            padding='max_length',
            max_length=self.max_smiles_len,
            truncation=True,
            return_tensors="pt"
        )
        
        # Return the voxelized molecule and tokenized SMILES string
        return {
            "pixel_values": voxel,
            "input_ids": smiles["input_ids"].squeeze(),
            "attention_mask": smiles["attention_mask"].squeeze(),
            "smiles_str": smiles_str
        }


class Poc2MolOutputDataset(Dataset):
    """
    Dataset that uses the output of a Poc2Mol model as input to Vox2Smiles.
    This is used for fine-tuning Vox2Smiles on the outputs of Poc2Mol.
    """
    def __init__(
        self,
        poc2mol_model,
        complex_dataset,
        tokenizer,
        max_smiles_len=200
    ):
        self.poc2mol_model = poc2mol_model
        self.complex_dataset = complex_dataset
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.pad_token
        self.max_smiles_len = max_smiles_len
        
        # Put the model in eval mode
        self.poc2mol_model.eval()

    def __len__(self):
        return len(self.complex_dataset)

    def __getitem__(self, idx):
        """
        Get a protein-ligand complex, generate a predicted ligand voxel using Poc2Mol,
        and pair it with the ground truth SMILES string.
        """
        # Get the protein-ligand complex
        complex_data = self.complex_dataset[idx]
        protein_voxel = complex_data['protein']
        ground_truth_ligand_voxel = complex_data['ligand']
        
        # Generate a predicted ligand voxel using Poc2Mol
        with torch.no_grad():
            predicted_ligand_voxel, _ = self.poc2mol_model(protein_voxel.unsqueeze(0))
            predicted_ligand_voxel = predicted_ligand_voxel.squeeze(0)
            # Move the tensor to CPU to avoid pin_memory issues
            predicted_ligand_voxel = predicted_ligand_voxel.cpu()
        
        # Get the SMILES string for the ground truth ligand
        # This would require additional processing to convert the voxel to a SMILES string
        # For now, we'll assume we have access to the SMILES string from somewhere else
        # In a real implementation, you would need to get this from your data source
        smiles_str = self.tokenizer.bos_token + "C1=CC=CC=C1" + self.tokenizer.eos_token  # Placeholder
        
        # Tokenize the SMILES string
        smiles = self.tokenizer(
            smiles_str,
            padding='max_length',
            max_length=self.max_smiles_len,
            truncation=True,
            return_tensors="pt"
        )
        
        # Return the predicted ligand voxel and tokenized SMILES string
        return {
            "pixel_values": predicted_ligand_voxel,
            "input_ids": smiles["input_ids"].squeeze(),
            "attention_mask": smiles["attention_mask"].squeeze(),
            "smiles_str": smiles_str
        }


class CombinedDataset(Dataset):
    """
    Dataset that combines Poc2Mol outputs and original Vox2Smiles data.
    This is used for fine-tuning Vox2Smiles on a mix of Poc2Mol outputs and original data.
    """
    def __init__(
        self,
        poc2mol_output_dataset,
        voxmiles_dataset,
        ratio=0.5
    ):
        self.poc2mol_output_dataset = poc2mol_output_dataset
        self.voxmiles_dataset = voxmiles_dataset
        self.ratio = ratio
        
        # Calculate the number of samples from each dataset
        self.n_poc2mol = len(poc2mol_output_dataset)
        self.n_voxmiles = len(voxmiles_dataset)
        
        # Calculate the total number of samples
        self.n_total = self.n_poc2mol + self.n_voxmiles
        
        # Calculate the probability of selecting a sample from each dataset
        self.p_poc2mol = self.n_poc2mol / self.n_total
        self.p_voxmiles = self.n_voxmiles / self.n_total

    def __len__(self):
        return self.n_total

    def __getitem__(self, idx):
        """
        Get a sample from either the Poc2Mol output dataset or the Vox2Smiles dataset.
        The probability of selecting from each dataset is proportional to the ratio.
        """
        # Determine which dataset to sample from
        if np.random.random() < self.ratio:
            # Sample from Poc2Mol output dataset
            idx_poc2mol = idx % self.n_poc2mol
            return self.poc2mol_output_dataset[idx_poc2mol]
        else:
            # Sample from Vox2Smiles dataset
            idx_voxmiles = idx % self.n_voxmiles
            return self.voxmiles_dataset[idx_voxmiles] 