import os
import glob
import pickle
import numpy as np
from rdkit import Chem
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding

from src.data.common.tokenizers.smiles_tokenizer import build_smiles_tokenizer
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
    
    pad_token_id = tokenizer.pad_token_id

    def collate_fn(batch, pad_token_id=pad_token_id):
        """Merge a list of dataset samples into a batch and trim trailing padding.
        """
        pixel_values = torch.stack([item["pixel_values"] for item in batch])  # (B, C, D, H, W)
        input_ids = torch.stack([item["input_ids"] for item in batch])        # (B, L)
        attention_mask = torch.stack([item["attention_mask"] for item in batch])  # (B, L)

        poc2mol_loss = None
        if "poc2mol_loss" in batch[0]:
            poc2mol_loss = torch.tensor([item["poc2mol_loss"] for item in batch])
        all_pad_positions = (input_ids == pad_token_id).all(dim=0)  # (L,)
        if torch.any(all_pad_positions):
            trim_len = torch.where(all_pad_positions)[0][0].item()
            input_ids = input_ids[:, :trim_len]
            attention_mask = attention_mask[:, :trim_len]

        smiles_str = [item["smiles_str"] for item in batch]
        # Optional decoy SMILES list (same for all items typically)
        has_candidates = "candidate_tokens" in batch[0]

        batch_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "smiles_str": smiles_str,
            "poc2mol_loss": poc2mol_loss,
        }
        if has_candidates:
            batch_dict["candidate_tokens"] = batch[0]["candidate_tokens"]
            batch_dict["binder_indices"] = torch.tensor([item["binder_index"] for item in batch], dtype=torch.long)
        return batch_dict
    
    return collate_fn


class Vox2SmilesDataset(Dataset):
    """
    Dataset for voxelized molecules with SMILES strings.
    Loads RDKit molecules from pickle files, voxelizes them, and pairs them with SMILES strings.
    """
    def __init__(
        self,
        data_path: str,
        config: Vox2SmilesDataConfig,
        random_rotation: bool = True,
        random_translation: float = 6.0
    ):
        self.data_path = data_path
        self.data = glob.glob(f"{data_path}/*.pickle")
        self.tokenizer = build_smiles_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.pad_token
        self.max_smiles_len = config.max_smiles_len
        
        # Override config values if provided
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        
        # Store the config
        self.config = config
        
        self.voxel_config = Vox2SmilesDataConfig(
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
        if not self.config.include_hydrogens:
            mol = Chem.RemoveHs(mol)
        
        # Voxelize the molecule
        voxel = voxelize_molecule(mol, self.voxel_config)
        
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


import os
import glob
import pandas as pd
import numpy as np
from rdkit import Chem
import torch
from torch.utils.data import Dataset

from src.data.common.voxelization.config import Vox2SmilesDataConfig
from src.data.common.voxelization.molecule_utils import voxelize_molecule


class ParquetVox2SmilesDataset(Dataset):
    """
    Dataset for voxelized molecules with SMILES strings loaded from parquet files.
    Each parquet file contains multiple molecules, allowing efficient storage and loading.
    """
    def __init__(
        self,
        data_path: str,
        config: Vox2SmilesDataConfig,
        index_file: str = "index.csv",
        random_rotation: bool = None,
        random_translation: float = None,
        cache_size: int = 10,
    ):
        self.data_path = data_path
        self.tokenizer = build_smiles_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.pad_token
        self.max_smiles_len = config.max_smiles_len
        
        # Override config values if provided
        self.random_rotation = random_rotation if random_rotation is not None else config.random_rotation
        self.random_translation = random_translation if random_translation is not None else config.random_translation
        
        # Store the config
        self.config = config
        
        # Load the index file which lists all parquet files
        index_path = os.path.join(data_path, index_file)
        if os.path.exists(index_path):
            df = pd.read_csv(index_path)
            self.file_list = df['parquet_file'].tolist()
            self.file_sizes = df['file_size'].tolist()
            self.total_molecules = sum(self.file_sizes)
        else:
            print("Indexing molecule files")
            self.file_list = glob.glob(os.path.join(data_path, "**.parquet"))
            print(f"Found {len(self.file_list)} parquet files")
            self.file_sizes = []
            self.total_molecules = 0
            for file_path in self.file_list:
                # Just read the metadata to get row count (faster than loading data)
                df = pd.read_parquet(file_path, columns=['source_file'])
                file_size = len(df)
                self.file_sizes.append(file_size)
                self.total_molecules += file_size
            pd.DataFrame({"parquet_file": self.file_list, "file_size": self.file_sizes}).to_csv(index_path, index=False)
        
        self.molecule_map = []
        for file_idx, size in enumerate(self.file_sizes):
            for row_idx in range(size):
                self.molecule_map.append((file_idx, row_idx))
        print(f"Molecule file index created with {len(self.molecule_map)} molecules")
        self.cache = {}
        self.cache_size = cache_size

        self.voxel_config =Vox2SmilesDataConfig(
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

    def __len__(self):
        return self.total_molecules

    def __getitem__(self, idx):
        """
        Load a molecule from a parquet file, voxelize it, and pair it with its SMILES string.
        """
        file_idx, row_idx = self.molecule_map[idx]
        file_path = self.file_list[file_idx]
        
        # Try to get the file from cache
        if file_path not in self.cache:
            # If cache is full, remove the oldest entry
            if len(self.cache) >= self.cache_size:
                # Get the least recently used file
                oldest_file = next(iter(self.cache))
                del self.cache[oldest_file]
            
            # Load the parquet file
            self.cache[file_path] = pd.read_parquet(file_path)
        
        # Get the molecule data
        mol_data = self.cache[file_path].iloc[row_idx]
        # Reconstruct the RDKit molecule from the mol block
        mol_block = mol_data['mol_block']
        mol = Chem.MolFromMolBlock(mol_block.decode() if isinstance(mol_block, bytes) else mol_block)
        if not self.config.include_hydrogens:
            mol = Chem.RemoveHs(mol)
        if mol is None:
            # If we can't parse the mol block, try to create from SMILES
            smiles = mol_data['smiles']
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self.__getitem__(np.random.randint(0, len(self)))

        
        # Add hydrogens and generate 3D coordinates if needed
        if mol.GetNumConformers() == 0:
            if self.config.include_hydrogens:
                mol = Chem.AddHs(mol)
            # Use a standard RDKit conformer generation
            try:
                from rdkit.Chem import AllChem
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            except:
                # If conformer generation fails, we'll skip this molecule
                # and provide a fallback
                mol = Chem.MolFromSmiles("c1ccccc1")
                if self.config.include_hydrogens:
                    mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        
        # Voxelize the molecule
        voxel = voxelize_molecule(mol, self.voxel_config)
        
        # Get the SMILES string
        smiles_str = self.tokenizer.bos_token + Chem.MolToSmiles(mol) + self.tokenizer.eos_token
        
        # Tokenize the SMILES string
        smiles = self.tokenizer(
            smiles_str,
            padding='max_length',
            max_length=self.max_smiles_len,
            truncation=True,
            return_tensors="pt"
        ).to(voxel.device)
        if self.tokenizer.unk_token_id in smiles.input_ids:
            print(f"UNK token in SMILES string: {smiles_str}")
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
        max_smiles_len=200,
        ckpt_path: str = None,
        decoy_smiles_list: list = None,
        include_decoys: bool = True,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.poc2mol_model = poc2mol_model.to(complex_dataset.config.dtype).to(self.device)
        self.complex_dataset = complex_dataset
        self.tokenizer = build_smiles_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.pad_token
        self.max_smiles_len = max_smiles_len
        if ckpt_path is not None:
            # Load the Lightning checkpoint
            checkpoint = torch.load(ckpt_path)
            # Extract the model state dict from the Lightning checkpoint
            if "state_dict" in checkpoint:
                # Remove 'model.' prefix if it exists in the keys
                state_dict = {k.replace('model.', ''): v for k, v in checkpoint["state_dict"].items()}
                self.poc2mol_model.model.load_state_dict(state_dict)
            else:
                # Fallback to direct loading if it's not a Lightning checkpoint
                self.poc2mol_model.load_state_dict(checkpoint)

        self.poc2mol_model.eval()

        self.include_decoys = include_decoys
        if self.include_decoys:
            assert decoy_smiles_list is not None, "decoy_smiles_list must be provided if include_decoys is True"
            self.decoy_smiles_list = decoy_smiles_list
            self.tokenize_decoys()
        else:
            self.decoy_smiles_list = []

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
        smiles_str = complex_data['smiles']
        # Generate a predicted ligand voxel using Poc2Mol
        with torch.no_grad():
            outputs = self.poc2mol_model(
                protein_voxel.unsqueeze(0),
                labels=ground_truth_ligand_voxel.unsqueeze(0)
            )
            predicted_ligand_voxel = outputs['predicted_ligand_voxels'].squeeze(0)

        binder_idx = None
        if self.include_decoys:
            # Find index of binder in global decoy list
            binder_idx = self.decoy_smiles_list.index(smiles_str)
            # candidate tokens are the full stacked tensors (shared across samples)
            candidate_tokens = self.tokenized_decoy_smiles
        else:
            candidate_tokens = None

        smiles_str = self.tokenizer.bos_token + smiles_str + self.tokenizer.eos_token
        
        # Tokenize the SMILES string
        smiles = self.tokenizer(
            smiles_str,
            padding='max_length',
            max_length=self.max_smiles_len,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        result = {
            "pixel_values": predicted_ligand_voxel,
            "input_ids": smiles["input_ids"].squeeze(),
            "attention_mask": smiles["attention_mask"].squeeze(),
            "smiles_str": smiles_str,
            "poc2mol_loss": outputs['loss'].item(),
        }

        if self.include_decoys:
            result.update({
                "candidate_tokens": candidate_tokens,  # dict with stacked tensors
                "binder_index": binder_idx,
            })

        return result

    def tokenize_decoys(self):
        """Tokenize the global decoy SMILES list **once** and store stacked tensors.

        The result is a dictionary with keys (input_ids, attention_mask, token_type_ids)
        each mapping to a tensor of shape (N_decoys, L).
        """
        smiles_with_tokens = [
            self.tokenizer.bos_token + smi + self.tokenizer.eos_token
            for smi in self.decoy_smiles_list
        ]

        max_len = max(
            len(self.tokenizer.tokenize(s, padding=False, truncation=False))
            for s in smiles_with_tokens
        )

        tokenized = self.tokenizer(
            smiles_with_tokens,
            padding='max_length',
            max_length=max_len,
            truncation=True,
            return_tensors='pt',
        )

        # Ensure tensors are on CPU to avoid unnecessary GPU memory duplication
        self.tokenized_decoy_smiles = {k: v for k, v in tokenized.items()}


class CombinedDataset(Dataset):
    """
    Dataset that combines Poc2Mol outputs and original Vox2Smiles data.
    This is used for fine-tuning Vox2Smiles on a mix of Poc2Mol outputs and original data.
    """
    def __init__(
        self,
        poc2mol_output_dataset,
        vox2smiles_dataset,
        prob_poc2mol=0.5, # probability of poc2mol
        max_poc2mol_loss=1.2, # worst loss tolerated to train on poc2mol
    ):
        self.poc2mol_output_dataset = poc2mol_output_dataset
        self.vox2smiles_dataset = vox2smiles_dataset
        self.prob_poc2mol = prob_poc2mol
        self.max_poc2mol_loss = max_poc2mol_loss
        # Calculate the number of samples from each dataset
        self.n_poc2mol = len(poc2mol_output_dataset)
        self.n_vox2smiles = len(vox2smiles_dataset)
        
        # Calculate the total number of samples
        self.n_total = self.n_poc2mol + self.n_vox2smiles
        print(f"Total number of samples: {self.n_total}")
        print(f"Number of Poc2Mol samples: {self.n_poc2mol}")
        print(f"Number of Vox2Smiles samples: {self.n_vox2smiles}")
        # Calculate the probability of selecting a sample from each dataset
        self.p_poc2mol = self.n_poc2mol / self.n_total
        self.p_vox2smiles = self.n_vox2smiles / self.n_total

    def __len__(self):
        return self.n_total

    def __getitem__(self, idx):
        """
        Get a sample from either the Poc2Mol output dataset or the Vox2Smiles dataset.
        """
        # Determine which dataset to sample from
        if np.random.random() < self.prob_poc2mol:
            # Sample from Poc2Mol output dataset
            idx_poc2mol = idx % self.n_poc2mol
            result = self.poc2mol_output_dataset[idx_poc2mol]
            if result['poc2mol_loss'] < self.max_poc2mol_loss:
                return result
            else:
                result =  self.vox2smiles_dataset[idx % self.n_vox2smiles]
                result['poc2mol_loss'] = -1
        else:
            # Sample from Vox2Smiles dataset
            idx_vox2smiles = idx % self.n_vox2smiles
            result = self.vox2smiles_dataset[idx_vox2smiles]
            result['poc2mol_loss'] = -1
        return result