import glob
import pickle

from docktgrid.transforms import RandomRotation
import numpy as np
from rdkit import Chem
import torch
from torch.utils.data import Dataset

from src.data.rdkit_voxelizer import RDkitVoxelGrid, RDkitMolecularComplex
from src.data.voxel_views import LigView
from src.utils.evaluation import show_3d_voxel_lig_only

from transformers import DataCollatorWithPadding

def get_collate_function(tokenizer):
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
    A custom dataset that loads pairs
    of 3d voxel images and SMILES strings.
    """
    def __init__(self,
                 data_path,
                 tokenizer,
                 vox_size,
                 box_dims,
                 max_smiles_len=256,
                 random_rotation=True,
                 random_translation=6.0):
        self.data_path = data_path
        self.data = glob.glob(f"{data_path}/*.pickle")
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.pad_token
        self.max_smiles_len = max_smiles_len
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        self.voxelizer = RDkitVoxelGrid(
            views=[LigView()],  # you can add multiple views; they are executed in order
            vox_size=vox_size,  # size of the voxel (in Angstrom)
            box_dims=box_dims,  # dimensions of the box (in Angstrom)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        load the pickled RDKit mol
        pass it through the voxelizer function

        :param idx:
        :return:
        """
        path = self.data[idx]
        with open(path, "rb") as f:
            mol_data = pickle.load(f)
        conformer_idx = np.random.randint(0, len(mol_data["conformers"])) #  todo weight by energy
        mol = mol_data["conformers"][conformer_idx]["rd_mol"]
        molecular_complex = RDkitMolecularComplex(mol)
        if self.random_rotation:
            rotation = RandomRotation()
            rotation(molecular_complex.coords, molecular_complex.ligand_center)
        if self.random_translation:
            translation_vector_length = np.random.uniform(0, self.random_translation)
            translation_vector = torch.tensor(
                np.random.uniform(-1, 1, 3) * translation_vector_length,
                dtype=torch.float16)
            molecular_complex.ligand_center += translation_vector # shift center without coord change means ligand not centre of box
        voxel = self.voxelizer.voxelize(molecular_complex)
        # show_3d_voxel_lig_only(voxel, angles=None, save_dir="/home/judewells/Downloads/vis_mols", identifier='lig')
        smiles_str = self.tokenizer.bos_token + Chem.MolToSmiles(mol) + self.tokenizer.eos_token
        smiles = self.tokenizer(smiles_str,
                                padding='max_length',
                                max_length=self.max_smiles_len,
                                truncation=True,
                                return_tensors="pt")
        return {"pixel_values": voxel,
                "input_ids": smiles["input_ids"].squeeze(),
                "attention_mask": smiles["attention_mask"].squeeze(),
                "smiles_str": smiles_str}
