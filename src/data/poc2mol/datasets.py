import os
import glob
import torch
from torch.utils.data import Dataset

from src.data.common.voxelization.config import Poc2MolDataConfig
from src.data.common.voxelization.molecule_utils import (
    prepare_protein_ligand_complex,
    voxelize_complex
)


class ComplexDataset(Dataset):
    """
    Dataset for protein-ligand complexes.
    Generates protein-ligand complexes on the fly from PDB and MOL2 files and voxelizes them.
    """
    def __init__(
        self,
        config: Poc2MolDataConfig,
        pdb_dir: str,
        translation: float = None,
        rotate: bool = None
    ):
        self.config = config
        self.pdb_dir = pdb_dir
        
        # Override config values if provided
        self.random_translation = translation if translation is not None else config.random_translation
        self.random_rotation = rotate if rotate is not None else config.random_rotation
        
        # Get paths to protein-ligand complexes
        self.struct_paths = self.get_complex_paths()
        
        # Set up channel indices
        self.ligand_channel_indices = config.ligand_channel_indices
        self.protein_channel_indices = config.protein_channel_indices
        
        # Set up maximum atom distance
        self.max_atom_dist = config.max_atom_dist

    def get_complex_paths(self):
        """Get paths to protein-ligand complexes."""
        protein_paths = ['/'.join(p.split("/")[:-1]) for p in glob.glob(f"{self.pdb_dir}/*/*_protein.pdb")]
        ligand_paths = ['/'.join(p.split("/")[:-1]) for p in glob.glob(f"{self.pdb_dir}/*/*_ligand.mol2")]
        return sorted(list(set(protein_paths).intersection(set(ligand_paths))))

    def __len__(self):
        return len(self.struct_paths)

    def __getitem__(self, idx):
        """Get a protein-ligand complex and voxelize it."""
        # Get paths to protein and ligand files
        directory = self.struct_paths[idx]
        pdb_path = glob.glob(os.path.join(directory, '*_protein.pdb'))[0]
        lig_path = glob.glob(os.path.join(directory, '*_ligand.mol2'))[0]
        
        # Create a temporary config with the current instance's settings
        temp_config = Poc2MolDataConfig(
            vox_size=self.config.vox_size,
            box_dims=self.config.box_dims,
            random_rotation=self.random_rotation,
            random_translation=self.random_translation,
            has_protein=self.config.has_protein,
            ligand_channel_names=self.config.ligand_channel_names,
            protein_channel_names=self.config.protein_channel_names,
            protein_channels=self.config.protein_channels,
            ligand_channels=self.config.ligand_channels,
            max_atom_dist=self.max_atom_dist,
            dtype=eval(self.config.dtype) if isinstance(self.config.dtype, str) else self.config.dtype
        )
        
        # Voxelize the complex
        protein_voxel, ligand_voxel, _ = voxelize_complex(pdb_path, lig_path, temp_config)
        
        # Return the voxelized complex
        return {
            'ligand': ligand_voxel,
            'protein': protein_voxel,
            'name': directory.split('/')[-1]
        }


class DockstringTestDataset(Dataset):
    """
    Dataset for testing on Dockstring data.
    This version tracks whether each ligand is active or inactive based on the directory structure.
    """
    def __init__(self, config: Poc2MolDataConfig):
        self.config = config
        self.voxel_dir = config.dockstring_test_dir
        self.target_dirs = [os.path.join(self.voxel_dir, f) for f in os.listdir(self.voxel_dir) if os.path.isdir(os.path.join(self.voxel_dir, f))]
        self.target_dirs = [f for f in self.target_dirs if {'actives', 'inactives'}.issubset(set(os.listdir(f)))]

        # Assign labels based on directory names
        self.voxel_files = []
        self.labels = []  # This list will hold the label for each file
        self.target_ids = []  # This list will hold the protein target id for each ligand

        for target_dir in self.target_dirs:
            active_dir = os.path.join(target_dir, 'actives')
            inactive_dir = os.path.join(target_dir, 'inactives')

            active_files = [os.path.join(active_dir, f) for f in os.listdir(active_dir) if f.endswith('.pt')]
            inactive_files = [os.path.join(inactive_dir, f) for f in os.listdir(inactive_dir) if f.endswith('.pt')]

            # Append files and labels
            self.voxel_files.extend(active_files)
            self.labels.extend([1] * len(active_files))  # 1 for active

            self.voxel_files.extend(inactive_files)
            self.labels.extend([0] * len(inactive_files))  # 0 for inactive

            # Append target ids
            target_id = target_dir.split('/')[-1]
            self.target_ids.extend([target_id] * (len(active_files) + len(inactive_files)))

        self.keep_channels = [1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        self.ligand_channels = [2, 15, 16, 17, 18, 19, 20]

    def __getitem__(self, idx):
        vox_file = self.voxel_files[idx]
        label = self.labels[idx]
        vox = torch.load(vox_file)
        lig = vox[self.ligand_channels].clone()
        vox[self.ligand_channels] = 0
        vox = vox[self.keep_channels]
        target = self.target_ids[idx]
        return vox, lig, label, target

    def __len__(self):
        return len(self.voxel_files) 