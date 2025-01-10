import os
import torch
from torch.utils.data import DataLoader, Dataset
import glob
import numpy as np
import pandas as pd
from docktgrid.transforms import RandomRotation
from docktgrid.molecule import MolecularComplex
from docktgrid import VoxelGrid

from src.data.docktgrid_mods import (
    ComplexView, 
    MolecularParserWrapper, 
    UnifiedAtomView,
    ProteinComplex
)

class ComplexDataset(Dataset):
    """
    generates protein-ligand complexes on the fly
    from pdb files and voxelizes them
    """
    def __init__(self, config, pdb_dir, rotate=True, translation=0.0):
        self.config = config
        self.pdb_dir = pdb_dir
        self.rotate = rotate
        self.random_translation = translation
        self.struct_paths = self.get_complex_paths()
        self.ligand_channels = config.ligand_channels
        self.protein_channels = config.protein_channels
        self.max_atom_dist = config.max_atom_dist

        self.voxelizer = VoxelGrid(
            views=[ComplexView(config.vox_config)],
            vox_size=config.vox_config.vox_size,
            box_dims=config.vox_config.box_dims,
        )

    def get_complex_paths(self):
        protein_paths = ['/'.join(p.split("/")[:-1]) for p in glob.glob(f"{self.pdb_dir}/*/*_protein.pdb")]
        ligand_paths = ['/'.join(p.split("/")[:-1]) for p in glob.glob(f"{self.pdb_dir}/*/*_ligand.mol2")]
        return sorted(list(set(protein_paths).intersection(set(ligand_paths))))

    def prune_far_atoms(self, complex: MolecularComplex):
        ligand_center = complex.ligand_center
        dists = torch.linalg.vector_norm(
            complex.coords.T - ligand_center, dim=1
        )
        mask = dists < self.max_atom_dist
        complex.coords = complex.coords[:, mask]
        complex.vdw_radii = complex.vdw_radii[mask]
        complex.element_symbols = complex.element_symbols[mask]
        complex.n_atoms = complex.coords.shape[1]

        lig_dists = torch.linalg.vector_norm(
            complex.ligand_data.coords.T - ligand_center, dim=1
        )
        mask = lig_dists < self.max_atom_dist
        complex.ligand_data.coords = complex.ligand_data.coords[:, mask]
        complex.ligand_data.element_symbols = complex.ligand_data.element_symbols[mask.numpy()]
        complex.n_atoms_ligand = complex.ligand_data.coords.shape[1]

        prot_dists = torch.linalg.vector_norm(
            complex.protein_data.coords.T - ligand_center, dim=1
        )
        mask = prot_dists < self.max_atom_dist
        complex.protein_data.coords = complex.protein_data.coords[:, mask]
        complex.protein_data.element_symbols = complex.protein_data.element_symbols[mask]
        complex.n_atoms_protein = complex.protein_data.coords.shape[1]

        return complex
    def __len__(self):
        return len(self.struct_paths)

    def __getitem__(self, idx):
        directory = self.struct_paths[idx]
        pdb_path = glob.glob(os.path.join(directory, '*_protein.pdb'))[0]
        lig_path = glob.glob(os.path.join(directory, '*_ligand.mol2'))[0]
        try:
            complex = MolecularComplex(pdb_path, lig_path, molparser=MolecularParserWrapper())
        except:
            bp=1
            complex = MolecularComplex(pdb_path, lig_path, molparser=MolecularParserWrapper())
        if self.max_atom_dist:
            complex = self.prune_far_atoms(complex)
        if self.rotate:
            rotation = RandomRotation()
            rotation(complex.coords, complex.ligand_center)
        if self.random_translation:
            translation_vector_length = np.random.uniform(0, self.random_translation)
            translation_vector = torch.tensor(
                np.random.uniform(-1, 1, 3) * translation_vector_length,
                dtype=torch.float16)
            complex.ligand_center += translation_vector

        vox = self.voxelizer.voxelize(complex)
        lig_vox = vox[self.ligand_channels]
        prot_vox = vox[self.protein_channels]
        return {
            'ligand': lig_vox,
            'protein': prot_vox,
            'name': directory.split('/')[-1]
        }


class DockstringTestDataset(ComplexDataset):
    """
    Inherit from LigMaskDataset and override __init__
    to use a different directory structure. This version also tracks whether
    each ligand is active or inactive based on the directory structure and returns
    this label along with the voxel data.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.voxel_dir = config['dockstring_test_dir']
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

        self.keep_channels = [1,2,9,10,11,12,13,14,15,16,17,18,19,20]
        self.ligand_channels = [2,15,16,17,18,19,20]

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


class StructuralPretrainDataset(Dataset):
    """
    Dataset for pre-training structural autoencoder.
    Generates voxelizations centered on random protein atoms.
    Uses UnifiedAtomView to create unified atom-type channels.
    """
    def __init__(self, config, pdb_dir, rotate=True):
        self.config = config
        self.pdb_dir = pdb_dir
        self.rotate = config.vox_config.random_rotation
        self.struct_paths = self.get_structure_paths()
        self.max_atom_dist = config.max_atom_dist

        # Initialize UnifiedAtomView with specified channels
        self.unified_view = UnifiedAtomView(
            element_channels={
                0: ["C"],    # Carbon channel
                1: ["N"],    # Nitrogen channel
                2: ["O"],    # Oxygen channel
                3: ["S"],    # Sulfur channel
                4: ["P"],    # Phosphorus channel
                5: ["F", "Cl", "Br", "I"],  # Halogens channel
                6: ["Zn", "Fe", "Mg", "Ca", "Na", "K"],  # Metals channel
                7: ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", 
                    "Zn", "Fe", "Mg", "Ca", "Na", "K"]  # All atoms channel
            }
        )

        self.voxelizer = VoxelGrid(
            views=[self.unified_view],
            vox_size=config.vox_config.vox_size,
            box_dims=config.vox_config.box_dims,
        )

    def get_structure_paths(self):
        """Get all PDB files in the directory."""
        return glob.glob(f"{self.pdb_dir}/**/*.pdb", recursive=True)

    def select_random_center(self, complex: MolecularComplex) -> torch.Tensor:
        """Select a random protein atom as the center."""
        protein_coords = complex.protein_data.coords
        num_atoms = protein_coords.shape[1]
        random_idx = torch.randint(0, num_atoms, (1,))
        return protein_coords[:, random_idx].squeeze()

    def prune_far_atoms(self, complex: MolecularComplex, center: torch.Tensor):
        """Remove atoms beyond max_atom_dist from the center."""
        dists = torch.linalg.vector_norm(
            complex.coords.T - center, dim=1
        )
        mask = dists < self.max_atom_dist
        complex.coords = complex.coords[:, mask]
        complex.vdw_radii = complex.vdw_radii[mask]
        complex.element_symbols = complex.element_symbols[mask]
        complex.n_atoms = complex.coords.shape[1]
        return complex

    def __len__(self):
        return len(self.struct_paths)

    def __getitem__(self, idx):
        pdb_path = self.struct_paths[idx]
        try:
            # Load structure as ProteinComplex instead of MolecularComplex
            complex = ProteinComplex(pdb_path, molparser=MolecularParserWrapper())
            
            # Select random center
            center = self.select_random_center(complex)
            
            # Set the center as ligand_center (required by voxelizer)
            complex.ligand_center = center
            
            # Prune distant atoms
            if self.max_atom_dist:
                complex = self.prune_far_atoms(complex, center)
            
            # Apply random rotation if specified
            if self.rotate:
                rotation = RandomRotation()
                rotation(complex.coords, center)

            # Voxelize the structure
            vox = self.voxelizer.voxelize(complex)
            
            return {
                'input': vox.cpu(),  # Input is same as target for autoencoder
                'name': os.path.basename(pdb_path)
            }
            
        except Exception as e:
            print(f"Error processing {pdb_path}: {str(e)}")
            # Return first item as fallback
            return self.__getitem__(0)


def build_loaders(config, dtype):
    path_to_split_csv = '../LP-PDBBind/dataset/LP_PDBBind.csv'
    df = pd.read_csv(path_to_split_csv)
    df.columns = ['pdb_id'] + list(df.columns)[1:]
    train_dataloader = DataLoader(VoxelDataset(config, fnames=df[df.new_split == 'train'].pdb_id.values, dtype=dtype),
                                  batch_size=config['batch_size'], shuffle=config.get('shufffle', True))
    val_dataloader =  DataLoader(VoxelDataset(config, fnames=df[df.new_split == 'val'].pdb_id.values,
                                              dtype=dtype, single_rotation=True),
                                 batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(VoxelDataset(config, fnames=df[df.new_split == 'test'].pdb_id.values,
                                              dtype=dtype, single_rotation=True),
                                 batch_size=config['batch_size'], shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader