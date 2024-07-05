import os
import torch
from torch.utils.data import DataLoader, Dataset
import glob
import numpy as np
import pandas as pd
from docktgrid.transforms import RandomRotation
from docktgrid.molecule import MolecularComplex
from docktgrid import VoxelGrid

from src.data.docktgrid_mods import ComplexView, MolecularComplexWrapper, MolecularParserWrapper
class ComplexDataset(Dataset):
    """
    generates protein-ligand complexes on the fly
    from pdb files and voxelizes them
    """
    def __init__(self, config):
        self.config = config
        self.pdb_dir = config.pdb_dir
        self.rotate = config.rotate
        self.translation = config.translation
        self.struct_paths = self.get_complex_paths()

        self.voxelizer = VoxelGrid(
            views=[ComplexView()],
            vox_size=config.vox_size,
            box_dims=config.box_dims,
        )

    def get_complex_paths(self):
        protein_paths = ['/'.join(p.split("/")[:-1]) for p in glob.glob(f"{self.pdb_dir}/*/*_protein.pdb")]
        ligand_paths = ['/'.join(p.split("/")[:-1]) for p in glob.glob(f"{self.pdb_dir}/*/*_ligand.mol2")]
        return sorted(list(set(protein_paths).intersection(set(ligand_paths))))
    def __len__(self):
        return len(self.pdb_dir)

    def __getitem__(self, idx):
        directory = self.struct_paths[idx]
        pdb_path = glob.glob(os.path.join(directory, '*_protein.pdb'))[0]
        lig_path = glob.glob(os.path.join(directory, '*_ligand.mol2'))[0]
        complex = MolecularComplex(pdb_path, lig_path, molparser=MolecularParserWrapper())
        if self.rotate:
            rotation = RandomRotation()
            rotation(complex.coords, complex.ligand_center)
        if self.translation:
            translation_vector_length = np.random.uniform(0, self.translation)
            translation_vector = torch.tensor(
                np.random.uniform(-1, 1, 3) * translation_vector_length,
                dtype=torch.float16)
            complex.ligand_center += translation_vector
        vox = self.voxelizer.voxelize(complex)
        return vox


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



def build_loaders(config, dtype):
    path_to_split_csv = '../LP-PDBBind/dataset/LP_PDBBind.csv'
    df = pd.read_csv(path_to_split_csv)
    df.columns = ['pdb_id'] + list(df.columns)[1:]
    VoxelDataset = LigMaskDataset
    train_dataloader = DataLoader(VoxelDataset(config, fnames=df[df.new_split == 'train'].pdb_id.values, dtype=dtype),
                                  batch_size=config['batch_size'], shuffle=config.get('shufffle', True))
    val_dataloader =  DataLoader(VoxelDataset(config, fnames=df[df.new_split == 'val'].pdb_id.values,
                                              dtype=dtype, single_rotation=True),
                                 batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(VoxelDataset(config, fnames=df[df.new_split == 'test'].pdb_id.values,
                                              dtype=dtype, single_rotation=True),
                                 batch_size=config['batch_size'], shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader