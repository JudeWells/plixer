import os
import glob
import torch
import numpy as np
import pandas as pd
import time
from torch.utils.data import Dataset
import tempfile
import os
from rdkit import Chem
import pickle
import base64
from src.data.common.voxelization.config import Poc2MolDataConfig
from src.data.docktgrid_mods import MolecularParserWrapper

# from docktgrid.molecule import MolecularComplex
from docktgrid.molecule import MolecularComplex, DTYPE, ptable
from docktgrid.molparser import MolecularData, MolecularParser, Parser


from src.data.common.voxelization.voxelizer import UnifiedVoxelGrid
from src.data.common.voxelization.molecule_utils import (
    apply_random_rotation,
    apply_random_translation,
    prune_distant_atoms,
    prepare_protein_ligand_complex,
    voxelize_complex
)

import json

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
        ligand_mol_object = Chem.MolFromMol2File(lig_path)
        if ligand_mol_object is None:
            return self.__getitem__(np.random.randint(0, len(self)))
        if self.config.remove_hydrogens:
            ligand_mol_object = Chem.RemoveHs(ligand_mol_object)
        # get the smiles string
        smiles = Chem.MolToSmiles(ligand_mol_object)
        # Voxelize the complex
        protein_voxel, ligand_voxel, _ = voxelize_complex(pdb_path, lig_path, temp_config)
        
        # Return the voxelized complex
        return {
            'ligand': ligand_voxel,
            'protein': protein_voxel,
            'name': directory.split('/')[-1],
            'smiles': smiles
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

class MolecularComplex:
    """Protein-ligand molecular complex.

    If the files are already parsed, pass them as MolecularData objects.

    Attrs:
        protein_data:
            A `MolecularData` object.
        ligand_data:
            A `MolecularData` object.
        coords:
            A torch.Tensor of shape (3, n_atoms).
        n_atoms:
            An integer with the total number of atoms.
        n_atoms_protein:
            An integer with the number of protein atoms.
        n_atoms_ligand:
            An integer with the number of ligand atoms.
        element_symbols:
            A np.ndarray of shape (n_atoms,), type str.
        vdw_radii:
            A torch.Tensor of shape (n_atoms,).

    """

    def __init__(
        self,
        protein_file: str | MolecularData,
        ligand_file: str | MolecularData,
        molparser: Parser | None = MolecularParser(),
        path="",
    ):
        """Initialize MolecularComplex.

        Args:
            protein_file:
                Path to the protein file or a MolecularData object.
            ligand_file:
                Path to the ligand file or a MolecularData object.
            molparser:
                A `MolecularParser` object.
            path:
                Path to the files.
        """
        if isinstance(protein_file, MolecularData):
            self.protein_data = protein_file
        else:
            self.protein_data: MolecularData = molparser.parse_file(
                os.path.join(path, protein_file), os.path.splitext(protein_file)[1]
            )

        if isinstance(ligand_file, MolecularData):
            self.ligand_data = ligand_file
        else:
            self.ligand_data: MolecularData = molparser.parse_file(
                os.path.join(path, ligand_file), os.path.splitext(ligand_file)[1]
            )

        self.ligand_center = torch.mean(self.ligand_data.coords, 1).to(dtype=DTYPE)
        self.coords = torch.cat((self.protein_data.coords, self.ligand_data.coords), 1)
        self.n_atoms: int = self.coords.shape[1]
        self.n_atoms_protein: int = self.protein_data.coords.shape[1]
        self.n_atoms_ligand: int = self.ligand_data.coords.shape[1]

        self.element_symbols: np.ndarray[str] = np.concatenate(
            (self.protein_data.element_symbols, self.ligand_data.element_symbols)
        )
        self.vdw_radii = self._get_vdw_radii()
    def _get_vdw_radii(self):
        return torch.tensor(
            [ptable[a.title()]["vdw"] for a in self.element_symbols],
            dtype=DTYPE,
        )

class PlinderParquetDataset(Dataset):
    """
    Dataset for protein-ligand complexes from Plinder parquet files.
    Loads protein-ligand data from parquet files and voxelizes them on the fly.
    Each item represents a cluster, and a random sample from that cluster is returned.
    """
    def __init__(
        self,
        config: Poc2MolDataConfig,
        data_path: str,
        translation: float = None,
        rotate: bool = None,
        cache_size: int = 10,
    ):
        self.config = config
        self.data_path = data_path
        
        indices_dir = os.path.join(data_path, 'indices')
        
        # Load indices
        with open(os.path.join(indices_dir, 'cluster_index.json'), 'r') as f:
            self.cluster_index = json.load(f)
        
        with open(os.path.join(indices_dir, 'file_mapping.json'), 'r') as f:
            file_mapping = json.load(f)
            # Convert to full paths
            self.file_mapping = {int(k): os.path.join(data_path, v) for k, v in file_mapping.items()}
        
        # Get list of cluster IDs
        self.cluster_ids = sorted(list(self.cluster_index.keys()))
        
        # Override config values if provided
        self.random_translation = translation if translation is not None else config.random_translation
        self.random_rotation = rotate if rotate is not None else config.random_rotation
        
        # Set up channel indices
        self.ligand_channel_indices = config.ligand_channel_indices
        self.protein_channel_indices = config.protein_channel_indices
        
        # Set up maximum atom distance
        self.max_atom_dist = config.max_atom_dist
        
        # Set up LRU cache for parquet dataframes
        self.cache_size = cache_size
        self.df_cache = {}
        self.cache_order = []
        
        # Set up parser for molecular data
        self.parser = MolecularParserWrapper()
        self.fail_counter = 0
        self.fail_threshold = 10

    def _get_dataframe(self, file_idx):
        """Get dataframe from cache or load it."""
        if file_idx in self.df_cache:
            # Move to the end of cache order (most recently used)
            self.cache_order.remove(file_idx)
            self.cache_order.append(file_idx)
            return self.df_cache[file_idx]
        
        # Load the dataframe
        df = pd.read_parquet(self.file_mapping[file_idx])
        # Add to cache
        self.df_cache[file_idx] = df
        self.cache_order.append(file_idx)
        
        # Manage cache size
        if len(self.cache_order) > self.cache_size:
            oldest_idx = self.cache_order.pop(0)
            del self.df_cache[oldest_idx]
        
        return df
    


    def _deserialize_molecular_data(self, serialized_data):
        """Deserialize MolecularData object from a base64 encoded string."""
        return pickle.loads(base64.b64decode(serialized_data))

    def __len__(self):
        return len(self.cluster_ids)

    def __getitem__(self, idx):
        """Get a random sample from the specified cluster."""
        # Get the cluster ID
        cluster_id = self.cluster_ids[idx]
        
        # Get samples for this cluster
        cluster_samples = self.cluster_index[cluster_id]
        
        # Select a random sample from this cluster
        sample_info = np.random.choice(cluster_samples)
        
        file_idx = sample_info['file_idx']
        row_idx = sample_info['row_idx']
        
        # Option 1: Get the full dataframe (original method)
        start_time = time.time()
        df = self._get_dataframe(file_idx)
        row = df.iloc[row_idx]
        end_time = time.time()
        load_time = end_time - start_time

        
        try:
            # Check if we have the new format columns
            if 'protein_coords' in row and 'ligand_coords' in row:
                print("labeled protein coords shape", row['protein_coords_shape'])
                print("actual protein coords shape", row['protein_coords'].shape)
                print("labeled ligand coords shape", row['ligand_coords_shape'])
                print("actual ligand coords shape", row['ligand_coords'].shape)
                protein_coords = torch.tensor(row['protein_coords'], dtype=torch.float16).reshape(
                    list(row['protein_coords_shape'])
                    )
                ligand_coords = torch.tensor(row['ligand_coords'], dtype=torch.float16).reshape(
                    list(row['ligand_coords_shape'])
                )
                
                protein_data = MolecularData(
                    molecule_object=None,
                    coords=protein_coords,
                    element_symbols=row['protein_element_symbols']
                )
                
                ligand_data = MolecularData(
                    molecule_object=None,
                    coords=ligand_coords,
                    element_symbols=row['ligand_element_symbols']
                )

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
                
                # Create molecular complex
                complex_obj = MolecularComplex(protein_data, ligand_data)
                
                # Apply transformations
                if self.random_rotation:
                    assert abs(complex_obj.ligand_data.coords.mean(axis=1) - complex_obj.ligand_center).max() < 1e-5, "Ligand center is not correct"
                    complex_obj = apply_random_rotation(complex_obj)
                    assert abs(complex_obj.ligand_data.coords.mean(axis=1) - complex_obj.ligand_center).max() < 1e-5, "Ligand center is not correct"
                
                if self.random_translation > 0:
                    complex_obj = apply_random_translation(complex_obj, self.random_translation)
                
                if self.max_atom_dist is not None and self.max_atom_dist > 0:
                    assert abs(complex_obj.ligand_data.coords.mean(axis=1) - complex_obj.ligand_center).max() <= self.random_translation, "Ligand center is not correct"
                    complex_obj = prune_distant_atoms(complex_obj, self.max_atom_dist)
                
                # Voxelize the complex
                voxelizer = UnifiedVoxelGrid(temp_config)
                voxel = voxelizer.voxelize(complex_obj)
                
                # Extract protein and ligand channels based on config
                if temp_config.has_protein:
                    protein_channels = len(temp_config.protein_channels)
                    protein_voxel = voxel[:protein_channels]
                    ligand_voxel = voxel[protein_channels:]
                else:
                    protein_voxel = None
                    ligand_voxel = voxel
                
                self.fail_counter = 0
                # Return the voxelized complex
                return {
                    'ligand': ligand_voxel,
                    'protein': protein_voxel,
                    'name': row['system_id'],
                    'smiles': row['smiles'],
                    'cluster': cluster_id,
                    'load_time': load_time
                }
            
            # Keep existing handling for old format as fallback
            elif 'protein_data_serialized' in row and 'ligand_data_serialized' in row:
                # Deserialize the data
                protein_data = self._deserialize_molecular_data(row['protein_data_serialized'])
                ligand_data = self._deserialize_molecular_data(row['ligand_data_serialized'])
                
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
                
                # Create molecular complex
                complex_obj = MolecularComplex(protein_data, ligand_data)
                
                # Apply transformations
                if self.random_rotation:
                    assert abs(complex_obj.ligand_data.coords.mean(axis=1) - complex_obj.ligand_center).max() < 1e-3, "Ligand center is not correct"
                    complex_obj = apply_random_rotation(complex_obj)
                    assert abs(complex_obj.ligand_data.coords.mean(axis=1) - complex_obj.ligand_center).max() < 1e-3, f"Ligand center is not correct: {complex_obj.ligand_data.coords.mean(axis=1)} {complex_obj.ligand_center}"
                
                if self.random_translation > 0:
                    complex_obj = apply_random_translation(complex_obj, self.random_translation)
                
                if self.max_atom_dist is not None and self.max_atom_dist > 0:
                    assert abs(complex_obj.ligand_data.coords.mean(axis=1) - complex_obj.ligand_center).max() <= self.random_translation, "Ligand center is not correct"
                    complex_obj = prune_distant_atoms(complex_obj, self.max_atom_dist)
                
                # Voxelize the complex
                voxelizer = UnifiedVoxelGrid(temp_config)
                voxel = voxelizer.voxelize(complex_obj)
                
                # Extract protein and ligand channels based on config
                if temp_config.has_protein:
                    protein_channels = len(temp_config.protein_channels)
                    protein_voxel = voxel[:protein_channels]
                    ligand_voxel = voxel[protein_channels:]
                else:
                    protein_voxel = None
                    ligand_voxel = voxel
                
                self.fail_counter = 0
                # Return the voxelized complex
                return {
                    'ligand': ligand_voxel,
                    'protein': protein_voxel,
                    'name': row['system_id'],
                    'smiles': row['smiles'],
                    'cluster': cluster_id,
                    'load_time': load_time
                }
            
            else:
                # Fall back to using temporary files if preprocessed data is not available
                # Get protein and ligand data
                pdb_content = row['pdb_content']
                mol_block = row['mol_block']
                
                # Create temporary files for the protein and ligand
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Write protein to PDB file
                    protein_path = os.path.join(temp_dir, "protein.pdb")
                    with open(protein_path, 'wb') as f:
                        f.write(pdb_content)
                    
                    # Write ligand to MOL file
                    ligand_path = os.path.join(temp_dir, "ligand.mol")
                    with open(ligand_path, 'wb') as f:
                        f.write(mol_block)
                    
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
                    
                    # Convert MOL to RDKit molecule
                    ligand_mol = Chem.MolFromMolBlock(mol_block.decode())
                    if ligand_mol is None:
                        return self.__getitem__(np.random.randint(0, len(self)))
                    
                    if self.config.remove_hydrogens:
                        ligand_mol = Chem.RemoveHs(ligand_mol)
                    
                    # Voxelize the complex
                    protein_voxel, ligand_voxel, _ = voxelize_complex(protein_path, ligand_path, temp_config)
                    
                    # Return the voxelized complex
                    self.fail_counter = 0
                    return {
                        'ligand': ligand_voxel,
                        'protein': protein_voxel,
                        'name': row['system_id'],
                        'smiles': row['smiles'],
                        'cluster': cluster_id,
                        'load_time': load_time
                    }
                    
        except Exception as e:
            print(f"Error processing sample {row['system_id']} from cluster {cluster_id}: {e}")
            # Return a random sample instead
            self.fail_counter += 1
            if self.fail_counter > self.fail_threshold:
                raise e
            return self.__getitem__(np.random.randint(0, len(self))) 