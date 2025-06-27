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

from docktgrid.molecule import MolecularComplex
# from docktgrid.molecule import MolecularComplex, DTYPE, ptable
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
from collections import defaultdict
from tqdm import tqdm

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
        
        # Set up maximum atom distance for pruning how much of protein is voxelized
        self.max_atom_dist = config.get('max_atom_dist', config.box_dims[0])

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


class ParquetDataset(Dataset):
    """
    Dataset for protein-ligand complexes from parquet files (eg. Plinder or HiQBind).
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
        use_cluster_member_zero: bool = False
    ):
        self.config = config
        self.data_path = data_path
        self.use_cluster_member_zero = use_cluster_member_zero
        indices_dir = os.path.join(data_path, 'indices')
        
        cluster_index_path = os.path.join(indices_dir, 'cluster_index.json')
        if os.path.exists(cluster_index_path):
            with open(os.path.join(indices_dir, 'cluster_index.json'), 'r') as f:
                self.cluster_index = json.load(f)
        else:
            #create cluster index
            self.cluster_index = self._create_indices(indices_dir)
            
        file_mapping_path = os.path.join(indices_dir, 'file_mapping.json')
        if os.path.exists(file_mapping_path):
            with open(file_mapping_path, 'r') as f:
                file_mapping = json.load(f)
            # Convert to full paths
            self.file_mapping = {int(k): os.path.join(data_path, v) for k, v in file_mapping.items()}
        else:
            #create file mapping
            if not hasattr(self, 'file_mapping'):
                # If indices were not created above, create file mapping separately
                self.file_mapping = self._create_file_mapping(indices_dir, data_path)
        # Get list of cluster IDs
        if self.config.system_ids is not None:
            filtered_cluster_index  = {}
            for cluster_id, samples in self.cluster_index.items():
                filtered_samples = [sample for sample in samples if sample['system_id'] in self.config.system_ids]
                if filtered_samples:
                    filtered_cluster_index[cluster_id] = filtered_samples
            self.cluster_index = filtered_cluster_index
        self.cluster_ids = sorted(list(self.cluster_index.keys()))
        
        # Override config values if provided
        self.random_translation = translation if translation is not None else config.random_translation
        self.random_rotation = rotate if rotate is not None else config.random_rotation
        
        # Set up channel indices
        self.ligand_channel_indices = config.ligand_channel_indices
        self.protein_channel_indices = config.protein_channel_indices
        
        # Set up maximum atom distance for pruning how much of protein is voxelized
        self.max_atom_dist = config.get('max_atom_dist', config.box_dims[0])
        
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
        
        if self.use_cluster_member_zero:
            sample_info = cluster_samples[0]
        else:
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
            dtype = eval(self.config.dtype) if isinstance(self.config.dtype, str) else self.config.dtype
            if 'protein_coords' in row and 'ligand_coords' in row:
                protein_coords = torch.tensor(row['protein_coords'], dtype=dtype).reshape(
                    list(row['protein_coords_shape'])
                    )
                ligand_coords = torch.tensor(row['ligand_coords'], dtype=dtype).reshape(
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

                # TODO see if we can remove this
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
                    assert abs(complex_obj.ligand_data.coords.mean(axis=1) - complex_obj.ligand_center).max() < 1, "Ligand center is not correct"
                    complex_obj = apply_random_rotation(complex_obj)
                    assert abs(complex_obj.ligand_data.coords.mean(axis=1) - complex_obj.ligand_center).max() < 1, "Ligand center is not correct"
                
                if self.random_translation > 0:
                    complex_obj = apply_random_translation(complex_obj, self.random_translation)
                
                if self.max_atom_dist is not None and self.max_atom_dist > 0:
                    assert abs(complex_obj.ligand_data.coords.mean(axis=1) - complex_obj.ligand_center).max() <= self.random_translation + 1, "Ligand center is not correct"
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
                    assert abs(complex_obj.ligand_data.coords.mean(axis=1) - complex_obj.ligand_center).max() < 1, "Ligand center is not correct"
                    complex_obj = apply_random_rotation(complex_obj)
                    assert abs(complex_obj.ligand_data.coords.mean(axis=1) - complex_obj.ligand_center).max() < 1, f"Ligand center is not correct: {complex_obj.ligand_data.coords.mean(axis=1)} {complex_obj.ligand_center}"
                
                if self.random_translation > 0:
                    complex_obj = apply_random_translation(complex_obj, self.random_translation)
                
                if self.max_atom_dist is not None and self.max_atom_dist > 0:
                    assert abs(complex_obj.ligand_data.coords.mean(axis=1) - complex_obj.ligand_center).max() <= self.random_translation + 1, "Ligand center is not correct"
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

    def _create_indices(self, indices_dir):
        """
        Create index files for faster dataset loading.
        
        Args:
            indices_dir: Directory to save index files
        
        Returns:
            dict: The cluster index
        """
        # Create output directory if it doesn't exist
        os.makedirs(indices_dir, exist_ok=True)
        
        # Get all parquet files in the data path
        parquet_files = glob.glob(os.path.join(self.data_path, "*.parquet"))
        
        if not parquet_files:
            raise ValueError(f"No parquet files found in {self.data_path}")
        
        print(f"Found {len(parquet_files)} parquet files to process")
        
        # Create global indices
        all_samples = []
        file_indices = []
        
        # Create cluster-based indices
        cluster_samples = defaultdict(list)
        
        # Process each parquet file
        for file_idx, file_path in enumerate(tqdm(parquet_files, desc="Processing files")):
            try:
                # Read the parquet file
                chunk_df = pd.read_parquet(file_path)
                
                # Process each row
                for row_idx, row in chunk_df.iterrows():
                    # Get cluster ID (default to '0' if not present)
                    cluster_id = str(row.get('cluster', '0'))
                    
                    sample = {
                        'system_id': row['system_id'],
                        'cluster': cluster_id,
                    }
                    
                    all_samples.append(sample)
                    file_indices.append(file_idx)
                    
                    # Add to cluster-based indices
                    cluster_samples[cluster_id].append({
                        'file_idx': file_idx,
                        'system_id': sample['system_id'],
                        'row_idx': int(row_idx)
                    })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Save global indices
        global_index = {
            'samples': all_samples,
            'file_indices': file_indices
        }
        
        with open(os.path.join(indices_dir, 'global_index.json'), 'w') as f:
            json.dump(global_index, f, indent=2)
        
        # Save cluster-based indices
        cluster_index = {str(cluster_id): samples for cluster_id, samples in cluster_samples.items()}
        
        with open(os.path.join(indices_dir, 'cluster_index.json'), 'w') as f:
            json.dump(cluster_index, f, indent=2)
        
        # Save file mapping
        file_mapping = {
            i: os.path.relpath(file_path, self.data_path) for i, file_path in enumerate(parquet_files)
        }
        
        with open(os.path.join(indices_dir, 'file_mapping.json'), 'w') as f:
            json.dump(file_mapping, f, indent=2)
        
        # Generate summary
        summary = {
            'total_samples': len(all_samples),
            'total_clusters': len(cluster_samples),
            'total_files': len(parquet_files),
            'clusters': {cluster: len(samples) for cluster, samples in cluster_samples.items()}
        }
        
        with open(os.path.join(indices_dir, 'index_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Created indices with {len(all_samples)} total samples across {len(cluster_samples)} clusters")
        
        # Set up file mapping for the current instance
        self.file_mapping = {int(k): os.path.join(self.data_path, v) for k, v in file_mapping.items()}
        
        return cluster_index
    
    def _create_file_mapping(self, indices_dir, data_path):
        """
        Create file mapping if it doesn't exist.
        
        Args:
            indices_dir: Directory to save index files
            data_path: Base directory for data files
            
        Returns:
            dict: The file mapping with full paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(indices_dir, exist_ok=True)
        
        # Get all parquet files in the data path
        parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
        
        if not parquet_files:
            raise ValueError(f"No parquet files found in {data_path}")
        
        # Create file mapping
        file_mapping = {
            i: os.path.relpath(file_path, data_path) for i, file_path in enumerate(parquet_files)
        }
        
        # Save file mapping
        with open(os.path.join(indices_dir, 'file_mapping.json'), 'w') as f:
            json.dump(file_mapping, f, indent=2)
        
        # Return file mapping with full paths
        return {int(k): os.path.join(data_path, v) for k, v in file_mapping.items()} 