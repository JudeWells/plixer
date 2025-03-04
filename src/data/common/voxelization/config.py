from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import torch


@dataclass
class VoxelizationConfig:
    """
    Unified configuration for voxelization of molecules and protein-ligand complexes.
    This ensures consistent voxelization parameters across different models.
    """
    # Basic voxelization parameters
    vox_size: float = 0.75
    box_dims: List[float] = field(default_factory=lambda: [24.0, 24.0, 24.0])
    
    # Rotation and translation parameters
    random_rotation: bool = True
    random_translation: float = 6.0
    
    # Channel configuration
    has_protein: bool = True
    
    # Channel names for better interpretability
    ligand_channel_names: List[str] = field(default_factory=lambda: [
        "carbon", "hydrogen", "oxygen", "nitrogen", "sulfur", 
        "chlorine", "fluorine", "iodine", "bromine", "all"
    ])
    
    protein_channel_names: List[str] = field(default_factory=lambda: [
        "carbon", "hydrogen", "oxygen", "nitrogen", "sulfur", "all"
    ])
    
    # Channel mappings (which elements go into which channel)
    protein_channels: Dict[int, List[str]] = field(default_factory=lambda: {
        0: ["C"],
        1: ["H"],
        2: ["O"],
        3: ["N"],
        4: ["S"],
        5: ["C", "H", "O", "N", "S"]
    })
    
    ligand_channels: Dict[int, List[str]] = field(default_factory=lambda: {
        0: ["C"],
        1: ["H"],
        2: ["O"],
        3: ["N"],
        4: ["S"],
        5: ["Cl"],
        6: ["F"],
        7: ["I"],
        8: ["Br"],
        9: ["C", "H", "O", "N", "S", "Cl", "F", "I", "Br"]
    })
    
    # Maximum atom distance from ligand center (for pruning distant atoms)
    max_atom_dist: Optional[float] = 32.0
    
    # Data type for tensors
    dtype: torch.dtype = torch.float16


@dataclass
class Poc2MolDataConfig(VoxelizationConfig):
    """
    Configuration specific to the Poc2Mol data pipeline.
    Extends the base VoxelizationConfig with Poc2Mol-specific parameters.
    """
    batch_size: int = 32
    
    # Indices of channels to use for ligand and protein
    # These are indices into the voxelized output, not the channel mappings above
    ligand_channel_indices: List[int] = field(default_factory=lambda: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    protein_channel_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    
    # Optional list of specific filenames to use
    fnames: Optional[List[str]] = None


@dataclass
class Vox2SmilesDataConfig(VoxelizationConfig):
    """
    Configuration specific to the Vox2Smiles data pipeline.
    Extends the base VoxelizationConfig with Vox2Smiles-specific parameters.
    """
    batch_size: int = 24
    num_workers: int = 0
    max_smiles_len: int = 200
    
    # For Vox2Smiles, we typically don't need protein channels
    has_protein: bool = False 
    include_hydrogens: bool = True