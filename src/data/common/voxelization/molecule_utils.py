import os
import pickle
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from docktgrid.transforms import RandomRotation
from docktgrid.molecule import MolecularComplex

from src.data.common.voxelization.voxelizer import RDkitMolecularComplex, UnifiedVoxelGrid
from src.data.common.voxelization.config import VoxelizationConfig


def load_mol_from_pickle(path):
    """Load an RDKit molecule from a pickle file."""
    with open(path, "rb") as f:
        mol_data = pickle.load(f)
    
    # If the pickle contains multiple conformers, use the first one
    if "conformers" in mol_data:
        mol = mol_data["conformers"][0]["rd_mol"]
    else:
        mol = mol_data["rd_mol"]
    
    return mol


def load_complex_from_files(protein_path, ligand_path, parser=None):
    """Load a protein-ligand complex from PDB and MOL2 files."""
    from src.data.docktgrid_mods import MolecularParserWrapper
    
    if parser is None:
        parser = MolecularParserWrapper()
    
    return MolecularComplex(protein_path, ligand_path, molparser=parser)


def apply_random_rotation(molecular_complex):
    """Apply a random rotation to a molecular complex."""
    rotation = RandomRotation()
    rotation(molecular_complex.coords, molecular_complex.ligand_center)
    return molecular_complex


def apply_random_translation(molecular_complex, max_translation):
    """Apply a random translation to a molecular complex."""
    if max_translation <= 0:
        return molecular_complex
    
    translation_vector_length = np.random.uniform(0, max_translation)
    translation_vector = torch.tensor(
        np.random.uniform(-1, 1, 3) * translation_vector_length,
        dtype=torch.float16
    )
    molecular_complex.ligand_center += translation_vector
    return molecular_complex


def prune_distant_atoms(complex_obj, max_atom_dist):
    """Remove atoms that are too far from the ligand center."""
    if max_atom_dist is None or max_atom_dist <= 0:
        return complex_obj
    
    ligand_center = complex_obj.ligand_center
    
    # Prune atoms in the entire complex
    dists = torch.linalg.vector_norm(
        complex_obj.coords.T - ligand_center, dim=1
    )
    mask = dists < max_atom_dist
    complex_obj.coords = complex_obj.coords[:, mask]
    complex_obj.vdw_radii = complex_obj.vdw_radii[mask]
    complex_obj.element_symbols = complex_obj.element_symbols[mask]
    complex_obj.n_atoms = complex_obj.coords.shape[1]

    # Prune ligand atoms
    lig_dists = torch.linalg.vector_norm(
        complex_obj.ligand_data.coords.T - ligand_center, dim=1
    )
    lig_mask = lig_dists < max_atom_dist
    complex_obj.ligand_data.coords = complex_obj.ligand_data.coords[:, lig_mask]
    
    # Handle element symbols differently based on type
    if isinstance(complex_obj.ligand_data.element_symbols, (np.ndarray, pd.Series)):
        complex_obj.ligand_data.element_symbols = complex_obj.ligand_data.element_symbols[lig_mask.numpy()]
    else:
        complex_obj.ligand_data.element_symbols = complex_obj.ligand_data.element_symbols[lig_mask]
    
    complex_obj.n_atoms_ligand = complex_obj.ligand_data.coords.shape[1]

    # If there are protein atoms, prune them too
    if complex_obj.n_atoms_protein > 0:
        prot_dists = torch.linalg.vector_norm(
            complex_obj.protein_data.coords.T - ligand_center, dim=1
        )
        prot_mask = prot_dists < max_atom_dist
        complex_obj.protein_data.coords = complex_obj.protein_data.coords[:, prot_mask]
        complex_obj.protein_data.element_symbols = complex_obj.protein_data.element_symbols[prot_mask]
        complex_obj.n_atoms_protein = complex_obj.protein_data.coords.shape[1]

    return complex_obj


def prepare_rdkit_molecule(mol, config):
    """Prepare an RDKit molecule for voxelization."""
    # Convert to our molecular complex format
    molecular_complex = RDkitMolecularComplex(mol)
    
    # Apply transformations
    if config.random_rotation:
        molecular_complex = apply_random_rotation(molecular_complex)
    
    if config.random_translation > 0:
        molecular_complex = apply_random_translation(molecular_complex, config.random_translation)
    
    if config.max_atom_dist is not None and config.max_atom_dist > 0:
        molecular_complex = prune_distant_atoms(molecular_complex, config.max_atom_dist)
    
    return molecular_complex


def prepare_protein_ligand_complex(protein_path, ligand_path, config):
    """Prepare a protein-ligand complex for voxelization."""
    # Load the complex
    complex_obj = load_complex_from_files(protein_path, ligand_path)
    
    # Apply transformations
    if config.random_rotation:
        complex_obj = apply_random_rotation(complex_obj)
    
    if config.random_translation > 0:
        complex_obj = apply_random_translation(complex_obj, config.random_translation)
    
    if config.max_atom_dist is not None and config.max_atom_dist > 0:
        complex_obj = prune_distant_atoms(complex_obj, config.max_atom_dist)
    
    return complex_obj


def voxelize_molecule(mol, config):
    """Voxelize an RDKit molecule using the unified voxelizer."""
    # Prepare the molecule
    molecular_complex = prepare_rdkit_molecule(mol, config)
    
    # Create the voxelizer
    voxelizer = UnifiedVoxelGrid(config)
    
    # Voxelize the molecule
    voxel = voxelizer.voxelize(molecular_complex)
    
    return voxel


def voxelize_complex(protein_path, ligand_path, config):
    """Voxelize a protein-ligand complex using the unified voxelizer."""
    # Prepare the complex
    complex_obj = prepare_protein_ligand_complex(protein_path, ligand_path, config)
    
    # Create the voxelizer
    voxelizer = UnifiedVoxelGrid(config)
    
    # Voxelize the complex
    voxel = voxelizer.voxelize(complex_obj)
    
    # Extract protein and ligand channels based on config
    if config.has_protein:
        protein_channels = len(config.protein_channels)
        protein_voxel = voxel[:protein_channels]
        ligand_voxel = voxel[protein_channels:]
    else:
        protein_voxel = None
        ligand_voxel = voxel
    
    return protein_voxel, ligand_voxel, complex_obj 