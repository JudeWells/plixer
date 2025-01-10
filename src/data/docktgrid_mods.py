import os
import subprocess
import re
import argparse
import shutil
import gzip
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from docktgrid import VoxelGrid
from docktgrid.view import BasicView, VolumeView, View
from docktgrid.molecule import MolecularComplex, ptable, DTYPE
from docktgrid.molparser import MolecularData, MolecularParser, Parser
from docktgrid.transforms import RandomRotation, Transform
from torch import save as torch_save
import torch

ptable_upper = {k.upper(): k for k in ptable.keys()}
class MolecularParserWrapper(MolecularParser):
    def __init__(self):
        super().__init__()
    def get_element_symbols_mol2(self) -> list[str]:
        symbols = self.df_atom['atom_type'].apply(lambda x: x.split(".")[0])
        return symbols


    def get_element_symbols_pdb(self) -> list[str]:
        """
        fixes PDBs with missing element symbols
        """
        mask_atm = self.df_atom["element_symbol"] == ''
        if mask_atm.any():
            self.df_atom.loc[mask_atm, "element_symbol"] = self.df_atom.loc[mask_atm, "atom_name"].apply(
                self.extract_element_symbol_from_name
            )
        mask_het = self.df_hetatm["element_symbol"] == ''
        if mask_het.any():
            self.df_hetatm.loc[mask_het, "element_symbol"] = self.df_hetatm.loc[mask_het, "atom_name"].apply(
                self.extract_element_symbol_from_name
            )
        return super().get_element_symbols_pdb()

    def extract_element_symbol_from_name(self, name: str) -> str:
        # remove numeric and return first character
        chars = re.sub(r'\d', '', name)[:2].upper()
        if chars[0] not in ptable_upper and chars in ptable_upper:
            return ptable_upper[chars]
        else:
            return chars[0]

class ComplexView(View):

    def __init__(self, vox_config: DictConfig):
        super().__init__()
        self.cfg = vox_config

    def get_num_channels(self):
        return len(self.cfg.protein_channels) + len(self.cfg.ligand_channels)

    def get_channels_names(self):
        return (
                + [f"{ch}_protein" for ch in self.cfg.protein_channel_names]
                + [f"{ch}_ligand" for ch in self.cfg.ligand_channel_names]
        )

    def get_molecular_complex_channels(
            self, molecular_complex: MolecularComplex
    ) -> torch.Tensor:
        """Set of channels for all atoms."""
        raise NotImplementedError("This method should not be called.")
        # get a list of bools representing each atom in each channel
        symbs = molecular_complex.element_symbols
        chs = np.asarray(
            [np.isin(
                symbs,
                self.cfg.ligand_channel_names[c]
            ) for c in range(len(self.cfg.ligand_channels))]
        )

        # invert bools in last channel, since it represents any atom except those explicitly defined
        np.invert(chs[-1], out=chs[-1])
        return torch.from_numpy(chs)

    def get_ligand_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Set of channels for ligand atoms."""
        symbs = molecular_complex.element_symbols
        chs = np.asarray(
            [np.isin(
                symbs,
                self.cfg.ligand_channels[c]
            ) for c in range(len(self.cfg.ligand_channels))])
        chs[..., : -molecular_complex.n_atoms_ligand] = False
        np.invert(chs[-1], out=chs[-1])
        return torch.from_numpy(chs)

    def get_protein_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Set of channels for protein atoms."""
        symbs = molecular_complex.element_symbols
        chs = np.asarray(
            [np.isin(symbs, self.cfg.protein_channels[c]
            ) for c in range(len(self.cfg.protein_channels))])
        chs[..., -molecular_complex.n_atoms_ligand:] = False
        np.invert(chs[-1], out=chs[-1])
        return torch.from_numpy(chs)

    def __call__(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Concatenate all channels in a single tensor.

        Args:
            molecular_complex: MolecularComplex object.

        Returns:
            A boolean torch.Tensor array with shape
            (num_of_channels_defined_for_this_view, n_atoms_complex)

        """
        protein = self.get_protein_channels(molecular_complex)
        ligand = self.get_ligand_channels(molecular_complex)
        return torch.cat(
            (
                protein if protein is not None else torch.tensor([], dtype=torch.bool),
                ligand if ligand is not None else torch.tensor([], dtype=torch.bool),
            ),
        ).bool()

class UnifiedAtomView(View):
    """
    A view that creates channels based on atom types, treating protein and ligand atoms the same.
    Each channel represents a specific element type (C, H, O, N, S, etc.) regardless of whether 
    the atom belongs to the protein or ligand.
    """
    
    def __init__(self, element_channels=None):
        super().__init__()
        # Default channels if none provided
        self.element_channels = element_channels or {
            0: ["C"],    # Carbon channel
            1: ["H"],    # Hydrogen channel
            2: ["O"],    # Oxygen channel
            3: ["N"],    # Nitrogen channel
            4: ["S"],    # Sulfur channel
            5: ["Cl"],   # Chlorine channel
            6: ["F"],    # Fluorine channel
            7: ["I"],    # Iodine channel
            8: ["Br"],   # Bromine channel
            9: ["C", "H", "O", "N", "S", "Cl", "F", "I", "Br"]
        }
        
    def get_num_channels(self):
        """Return the total number of channels."""
        return len(self.element_channels)
        
    def get_channels_names(self):
        """Return names for each channel."""
        # Create descriptive names for each channel
        names = []
        for idx, elements in self.element_channels.items():
            if len(elements) == 1:
                names.append(f"{elements[0]}_atoms")
            else:
                names.append("other_atoms")
        return names
        
    def get_molecular_complex_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Create channels for all atoms based on their element type."""
        symbs = molecular_complex.element_symbols
        chs = np.asarray([
            np.isin(symbs, elements) 
            for elements in self.element_channels.values()
        ])
        
        # Invert the last channel to capture all non-CHONS atoms
        np.invert(chs[-1], out=chs[-1])
        
        return torch.from_numpy(chs)
        
    def get_protein_channels(self, molecular_complex: MolecularComplex) -> None:
        """Not used in unified view."""
        return None
        
    def get_ligand_channels(self, molecular_complex: MolecularComplex) -> None:
        """Not used in unified view."""
        return None
        
    def __call__(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Override to only return molecular complex channels."""
        return self.get_molecular_complex_channels(molecular_complex)

class ProteinComplex(MolecularComplex):
    """A modified MolecularComplex that handles protein-only structures."""
    
    def __init__(
        self,
        protein_file: str | MolecularData,
        molparser: Parser | None = MolecularParser(),
        path="",
    ):
        """Initialize ProteinComplex with only protein structure.
        
        Args:
            protein_file: Path to the protein file or a MolecularData object
            molparser: A MolecularParser object
            path: Path to the files
        """
        # Parse protein data
        if isinstance(protein_file, MolecularData):
            self.protein_data = protein_file
        else:
            self.protein_data: MolecularData = molparser.parse_file(
                os.path.join(path, protein_file), os.path.splitext(protein_file)[1]
            )

        # Create dummy ligand data with a single atom at origin
        dummy_coords = torch.zeros((3, 1), dtype=DTYPE)
        dummy_symbols = np.array(['C'])  # Single carbon atom
        self.ligand_data = MolecularData(None, dummy_coords, dummy_symbols)
        
        # Initialize other attributes
        self.coords = self.protein_data.coords
        self.n_atoms = self.coords.shape[1]
        self.n_atoms_protein = self.n_atoms
        self.n_atoms_ligand = 0  # No real ligand atoms
        
        self.element_symbols = self.protein_data.element_symbols
        self.vdw_radii = self._get_vdw_radii()
        
        # Set ligand center to None initially - will be set externally
        self.ligand_center = None

