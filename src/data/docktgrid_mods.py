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
from docktgrid.molparser import MolecularParser
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
        symbs = molecular_complex.element_symbols[-molecular_complex.n_atoms_ligand:]
        chs = np.asarray(
            [np.isin(
                symbs,
                self.cfg.ligand_channels[c]
            ) for c in range(len(self.cfg.ligand_channels))])
        np.invert(chs[-1], out=chs[-1])
        return torch.from_numpy(chs)

    def get_protein_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Set of channels for protein atoms."""
        symbs = molecular_complex.element_symbols[:-molecular_complex.n_atoms_ligand]
        chs = np.asarray(
            [np.isin(symbs, self.cfg.protein_channels[c]
            ) for c in range(len(self.cfg.protein_channels))])
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
        joined = torch.zeros(
            size=(max(protein.shape[0], ligand.shape[0]), molecular_complex.n_atoms),
        )
        joined[:protein.shape[0], :molecular_complex.n_atoms_protein] = protein
        joined[:ligand.shape[0], -molecular_complex.n_atoms_ligand:] = ligand
        return joined.bool()

