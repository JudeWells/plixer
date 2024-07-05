import os
import subprocess
import argparse
import shutil
import gzip
import numpy as np
import pandas as pd
from docktgrid import VoxelGrid
from docktgrid.view import BasicView, VolumeView, View
from docktgrid.molecule import MolecularComplex
from docktgrid.transforms import RandomRotation, Transform
from torch import save as torch_save
import torch

class ComplexView(View):
    """
    The `x` below stands for any other chemical element different from CHONS.

    Protein channels (in this order):
        carbon, hydrogen, oxygen, nitrogen, sulfur, x*.
    Ligand channels:
        carbon, hydrogen, oxygen, nitrogen, sulfur, x*.
    """

    def get_num_channels(self):
        return sum((6, 6))

    def get_channels_names(self):
        chs = ["carbon", "hydrogen", "oxygen", "nitrogen", "sulfur", "other"]
        return (
                + [f"{ch}_protein" for ch in chs]
                + [f"{ch}_ligand" for ch in chs]
        )

    def get_molecular_complex_channels(
            self, molecular_complex: MolecularComplex
    ) -> torch.Tensor:
        """Set of channels for all atoms."""

        channels = {
            0: ["C"],
            1: ["H"],
            2: ["O"],
            3: ["N"],
            4: ["S"],
            5: ["C", "H", "O", "N", "S"],
        }
        nchs = len(channels)

        # get a list of bools representing each atom in each channel
        symbs = molecular_complex.element_symbols
        chs = np.asarray([np.isin(symbs, channels[c]) for c in range(nchs)])

        # invert bools in last channel, since it represents any atom except CHONS
        np.invert(chs[-1], out=chs[-1])

        return torch.from_numpy(chs)

    def get_ligand_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Set of channels for ligand atoms."""
        chs = self.get_molecular_complex_channels(molecular_complex)

        # exclude protein atoms from ligand channels
        chs[..., : -molecular_complex.n_atoms_ligand] = False
        return chs

    def get_protein_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Set of channels for protein atoms."""
        chs = self.get_molecular_complex_channels(molecular_complex)

        # exclude ligand atoms from protein channels
        chs[..., -molecular_complex.n_atoms_ligand:] = False
        return chs

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
        )
