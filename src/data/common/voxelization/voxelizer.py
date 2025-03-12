import torch
import numpy as np
from typing import List, Dict, Optional, Union, Any

from docktgrid.grid import Grid3D
from docktgrid.view import View
from docktgrid.molecule import MolecularComplex
from docktgrid.config import DEVICE, DTYPE
from docktgrid.periodictable import ptable

from src.data.common.voxelization.config import VoxelizationConfig


class UnifiedView(View):
    """
    A unified view class that can handle both protein-ligand complexes and standalone ligands.
    """
    def __init__(
        self,
        config: VoxelizationConfig,
    ):
        super().__init__()
        self.channels = config.ligand_channels
        self.ch_names = config.ligand_channel_names
        self.has_protein = config.has_protein
        self.protein_channels = config.protein_channels
        self.protein_ch_names = config.protein_channel_names if config.has_protein else []

    def get_num_channels(self):
        """Get the total number of channels."""
        protein_channels = len(self.protein_channels) if self.has_protein else 0
        ligand_channels = len(self.channels)
        return protein_channels + ligand_channels

    def get_channels_names(self):
        """Get the names of all channels."""
        protein_names = [f"{ch}_protein" for ch in self.protein_ch_names] if self.has_protein else []
        ligand_names = [f"{ch}_ligand" for ch in self.ch_names]
        return protein_names + ligand_names

    def get_molecular_complex_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Set of channels for all atoms."""
        # Get element symbols from the molecular complex
        symbs = molecular_complex.element_symbols
        
        # Initialize channels for ligand
        ligand_chs = np.asarray([np.isin(symbs, self.channels[c]) for c in range(len(self.channels))])
        
        # Invert bools in last channel for ligand, since it represents any atom except those already defined
        np.invert(ligand_chs[-1], out=ligand_chs[-1])
        
        # If we have protein channels, process them too
        if self.has_protein:
            protein_chs = np.asarray([np.isin(symbs, self.protein_channels[c]) for c in range(len(self.protein_channels))])
            np.invert(protein_chs[-1], out=protein_chs[-1])
            return torch.from_numpy(np.vstack([protein_chs, ligand_chs]))
        
        return torch.from_numpy(ligand_chs)

    def get_ligand_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Set of channels for ligand atoms."""
        chs = self.get_molecular_complex_channels(molecular_complex)
        
        # If we have protein channels, they come first, so we need to slice accordingly
        if self.has_protein:
            ligand_chs = chs[len(self.protein_channels):]
        else:
            ligand_chs = chs
            
        # Exclude protein atoms from ligand channels
        ligand_chs[..., : -molecular_complex.n_atoms_ligand] = False
        return ligand_chs

    def get_protein_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Set of channels for protein atoms."""
        if not self.has_protein:
            return torch.tensor([], dtype=torch.bool)
            
        chs = self.get_molecular_complex_channels(molecular_complex)
        protein_chs = chs[:len(self.protein_channels)]
        
        # Exclude ligand atoms from protein channels
        protein_chs[..., -molecular_complex.n_atoms_ligand:] = False
        return protein_chs

    def __call__(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Concatenate all channels in a single tensor."""
        if self.has_protein:
            protein = self.get_protein_channels(molecular_complex)
        else:
            protein = None
            
        ligand = self.get_ligand_channels(molecular_complex)
        
        return torch.cat(
            (
                protein if protein is not None else torch.tensor([], dtype=torch.bool),
                ligand if ligand is not None else torch.tensor([], dtype=torch.bool),
            ),
        )


class UnifiedVoxelGrid:
    """
    A unified voxel grid class that can handle both protein-ligand complexes and standalone ligands.
    This is based on the existing VoxelGrid and RDkitVoxelGrid classes but with a unified interface.
    """
    def __init__(
        self,
        config: VoxelizationConfig,
    ):
        """Initialize the unified voxel grid."""
        self.config = config
        self.occupancy_func = self._voxelize_vdw  # Currently only supporting vdw occupancy
        self.grid = Grid3D(config.vox_size, config.box_dims)
        self.view = UnifiedView(config)

    @property
    def num_channels(self):
        """Get total number of channels for the chosen view configuration."""
        return self.view.get_num_channels()

    @property
    def shape(self):
        """Get voxel grid shape with channels first (n_channels, dim1, dim2, dim3)."""
        n_channels = self.num_channels
        dim1, dim2, dim3 = self.grid.axes_dims
        return (n_channels, dim1, dim2, dim3)

    def get_channels_mask(self, molecule):
        """Build channels mask for each atom."""
        return self.view(molecule)

    def voxelize(self, molecule, out=None, channels=None, requires_grad=False):
        """Voxelize molecule and return voxel grid."""
        if out is None:
            out = torch.zeros(
                self.shape, dtype=self.config.dtype, device=DEVICE, requires_grad=requires_grad
            )
        else:
            if out.shape != self.shape:
                raise ValueError(
                    f"`out` shape must be == {self.shape}, currently it is {out.shape}"
                )
            out = torch.as_tensor(out, self.config.dtype, DEVICE, requires_grad=requires_grad)

        if channels is None:
            channels = self.get_channels_mask(molecule)
        else:
            cshape = (self.num_channels, molecule.n_atoms)
            if channels.shape != cshape:
                raise ValueError(
                    f"`channels` shape must be == {cshape}, currently it is {channels.shape}"
                )
            channels = torch.as_tensor(channels, dtype=self.config.dtype, device=DEVICE)

        # Create voxel based on occupancy option
        self._voxelize_vdw(molecule, out, channels)

        return out.view(self.shape)

    @torch.no_grad()
    def _voxelize_vdw(self, molecule, out, channels) -> None:
        """Voxelize using van der Waals radii."""
        points = self.grid.points
        center = molecule.ligand_center
        # Translate grid points and reshape for proper broadcasting
        grid = [(u + v).unsqueeze(-1) for u, v in zip(points, center)]

        x, y, z = 0, 1, 2
        # Reshape to n_channels, n_points
        out = out.view(channels.shape[0], grid[x].shape[0])

        self._calc_vdw_occupancies(
            out,
            channels,
            molecule.coords[x].to(DEVICE),
            molecule.coords[y].to(DEVICE),
            molecule.coords[z].to(DEVICE),
            grid[x].to(DEVICE),
            grid[y].to(DEVICE),
            grid[z].to(DEVICE),
            molecule.vdw_radii.to(DEVICE),
        )

    @staticmethod
    @torch.jit.script
    def _calc_vdw_occupancies(
        out: torch.Tensor,  # output tensor, shape (n_channels, n_points)
        channels: torch.Tensor,  # bool mask of channels, shape (n_channels, n_atoms)
        ax: torch.Tensor,  # x coords of atoms, shape (n_atoms,)
        ay: torch.Tensor,  # y coords of atoms, shape (n_atoms,)
        az: torch.Tensor,  # z coords of atoms, shape (n_atoms,)
        px: torch.Tensor,  # x coords of grid points, shape (n_points, 1)
        py: torch.Tensor,  # y coords of grid points, shape (n_points, 1)
        pz: torch.Tensor,  # z coords of grid points, shape (n_points, 1)
        vdws: torch.Tensor,  # vdw radii of atoms, shape (n_atoms,)
    ):
        """Calculate voxel occupancies using van der Waals radii."""
        dist = torch.sqrt(
            torch.pow(ax - px, 2) + torch.pow(ay - py, 2) + torch.pow(az - pz, 2)
        )
        occs = 1 - torch.exp(-1 * torch.pow(vdws / dist, 12))  # voxel occupancies
        
        # Convert occs to the same dtype as out to avoid dtype mismatch
        occs = occs.to(dtype=out.dtype)

        for i, mask in enumerate(channels):
            if torch.any(mask):
                torch.amax(occs[:, mask], dim=1, out=out[i])


class RDkitMolecularComplex(MolecularComplex):
    """
    A wrapper for RDKit molecules to make them compatible with the MolecularComplex interface.
    This allows us to use the same voxelization code for both protein-ligand complexes and standalone ligands.
    """
    def __init__(self, mol):
        """Initialize from an RDKit molecule."""
        self.mol = mol
        self.coords = self._get_coords()
        self.element_symbols = self._get_element_symbols()
        self.ligand_center = self._get_ligand_center()
        self.n_atoms = self.mol.GetNumAtoms()
        self.n_atoms_ligand = self.n_atoms
        self.n_atoms_protein = 0
        self.vdw_radii = self._get_vdw_radii()
        
        # These are placeholders to maintain compatibility with MolecularComplex
        class DummyData:
            def __init__(self, parent):
                self.coords = parent.coords
                self.element_symbols = parent.element_symbols
                self.vdw_radii = parent.vdw_radii
        
        self.ligand_data = DummyData(self)
        self.protein_data = DummyData(self)

    def _get_coords(self):
        """Get atom coordinates from the RDKit molecule."""
        conf = self.mol.GetConformer()
        coords = np.array(conf.GetPositions(), dtype=np.float32).T
        return torch.tensor(coords, dtype=DTYPE)

    def _get_element_symbols(self):
        """Get element symbols from the RDKit molecule."""
        symbols = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        return np.array(symbols)

    def _get_ligand_center(self):
        """Calculate the center of the molecule."""
        return torch.mean(self.coords, 1).to(dtype=DTYPE)

    def _get_vdw_radii(self):
        """Get van der Waals radii for each atom."""
        return torch.tensor(
            [ptable[a.title()]["vdw"] for a in self.element_symbols],
            dtype=DTYPE,
        ) 