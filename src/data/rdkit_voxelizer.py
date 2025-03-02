import pickle
import glob
import numpy as np
from rdkit.Chem import Mol
from src.data.voxel_views import LigView
from docktgrid.molecule import MolecularComplex

import torch

from docktgrid.config import DEVICE, DTYPE
from docktgrid.grid import Grid3D
from docktgrid.view import View

from docktgrid.periodictable import ptable


class RDkitVoxelGrid:
    """Class to generate voxel representations of protein-ligand complexes.

    Attributes:
        grid:
            A Grid3D object.
        views:
            List of Views.
        num_channels:
            Total number of channels for the chosen view configuration.
        shape:
            Voxel grid shape with channels first (n_channels, dim1, dim2, dim3).
        occupancy_func:
            Occupancy function to use.
    """

    def __init__(
        self,
        views: list[View],
        vox_size: float,
        box_dims: list[float],
        occupancy: str = "vdw",
    ):
        """Initialize voxel grid.

        Args:
            views: List of views.
            vox_size: Voxel size.
            box_dims: Dimensions of the box containing the grid.
            occupancy: Occupancy function to use.

        """
        self.occupancy_func = self.get_occupancy_func(occupancy)
        self.grid = Grid3D(vox_size, box_dims)
        self.views = views

    @property
    def num_channels(self):
        """Get total number of channels for the chosen view configuration."""
        return sum([v.get_num_channels() for v in self.views])

    @property
    def shape(self):
        """Get voxel grid shape with channels first.

        Voxel grid has shape (n_channels, dim1, dim2, dim3).

        """
        n_channels = self.num_channels
        dim1, dim2, dim3 = self.grid.axes_dims

        return (n_channels, dim1, dim2, dim3)

    def get_occupancy_func(self, occ):
        """Get occupancy function."""
        if occ == "vdw":
            return self._voxelize_vdw
        else:
            raise NotImplementedError(
                " ".join((f"Occupancy function for {occ} is not implemented yet."))
            )

    def get_channels_mask(self, molecule):
        """Build channels mask.

        Each channel is a boolean mask that indicates which atoms are present in the
        channel.

        Args:
            molecule (docktgrid.molecule.MolecularComplex)

        Returns:
            A torch.Tensor with shape (n_channels, n_atoms) type bool

        """
        return torch.cat([v(molecule) for v in self.views])

    def voxelize(self, molecule, out=None, channels=None, requires_grad=False):
        """Voxelize protein-ligand complex and return voxel grid (features).

        Args:
            molecule: docktgrid.molecule.MolecularComplex.

            out (array-like or None): Alternate output array in which to place the result.
            The default is None; if provided, it must have shape corresponding to
            (n_channels, nvoxels).

            channels (array-like or None): Must have shape (n_channels, n_atoms); if
            provided overrides channels
            created from `view`.

        Returns:
            A torch tensor of shape (n_channels, dim1, dim2, dim3). Each element
            corresponds to voxel values, calculated according to the occupancy model.

        """
        if out is None:
            out = torch.zeros(
                self.shape, dtype=DTYPE, device=DEVICE, requires_grad=requires_grad
            )
        else:
            if out.shape != self.shape:
                raise ValueError(
                    " ".join(
                        (
                            "`out` shape must be == {},".format(self.shape),
                            "currently it is {}".format(out.shape),
                        )
                    )
                )
            out = torch.as_tensor(out, DTYPE, DEVICE, requires_grad=requires_grad)

        if channels is None:
            channels = self.get_channels_mask(molecule)
        else:
            cshape = (self.num_channels, molecule.n_atoms)
            if channels.shape != cshape:
                raise ValueError(
                    " ".join(
                        (
                            "`channels` shape must be == {},".format(cshape),
                            "currently it is {}".format(channels.shape),
                        )
                    )
                )
            channels = torch.as_tensor(channels, dtype=DTYPE, device=DEVICE)

        # create voxel based in occupancy option
        self.occupancy_func(molecule, out, channels)

        return out.view(self.shape)

    @torch.no_grad()
    def _voxelize_vdw(self, molecule, out, channels) -> None:
        points = self.grid.points
        center = molecule.ligand_center
        # translate grid points and reshape for proper broadcasting
        grid = [(u + v).unsqueeze(-1) for u, v in zip(points, center)]

        x, y, z = 0, 1, 2
        # reshape to n_channls, n_points
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
        dist = torch.sqrt(
            torch.pow(ax - px, 2) + torch.pow(ay - py, 2) + torch.pow(az - pz, 2)
        )
        occs = 1 - torch.exp(-1 * torch.pow(vdws / dist, 12))  # voxel occupancies

        for i, mask in enumerate(channels):
            if torch.any(mask):
                torch.amax(occs[:, mask], dim=1, out=out[i])

def load_mol_from_pickle(path):
    with open(path, "rb") as f:
        mol_data = pickle.load(f)
    mol = mol_data["conformers"][0]["rd_mol"]
    return mol

def convert_mol_format(mol):
    pass

class RDkitMolecularComplex(MolecularComplex):
    def __init__(self, mol: Mol):
        self.mol = mol
        self.coords = self._get_coords()
        self.element_symbols = self._get_element_symbols()
        self.ligand_center = self._get_ligand_center()
        self.n_atoms = self.mol.GetNumAtoms()
        self.n_atoms_ligand = self.n_atoms
        self.n_atoms_protein = 0
        self.vdw_radii = self._get_vdw_radii()

    def _get_coords(self):
        conf = self.mol.GetConformer()
        coords = np.array(conf.GetPositions(), dtype=np.float32).T
        return torch.tensor(coords, dtype=DTYPE)

    def _get_element_symbols(self):
        symbols = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        return np.array(symbols)

    def _get_ligand_center(self):
        return torch.mean(self.coords, 1).to(dtype=DTYPE)

    def _get_vdw_radii(self):
        return torch.tensor(
            [ptable[a.title()]["vdw"] for a in self.element_symbols],
            dtype=DTYPE,
        )

if __name__ == "__main__":
    config = {
        "vox_size": 0.75,
        "box_dims": [24.0, 24.0, 24.0],
    }
    voxelizer = RDkitVoxelGrid(
        views=[LigView()],  # you can add multiple views; they are executed in order
        vox_size=config['vox_size'],  # size of the voxel (in Angstrom)
        box_dims=config['box_dims'],  # dimensions of the box (in Angstrom)
        # complex_view=False, # maybe I added this option in my custom implementation
    )
    data_path = "../partial_4327252_rdkit_folder/drugs"
    pickle_files = glob.iglob(f"{data_path}/*.pickle")
    for path in pickle_files:
        name = path.split("/")[-1].split(".")[0]
        mol = load_mol_from_pickle(path)
        molecular_complex = RDkitMolecularComplex(mol)
        voxels = voxelizer.voxelize(molecular_complex)  # this will return a voxel grid; you can save it or
        voxels = voxels[6:]
        show_3d_voxel_lig_only(voxels, angles=None, save_dir='../rdkit_visualisations', identifier=name)
        # show_3d_voxel_lig_only(voxels[None], voxels[None], [name], angles=None, save_dir='../rdkit_visualisations', batch='none')
    pass