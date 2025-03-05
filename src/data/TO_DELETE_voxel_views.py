import numpy as np
import torch
from docktgrid.molecule import MolecularComplex
from docktgrid.view import View

DEFAULT_CHANNELS = {
    0: ["C"],
    1: ["H"],
    2: ["O"],
    3: ["N"],
    4: ["S"],
    5: ["Cl"],
    6: ["F"],
    7: ["I"],
    8: ["Br"],
    9: ["C", "H", "O", "N", "S"],
}

DEFAULT_CH_NAMES = [
    "carbon",
    "hydrogen",
    "oxygen",
    "nitrogen",
    "sulfur",
    "chlorine",
    "fluorine",
    "iodine",
    "bromine",
    "x",
]

class LigView(View):
    """Inpaint view.

    The `x` below stands for any other chemical element different from CHONS.

    Protein channels (in this order):
        carbon, hydrogen, oxygen, nitrogen, sulfur, x*.
    Ligand channels:
        carbon, hydrogen, oxygen, nitrogen, sulfur, x*.
    """
    def __init__(self,
                 channel_names: list = DEFAULT_CH_NAMES,
                 channels_dict: dict = DEFAULT_CHANNELS,
                 has_protein: bool = False,
                 ):
        super().__init__()
        self.channels = channels_dict
        self.ch_names = channel_names
        self.has_protein = has_protein
    def get_num_channels(self):
        return len(self.channels)

    def get_channels_names(self):
        return (
                [f"{ch}_ligand" for ch in self.ch_names]
        )

    def get_molecular_complex_channels(
            self, molecular_complex: MolecularComplex
    ) -> torch.Tensor:
        """Set of channels for all atoms."""


        nchs = len(self.channels)

        # get a list of bools representing each atom in each channel
        symbs = molecular_complex.element_symbols
        chs = np.asarray([np.isin(symbs, self.channels[c]) for c in range(nchs)])

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
