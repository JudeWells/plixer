# ------------------------------------------------------------
# 1.  Install RDKit if you don't already have it
#     (works in conda, or with pip on most recent Python builds)
# ------------------------------------------------------------
# conda install -c rdkit rdkit
#    – or –
# pip install rdkit-pypi CairoSVG reportlab  # Cairo backend helpers

# ------------------------------------------------------------
# 2.  Code to draw a SMILES string on a black canvas with white bonds
# ------------------------------------------------------------
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from pathlib import Path

def mol_to_blackwhite_png(
        smiles: str,
        out_file: str = "molecule.png",
        size: tuple[int, int] = (400, 400),
        line_width: float = 2.0,
        kekulize: bool = True
    ) -> Path:
    """
    Render a 2-D depiction with black background and white bonds.

    Parameters
    ----------
    smiles : str
        Molecule to depict.
    out_file : str | Path
        Destination PNG path (extension *.png*).
    size : (int, int)
        Width, height in pixels.
    line_width : float
        Stroke thickness for bonds (in pixels, ~2–3 looks good).
    kekulize : bool
        Whether to kekulize aromatic rings for clearer drawings.
    """
    # 1. Parse and (optionally) kekulize
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES could not be parsed.")
    if kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)

    # 2. Add 2-D coordinates if missing
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)

    # 3. Set up the Cairo drawer
    w, h = size
    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts   = drawer.drawOptions()

    # --- colours ----------------------------------------------------------
    opts.backgroundColour = (0, 0, 0)           # black

    # Force all atom & bond colours to white.  Different RDKit versions expose
    # *different* APIs for colour palettes, so we try the most robust method
    # first (available in RDKit >=2020), and fall back gracefully.
    white, black = (1, 1, 1), (0, 0, 0)

    try:
        # Preferred: monochrome helper added in RDKit 2020.03
        rdMolDraw2D.SetMonochromeMode(opts, white, black)
    except Exception:
        # Manual fall-backs for older releases.
        try:
            # Newer API: explicit setter methods
            if hasattr(opts, "setAtomPalette"):
                opts.setAtomPalette({i: white for i in range(1, 119)})
            elif hasattr(opts, "updateAtomPalette"):
                opts.updateAtomPalette({i: white for i in range(1, 119)})
            else:
                # Last-ditch: overwrite existing palette dictionary
                if hasattr(opts, "atomColourPalette") and isinstance(opts.atomColourPalette, dict):
                    for k in list(opts.atomColourPalette.keys()):
                        opts.atomColourPalette[k] = white
        finally:
            # Ensure other colours are white too
            opts.highlightColour = white
            opts.defaultColour   = white
    # ----------------------------------------------------------------------

    # 4. Draw & export
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()

    # 5. Write
    out_file = Path(out_file).with_suffix(".png")
    out_file.write_bytes(png_bytes)
    return out_file


# -------------------------------------------------------------------------
# 3.  Example
# -------------------------------------------------------------------------
if __name__ == "__main__":
    img_path = mol_to_blackwhite_png("c1ccccc1C(=O)O", "benzoic_acid.png")
    print("Saved to", img_path.resolve())
