import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd

# Create molecule from SMILES
def smiles_to_transparent_png(smiles, save_path):
    smiles = smiles.replace("[BOS]", "").replace("[EOS]", "")
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    # Set up drawer with Cairo backend
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
    # Transparent background
    drawer.drawOptions().clearBackground = False

    # Use default color scheme (which colors atom types automatically)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Save to file or bytes
    with open(save_path, "wb") as f:
        f.write(drawer.GetDrawingText())
if __name__ == "__main__":
    output_dir = "/Users/judewells/Downloads/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol/transparent_mols"
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv("/Users/judewells/Downloads/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol/combined_model_results.csv")
    df = df[df["valid_smiles"] == 1]
    df = df.sort_values(by="tanimoto_similarity", ascending=False)
    for i, row in df.iterrows():
        smiles = row["smiles"]
        sampled_smiles = row["sampled_smiles"]
        ts = row["tanimoto_similarity"]
        img_dir = os.path.join(output_dir, f"ts_{ts:.2f}_{row['name']}")
        os.makedirs(img_dir, exist_ok=True)
        save_path = os.path.join(img_dir, f"true.png")
        smiles_to_transparent_png(smiles, save_path)
        save_path = os.path.join(img_dir, f"sampled.png")
        smiles_to_transparent_png(sampled_smiles, save_path)

