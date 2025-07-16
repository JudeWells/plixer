import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import os
from pathlib import Path

# Read the dataframe
df_path = "/Users/judewells/Documents/dataScienceProgramming/binding_affinity/VoxelDiffOuter/plixer/evaluation_results/checkpoints_model_run_2025-07-02_batched/chrono_combined_model_results_20250702_140314.csv"
df = pd.read_csv(df_path)

# Create output directory
output_dir = "outputs/molecular_images"
os.makedirs(output_dir, exist_ok=True)

# Function to generate and save molecular image
def generate_mol_image(smiles, output_path, size=(400, 400)):
    try:
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Failed to parse SMILES: {smiles}")
            return False
        
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        
        # Create the drawing object with custom settings
        drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
        opts = drawer.drawOptions()
        
        # Set up black background
        opts.clearBackground = False
        opts.backgroundColour = (0, 0, 0)
        
        # Set atom colors - brighter colors to stand out against black
        atom_colors = {
            6: (1, 1, 1),      # Carbon - white
            7: (0.5, 0.5, 1),  # Nitrogen - light blue
            8: (1, 0.5, 0.5),  # Oxygen - light red
            9: (0.5, 1, 0.5),  # Fluorine - light green
            15: (1, 0.65, 0),  # Phosphorus - orange
            16: (1, 1, 0),     # Sulfur - yellow
            17: (0, 1, 0),     # Chlorine - green
            35: (0.6, 0.2, 0), # Bromine - brown
            53: (0.6, 0, 0.6)  # Iodine - purple
        }
        
        # Set the atom colors
        if hasattr(opts, 'setAtomPalette'):
            opts.setAtomPalette(atom_colors)
        elif hasattr(opts, 'updateAtomPalette'):
            opts.updateAtomPalette(atom_colors)
        else:
            opts.atomColourPalette = atom_colors
        
        # Set drawing options
        opts.bondLineWidth = 4
        opts.highlightColour = (1, 1, 1)  # White highlights
        
        # Draw molecule
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        # Get the drawing and write to file
        png_bytes = drawer.GetDrawingText()
        out_file = Path(output_path).with_suffix(".png")
        out_file.write_bytes(png_bytes)
        
        return True

    except Exception as e:
        print(f"Error generating image for SMILES {smiles}: {str(e)}")
        return False

# Process both sampled_smiles and original smiles
for idx, row in df.iterrows():
    # Get identifier from current row
    identifier = row['name'].split("_")[0]
    ts = round(row['tanimoto_similarity'], 2)
    p2m_loss = round(row['poc2mol_loss'], 2)
    # Generate image for sampled SMILES
    if 'sampled_smiles' in df.columns:
        sampled_path = os.path.join(output_dir, f"{identifier}_sampled_mol_{idx}_{ts}_{p2m_loss}.png")
        generate_mol_image(row['sampled_smiles'].replace("[BOS]", "").replace("[EOS]", ""), sampled_path)
    
    # Generate image for original SMILES
    if 'smiles' in df.columns:
        original_path = os.path.join(output_dir, f"{identifier}_original_mol_{idx}_{ts}_{p2m_loss}.png")
        generate_mol_image(row['smiles'].replace("[BOS]", "").replace("[EOS]", ""), original_path)

print("Image generation complete. Check the 'molecular_images' directory for the output.")

