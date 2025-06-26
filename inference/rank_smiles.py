import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.utils import load_model, get_center_from_ligand, voxelize_protein
from src.data.common.voxelization.config import Poc2MolDataConfig
from src.data.common.tokenizers.smiles_tokenizer import build_smiles_tokenizer

def main():
    parser = argparse.ArgumentParser(description="Rank SMILES against a protein PDB file.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--pdb_file", type=str, required=True, help="Path to the protein PDB file.")
    parser.add_argument("--smiles_file", type=str, required=True, help="Path to a CSV file with a 'smiles' column.")
    parser.add_argument("--ligand_file", type=str, help="Path to a ligand file (SDF/MOL2) to define the pocket center.")
    parser.add_argument("--center", type=float, nargs=3, help="Center coordinates (x y z) for the pocket.")
    parser.add_argument("--output_file", type=str, default="ranked_smiles.csv", help="Path to save the ranked SMILES CSV file.")
    
    args = parser.parse_args()

    if not args.ligand_file and not args.center:
        raise ValueError("Either --ligand_file or --center must be provided.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and config
    model, config = load_model(args.ckpt_path, device)
    
    # Determine center
    if args.ligand_file:
        center = get_center_from_ligand(args.ligand_file)
    else:
        center = np.array(args.center)

    print(f"Using center: {center}")

    # Prepare voxelization config
    inference_config = Poc2MolDataConfig(
        **config.data.config
    )
    inference_config.random_rotation = False
    inference_config.random_translation = 0.0
    inference_config.has_protein = True
    
    # Voxelize protein
    protein_voxels = voxelize_protein(args.pdb_file, center, inference_config).to(device)

    # Load SMILES
    smiles_df = pd.read_csv(args.smiles_file)
    smiles_list = smiles_df['smiles'].tolist()

    # Prepare tokenizer
    tokenizer = build_smiles_tokenizer()
    max_smiles_len = config.data.config.max_smiles_len

    # Score SMILES
    print(f"Scoring {len(smiles_list)} SMILES...")
    with torch.no_grad():
        scores = model.score_smiles(
            protein_voxels,
            smiles_list,
            tokenizer,
            max_smiles_len
        )
    
    smiles_df['likelihood_score'] = scores
    
    # Sort by score
    ranked_df = smiles_df.sort_values(by='likelihood_score', ascending=False)

    # Save results
    ranked_df.to_csv(args.output_file, index=False)
    print(f"Saved ranked SMILES to {args.output_file}")

if __name__ == "__main__":
    main() 