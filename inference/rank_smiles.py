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
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="checkpoints/combined_protein_to_smiles/epoch_000.ckpt", 
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--pdb_file", 
        type=str, 
        default="data/5sry.pdb", 
        help="Path to the protein PDB file."
        )
    parser.add_argument(
        "--smiles_file", 
        type=str, 
        default="data/cache_round1_smiles_all_out_hits_and_others.csv", 
        help="Path to a CSV file with a 'smiles' column."
    )
    parser.add_argument(
        "--ligand_file", 
        type=str, 
        default="data/5sry_no_hydrogens.mol2", 
        help="Path to a ligand file (SDF/MOL2) to define the pocket center."
    )
    parser.add_argument(
        "--center", type=float, nargs=3, help="Center coordinates (x y z) for the pocket."
    )
    parser.add_argument("--output_file", type=str, default="outputs/ranked_smiles_batch_1.csv", help="Path to save the ranked SMILES CSV file.")
    parser.add_argument("--dtype", type=str, default="torch.float32", help="Data type for the model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for scoring SMILES.")
    
    args = parser.parse_args()
    if isinstance(args.dtype, str):
        args.dtype = eval(args.dtype)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and config
    model, config = load_model(args.ckpt_path, device, dtype=args.dtype)
    
    # Determine pocket center following the same priority logic as generate_smiles_from_pdb.py
    if args.center:
        if args.ligand_file:
            print("--ligand_file provided with --center, ignoring ligand file and using --center")
        center = np.array(args.center)
    elif args.ligand_file:
        center = get_center_from_ligand(args.ligand_file)
    else:
        raise ValueError("Either --ligand_file or --center must be provided.")

    print(f"Using center: {center}")

    # Prepare voxelization config using the training dataset settings stored in the checkpoint config
    complex_dataset_config = config.data.train_dataset.poc2mol_output_dataset.complex_dataset.config
    complex_dataset_config = {k: v for k, v in complex_dataset_config.items() if k != '_target_'}
    inference_config = Poc2MolDataConfig(**complex_dataset_config)
    inference_config.random_rotation = False
    inference_config.random_translation = 0.0
    inference_config.dtype = args.dtype
    
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
            max_smiles_len,
            batch_size=args.batch_size,
        )
    
    smiles_df['likelihood_score'] = scores
    ranked_df = smiles_df.sort_values(by='likelihood_score', ascending=False)
    ranked_df.to_csv(args.output_file, index=False)
    if "hit" in smiles_df.columns:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(smiles_df['hit'], smiles_df['likelihood_score'])
        print(f"AUROC: {auroc}")
    print(f"Saved ranked SMILES to {args.output_file}")

if __name__ == "__main__":
    main() 