import os
import argparse
import torch
import yaml
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import matplotlib.colors as mcolors
from sklearn.metrics import roc_auc_score
from src.models.vox2smiles import VoxToSmilesModel
from src.data.common.voxelization.molecule_utils import voxelize_molecule
from src.data.common.voxelization.config import Vox2SmilesDataConfig
from src.data.common.tokenizers.smiles_tokenizer import build_smiles_tokenizer
from scripts.benchmarks.cache3_tanimoto_similarity import get_metrics, print_metrics
from src.utils.utils import get_config_from_cpt_path, build_combined_model_from_config
from evaluations.evaluate_combined_vox2smiles import (
    get_tanimoto_similarity_from_smiles,
    get_tanimoto_similarity_from_mol
)
from evaluations.evaluate_vox2smiles import (
    load_molecule, 
    voxelize_input_molecule, 
    visualize_voxel, 
    sample_from_model,
    visualize_molecules
)


def create_voxel_config(data_config_path=None, data_config_dict=None, dtype=torch.float32):
    """Create a voxelization config from the data config file."""
    # Load data config
    if data_config_path is not None:
        with open(data_config_path, 'r') as f:
            data_config_dict = yaml.safe_load(f)
    elif data_config_dict is not None:
        data_config_dict = data_config_dict
    else:
        raise ValueError("Either data_config_path or data_config_dict must be provided")
    
    # Extract the config section
    if '_target_' in data_config_dict:
        del data_config_dict['_target_']
    
    if 'protein_channel_indices' in data_config_dict:
        del data_config_dict['protein_channel_indices']
    
    if 'ligand_channel_indices' in data_config_dict:
        del data_config_dict['ligand_channel_indices']
    
    # Set default values if not present
    default_config = {
        'vox_size': 0.75,
        'box_dims': [24.0]*3,
        'random_rotation': False,
        'random_translation': 0.0,
        'has_protein': False,
        'ligand_channel_names': [
            'C',
            'H',
            'O', 
            'N', 
            'S', 
            'Cl', 
            'F', 
            'I', 
            'Br',  
            'other'
        ],
        'protein_channel_names': [],
        'protein_channels': {},
        'ligand_channels': {
            0: ['C'],
            1: ['O'],
            2: ['N'],
            3: ['S'],
            4: ['Cl'],
            5: ['F'],
            6: ['I'],
            7: ['Br'],
            8: ['C', 'H', 'O', 'N', 'S', 'Cl', 'F', 'I', 'Br']
        },
        'max_atom_dist': 32.0,
        # 'dtype': dtype,
        'include_hydrogens': False,
        'batch_size': 32,
        'max_smiles_len': 200
    }
    data_config_dict['has_protein'] = False
    data_config_dict['protein_channels'] = {}
    data_config_dict['protein_channel_names'] = []

    # Merge with defaults
    for key, value in default_config.items():
        if key not in data_config_dict:
            data_config_dict[key] = value
    data_config_dict['random_rotation'] = False
    data_config_dict['random_translation'] = 0.0
    # Create config object
    return Vox2SmilesDataConfig(**data_config_dict)

def save_smiles_with_similarity(smiles_list, original_mol, output_path):
    # Calculate Tanimoto similarity for each SMILES
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    similarities = [get_tanimoto_similarity_from_mol(original_mol, mol) for mol in mols]
    
    # Create a DataFrame
    df = pd.DataFrame({'SMILES': smiles_list, 'Tanimoto Similarity': similarities})
    print(f"average similarity: {round(df['Tanimoto Similarity'].mean(), 3)}")
    print(f"median similarity: {round(df['Tanimoto Similarity'].median(), 3)}")
    print(f"max similarity: {round(df['Tanimoto Similarity'].max(), 3)}")
    print(f"min similarity: {round(df['Tanimoto Similarity'].min(), 3)}")
    # Save DataFrame to CSV
    df.to_csv(output_path, index=False)

    print(f"DataFrame saved to {output_path}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Load model
    config = get_config_from_cpt_path(args.checkpoint)
    model = build_combined_model_from_config(
        config=config,
        ckpt_path=args.checkpoint,
        dtype=torch.bfloat16,
        device=device
    )
    model.to(device)
    model.eval()
    original_mol = load_molecule(args.molecule)
    voxel_config = create_voxel_config(data_config_dict=config['data']['train_dataset']['poc2mol_output_dataset']['complex_dataset']['config'])

    voxel, input_mol = voxelize_input_molecule(args.molecule, voxel_config)
    print(f"Molecule voxelized, shape: {voxel.shape}")
    voxel_img_path = visualize_voxel(voxel, args.output_dir, identifier=os.path.basename(args.molecule))
    print(f"Voxelized molecule visualization saved to {voxel_img_path}")
    
    vox2smiles_model = model.vox2smiles_model
    print(f"Generating {args.num_samples} samples...")
    smiles_list = sample_from_model(
        vox2smiles_model, 
        voxel,
        max_length=75,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p
    )

    print(f"Generated {len(smiles_list)} SMILES strings")
    # Print generated SMILES
    print("\nGenerated SMILES:")
    for i, smiles in enumerate(smiles_list):
        print(f"{i+1}: {smiles}")
    # count unique smiles
    unique_smiles = set(smiles_list)
    print(f"Number of unique SMILES: {len(unique_smiles)} out of {len(smiles_list)}")
    #select only unique smiles
    smiles_list = list(unique_smiles)

    # Visualize molecules
    input_img_path, valid_mols, valid_smiles = visualize_molecules(
        smiles_list, 
        args.molecule, 
        args.output_dir,
        identifier='generated_' + os.path.basename(args.molecule)
    )
    print(f"\nVisualized {len(valid_mols)} valid molecules (out of {len(smiles_list)})")
    print(f"Images saved to {args.output_dir}")
    
    save_smiles_with_similarity(
        smiles_list=smiles_list, 
        original_mol=original_mol, 
        output_path=os.path.join(args.output_dir, f"{os.path.basename(args.molecule)}_generated_smiles_with_similarity.csv")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate vox2smiles model")
    parser.add_argument("--checkpoint", type=str, 
                        default="logs/CombinedHiQBAggPropPoc2Mol/runs/2025-05-09_04-38-52/checkpoints/epoch_000.ckpt",
                        help="Path to model checkpoint")
    parser.add_argument("--molecule", type=str, 
                        default="data/agonists/5-MeO-DMT_8fy8_E_YFW.mol2",
                        help="Path to molecule file")

    parser.add_argument("--output-dir", type=str, 
                        default="outputs/agonists_CombinedHiQBAggPropPoc2Mol_2025-05-09_04-38-52_temp_0p5",
                        help="Directory to save output files")
    parser.add_argument("--num-samples", type=int, 
                        default=10,
                        help="Number of SMILES strings to generate")
    parser.add_argument("--temperature", type=float, 
                        default=0.5,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, 
                        default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    
    args = parser.parse_args()
    # args.cpu = True
    main(args)
