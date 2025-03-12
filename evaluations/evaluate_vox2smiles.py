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


def load_model(checkpoint_path, config_path):
    """Load the VoxToSmilesModel from a checkpoint."""
    # Load model config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create model config
    model_config = config_dict['config']
    
    # Ensure layer_norm_eps is a float
    if 'layer_norm_eps' in model_config and isinstance(model_config['layer_norm_eps'], str):
        model_config['layer_norm_eps'] = float(model_config['layer_norm_eps'])
    
    # Initialize model
    model = VoxToSmilesModel(
        config=OmegaConf.create(model_config),
        override_optimizer_on_load=True
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def load_molecule(mol_path):
    """Load a molecule from a file."""
    # Determine file format from extension
    ext = os.path.splitext(mol_path)[1].lower()
    
    if ext == '.mol2':
        # Use a more robust approach for mol2 files
        try:
            # First try with RDKit's built-in function
            mol = Chem.MolFromMol2File(mol_path, sanitize=False)
            # Perform sanitization separately with error handling
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ 
                                     Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    # Try to kekulize, but continue even if it fails
                    try:
                        Chem.Kekulize(mol, clearAromaticFlags=False)
                    except:
                        print("Warning: Kekulization failed, continuing with aromatic structure")
                except:
                    print("Warning: Sanitization failed, using unsanitized molecule")
            
            # If RDKit's method failed, try using the docktgrid parser
            if mol is None:
                from src.data.docktgrid_mods import MolecularParserWrapper
                parser = MolecularParserWrapper()
                mol_data = parser.parse_file(mol_path, '.mol2')
                # Convert to RDKit mol
                mol = Chem.MolFromMol2Block(mol_data.mol2block, sanitize=False)
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ 
                                         Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    except:
                        print("Warning: Sanitization failed, using unsanitized molecule")
        except Exception as e:
            print(f"Error loading mol2 file: {e}")
            mol = None
    elif ext == '.sdf':
        mol = Chem.SDMolSupplier(mol_path, sanitize=False)[0]
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ 
                                 Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except:
                print("Warning: Sanitization failed, using unsanitized molecule")
    elif ext == '.mol':
        mol = Chem.MolFromMolFile(mol_path, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ 
                                 Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except:
                print("Warning: Sanitization failed, using unsanitized molecule")
    elif ext == '.pdb':
        mol = Chem.MolFromPDBFile(mol_path, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ 
                                 Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except:
                print("Warning: Sanitization failed, using unsanitized molecule")
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    if mol is None:
        raise ValueError(f"Failed to load molecule from {mol_path}")
    
    return mol


def create_voxel_config(data_config_path):
    """Create a voxelization config from the data config file."""
    # Load data config
    with open(data_config_path, 'r') as f:
        data_config_dict = yaml.safe_load(f)
    
    # Extract the config section
    config_dict = data_config_dict.get('config', {})
    if '_target_' in config_dict:
        del config_dict['_target_']
    
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
            1: ['H'],
            2: ['O'],
            3: ['N'],
            4: ['S'],
            5: ['Cl'],
            6: ['F'],
            7: ['I'],
            8: ['Br'],
            9: ['C', 'H', 'O', 'N', 'S', 'Cl', 'F', 'I', 'Br']
        },
        'max_atom_dist': 32.0,
        'dtype': torch.float32,
        'include_hydrogens': False,
        'batch_size': 32,
        'max_smiles_len': 200
    }
    
    # Merge with defaults
    for key, value in default_config.items():
        if key not in config_dict:
            config_dict[key] = value
    
    # Create config object
    return Vox2SmilesDataConfig(**config_dict)


def voxelize_input_molecule(mol_path, voxel_config):
    """Voxelize a molecule according to the pipeline in Vox2SmilesDataset."""
    # Load the molecule
    mol = load_molecule(mol_path)
    
    # Remove hydrogens if specified in config
    if not voxel_config.include_hydrogens:
        try:
            mol = Chem.RemoveHs(mol)
        except:
            print("Warning: Failed to remove hydrogens, continuing with original molecule")
    
    # Voxelize the molecule
    try:
        voxel = voxelize_molecule(mol, voxel_config)
    except Exception as e:
        print(f"Error in standard voxelization: {e}")
        print("Falling back to alternative voxelization method...")
        
        # Fall back to using the voxelization pipeline from ComplexDataset
        # This is a simplified version that only handles the ligand part
        from src.data.common.voxelization.molecule_utils import (
            apply_random_rotation,
            apply_random_translation,
            prepare_rdkit_molecule
        )
        
        # Prepare the molecule
        mol = prepare_rdkit_molecule(mol, voxel_config)
        
        # Apply transformations if needed
        if voxel_config.random_rotation:
            mol = apply_random_rotation(mol)
        
        if voxel_config.random_translation > 0:
            mol = apply_random_translation(mol, voxel_config.random_translation)
        
        # Create voxelizer and voxelize
        from src.data.common.voxelization.voxelizer import UnifiedVoxelGrid
        voxelizer = UnifiedVoxelGrid(voxel_config)
        voxel = voxelizer.voxelize_ligand(mol)
    
    return voxel, mol


def sample_from_model(model, voxel, num_samples=5, max_length=200, temperature=1.0, top_p=0.9):
    """Generate SMILES strings from the voxelized molecule."""
    model.eval()
    device = next(model.parameters()).device
    voxel = voxel.to(device)
    
    # Expand voxel for multiple samples
    batch_voxel = voxel.unsqueeze(0).repeat(num_samples, 1, 1, 1, 1)
    
    # Generate sequences
    with torch.no_grad():
        outputs = model.model.generate(
            batch_voxel,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_samples
        )
    
    # Decode sequences
    tokenizer = build_smiles_tokenizer()
    smiles_list = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    smiles_list = [s.replace(' ', '') for s in smiles_list]
    
    return smiles_list


def calculate_log_likelihood(model, voxel, smiles_strings, max_length=200):
    """Calculate log likelihood for a list of SMILES strings given the voxel."""
    model.eval()
    device = next(model.parameters()).device
    voxel = voxel.to(device)
    tokenizer = build_smiles_tokenizer()
    
    results = []
    
    for smiles in smiles_strings:
        # Add special tokens if not present
        if not smiles.startswith(tokenizer.bos_token):
            smiles = tokenizer.bos_token + smiles
        if not smiles.endswith(tokenizer.eos_token):
            smiles = smiles + tokenizer.eos_token
        
        # Tokenize
        tokens = tokenizer(
            smiles,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Calculate log likelihood
        with torch.no_grad():
            outputs = model(
                voxel.unsqueeze(0),
                labels=tokens.input_ids
            )
            
            # Get loss (negative log likelihood)
            nll = outputs.loss.item()
            log_likelihood = -nll
            
            # Calculate per-token log likelihood
            seq_length = tokens.attention_mask.sum().item()
            per_token_log_likelihood = log_likelihood / seq_length
        
        results.append({
            'smiles': smiles.replace(tokenizer.bos_token, '').replace(tokenizer.eos_token, ''),
            'log_likelihood': log_likelihood,
            'per_token_log_likelihood': per_token_log_likelihood,
            'sequence_length': seq_length
        })
    
    return results


def visualize_molecules(smiles_list, mol_path, output_dir):
    """Visualize the generated molecules alongside the input molecule."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and visualize input molecule
    input_mol = load_molecule(mol_path)
    input_img = Draw.MolToImage(input_mol, size=(300, 300))
    input_img_path = os.path.join(output_dir, "input_molecule.png")
    input_img.save(input_img_path)
    
    # Visualize generated molecules
    valid_mols = []
    valid_smiles = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_mols.append(mol)
                valid_smiles.append(smiles)
        except:
            print(f"Failed to parse SMILES: {smiles}")
    
    # Create a grid of molecule images
    if valid_mols:
        img = Draw.MolsToGridImage(
            valid_mols,
            molsPerRow=3,
            subImgSize=(300, 300),
            legends=[f"{i+1}: {s}" for i, s in enumerate(valid_smiles)]
        )
        grid_path = os.path.join(output_dir, "generated_molecules.png")
        img.save(grid_path)
    
    return input_img_path, valid_mols, valid_smiles


def visualize_voxel(voxel, output_dir, identifier='voxelized_molecule'):
    """Visualize the voxelized molecule and save it to the output directory.
    
    Args:
        voxel: The voxelized molecule tensor
        output_dir: Directory to save the visualization
        identifier: Base name for the saved files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy if it's a tensor
    if not isinstance(voxel, np.ndarray):
        voxel = voxel.detach().cpu().numpy()
    
    # Threshold the voxel grid
    voxel = (voxel > 0.5).astype(int)
    
    # Define colors for each atom type
    colors = np.zeros((voxel.shape[0], 4))
    colors[0] = mcolors.to_rgba('green')    # carbon
    colors[1] = mcolors.to_rgba('white')    # hydrogen
    colors[2] = mcolors.to_rgba('red')      # oxygen
    colors[3] = mcolors.to_rgba('blue')     # nitrogen
    colors[4] = mcolors.to_rgba('yellow')   # sulfur
    colors[5] = mcolors.to_rgba('magenta')  # chlorine
    colors[6] = mcolors.to_rgba('magenta')  # fluorine
    colors[7] = mcolors.to_rgba('magenta')  # iodine
    colors[8] = mcolors.to_rgba('magenta')  # bromine
    colors[9] = mcolors.to_rgba('cyan')     # other
    
    # Set transparency for all colors
    colors[:, 3] = 0.2
    
    # Define viewing angles
    angles = [(45, 45), (15, 90), (0, 30), (45, -45)]
    
    # Create figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each channel with its corresponding color
    for channel in range(voxel.shape[0]):
        ax.voxels(voxel[channel], facecolors=colors[channel], edgecolors=colors[channel])
    
    # Save from different angles
    for i, angle in enumerate(angles):
        ax.view_init(elev=angle[0], azim=angle[1])
        ax.axis('off')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        save_path = os.path.join(output_dir, f"{identifier}_angle_{i+1}.png")
        plt.savefig(save_path)
        print(f"Saved voxel visualization to {save_path}")
    
    plt.close(fig)
    
    return os.path.join(output_dir, f"{identifier}_angle_1.png")


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Create voxel config
    voxel_config = create_voxel_config(args.data_config)
    
    # Load model
    model = load_model(args.checkpoint, args.model_config)
    model = model.to(device)
    print("Model loaded successfully")
    
    # Voxelize input molecule
    voxel, input_mol = voxelize_input_molecule(args.molecule, voxel_config)
    print(f"Molecule voxelized, shape: {voxel.shape}")
    
    # Visualize the voxelized molecule
    voxel_img_path = visualize_voxel(voxel, args.output_dir)
    print(f"Voxelized molecule visualization saved to {voxel_img_path}")
    
    # Sample from model
    print(f"Generating {args.num_samples} samples...")
    smiles_list = sample_from_model(
        model, 
        voxel, 
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
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
        args.output_dir
    )
    print(f"\nVisualized {len(valid_mols)} valid molecules (out of {len(smiles_list)})")
    print(f"Images saved to {args.output_dir}")
    
    # Calculate log likelihood if CSV file is provided
    if args.smiles_csv:
        print(f"\nCalculating log likelihood for SMILES in {args.smiles_csv}")
        df = pd.read_csv(args.smiles_csv)
        
        # Get SMILES column
        smiles_col = args.smiles_column
        if smiles_col not in df.columns:
            # Try to find a column that might contain SMILES
            potential_cols = [col for col in df.columns if 'smiles' in col.lower()]
            if potential_cols:
                smiles_col = potential_cols[0]
                print(f"SMILES column not found, using {smiles_col} instead")
            else:
                raise ValueError(f"SMILES column '{args.smiles_column}' not found in CSV")
        
        # Get SMILES strings
        smiles_strings = df[smiles_col].tolist()
        
        # Calculate log likelihood
        results = calculate_log_likelihood(model, voxel, smiles_strings)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        # merge results_df with df on smiles column
        results_df = pd.merge(df, results_df, on='smiles', how='left')
        
        # Calculate metrics for both log likelihood and per-token log likelihood
        ll_metrics = get_metrics(results_df, 'log_likelihood')
        per_token_ll_metrics = get_metrics(results_df, 'per_token_log_likelihood')
        
        # Print metrics
        print_metrics(ll_metrics, score_name="log_likelihood")
        print_metrics(per_token_ll_metrics, score_name="per_token_log_likelihood")
        
        # Save results
        output_path = os.path.join(args.output_dir, "log_likelihood_results.csv")
        results_df.to_csv(output_path, index=False)
        print(f"Log likelihood results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate vox2smiles model")
    parser.add_argument("--checkpoint", type=str, 
                        default="logs/vox2smilesCombined/runs/2025-03-06_09-11-10/checkpoints/interrupted.ckpt",
                        help="Path to model checkpoint")
    parser.add_argument("--molecule", type=str, 
                        default="data/5sry_no_hydrogens.mol2",
                        help="Path to molecule file")
    parser.add_argument("--model-config", type=str, 
                        default="configs/model/vox2smiles.yaml",
                        help="Path to model config file")
    parser.add_argument("--data-config", type=str, 
                        default="configs/data/vox2smiles_geom_data.yaml",
                        help="Path to data config file")
    parser.add_argument("--output-dir", type=str, 
                        default="vox2smiles_evaluation2",
                        help="Directory to save output files")
    parser.add_argument("--num-samples", type=int, 
                        default=10,
                        help="Number of SMILES strings to generate")
    parser.add_argument("--temperature", type=float, 
                        default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, 
                        default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--smiles-csv", type=str, 
                        default="data/cache_round1_smiles_all_out_hits_and_others.csv",
                        help="Path to CSV file containing SMILES strings for log likelihood calculation")
    parser.add_argument("--smiles-column", type=str, 
                        default="smiles",
                        help="Column name in CSV file containing SMILES strings")
    parser.add_argument("--cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    
    args = parser.parse_args()
    # args.cpu = True
    main(args)