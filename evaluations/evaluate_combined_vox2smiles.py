import os
import copy
import torch
import numpy as np
import yaml
import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from transformers import PreTrainedTokenizerFast
from src.models.poc2mol import Poc2Mol
from src.evaluation.visual import visualise_batch, show_3d_voxel_lig_only, visualize_2d_molecule_batch
from src.utils.utils import get_config_from_cpt_path, build_combined_model_from_config
from src.data.poc2mol.datasets import ComplexDataset, ParquetDataset
from src.data.common.voxelization.config import Poc2MolDataConfig
from src.models.poc2smiles import CombinedProteinToSmilesModel
from src.data.common.tokenizers.smiles_tokenizer import build_smiles_tokenizer
from visualisation import generate_plots_from_results_df
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
import pandas as pd
"""
This script evaluates a combined model
where we have a checkpoint for the poc2mol model
which has been trained on the HiQBind dataset
then we combine this with a vox2smiles model which has
been trained on the Zinc dataset and the outputs of the 
poc2mol dataset.

"""

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Poc2Mol model")
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="logs/CombinedHiQBindCkptFrmPrevCombined/runs/2025-05-06_20-51-46/checkpoints/last.ckpt",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluation_results/CombinedModel_2025-05-06_v5",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--pdb_dir", 
        type=str, 
        # default="../PDBbind_v2020_refined-set",
        default="../hiqbind/parquet/test/",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="torch.bfloat16",
        help="Evaluation dtype"
    )
    parser.add_argument(
        "--skip_visualisation", 
        action="store_true",
    )
    return parser.parse_args()

def build_test_config(complex_dataset_config: DictConfig, dtype: str):
    test_config = copy.deepcopy(complex_dataset_config)
    test_config['random_rotation'] = False
    test_config['random_translation'] = 0.0
    test_config['batch_size'] = 1
    test_config.dtype = dtype
    test_config.remove_hydrogens = True
    return test_config

def build_parquet_test_dataloader(poc2mol_config: DictConfig, dtype: str, pdb_dir: str):
    test_config = build_test_config(poc2mol_config, dtype)
    dataset = ParquetDataset(
        config=test_config,
        data_path=pdb_dir,
        use_cluster_member_zero=True
    )
    return DataLoader(dataset, batch_size=1, shuffle=True)


def build_pdb_test_dataloader(
        poc2mol_config: DictConfig, 
        pdb_dir: str, 
        dtype: str
    ):
    test_config = build_test_config(poc2mol_config, dtype)
    
    dataset = ComplexDataset(
        config=test_config,
        pdb_dir=pdb_dir,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False)

def add_smiles_to_batch(batch: dict, tokenizer: PreTrainedTokenizerFast, max_smiles_len: int):
    for i, smi in enumerate(batch['smiles']):
        if not smi.startswith(tokenizer.bos_token):
            batch['smiles'][i] = tokenizer.bos_token + tokenizer.bos_token + smi
        if not smi.endswith(tokenizer.eos_token):
            batch['smiles'][i] = batch['smiles'][i] + tokenizer.eos_token
        tokenized = tokenizer(
            batch['smiles'],
            padding='max_length',
            max_length=max_smiles_len,
            truncation=True,
            return_tensors="pt"
        )
        device  = batch['protein'].device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
    batch.update(tokenized)
    return batch

def get_tanimoto_similarity_from_smiles(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        # Return None if molecule creation fails
        if mol1 is None or mol2 is None:
            return None

        # Generate Morgan fingerprints with radius 2
        morgan_fp1 = AllChem.GetMorganFingerprint(mol1, 2)
        morgan_fp2 = AllChem.GetMorganFingerprint(mol2, 2)
        
        # Compute Tanimoto similarity between the fingerprints
        tanimoto_similarity = DataStructs.TanimotoSimilarity(morgan_fp1, morgan_fp2)
        return tanimoto_similarity
    except Exception as e:
        return None


def get_max_common_substructure_num_atoms(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return None

        mcs_result = rdFMCS.FindMCS([mol1, mol2])
        mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
        if mcs_mol is None:
            return 0
        # Use GetNumHeavyAtoms() instead of GetNumAtoms()
        return mcs_mol.GetNumHeavyAtoms()
    except Exception as e:
        print("Error computing MCS:", e)
        return None




def evaluate_combined_model(
        combined_model: CombinedProteinToSmilesModel, 
        test_dataloader: DataLoader,
        output_dir="evaluation_results",
        save_voxels: bool = False,
        skip_visualisation: bool = False
        ):
    os.makedirs(output_dir, exist_ok=True)
    # Get tokenizer configuration from the model config
    config = combined_model.config
    max_smiles_len = config.data.config.max_smiles_len
    
    # Instantiate the tokenizer
    tokenizer = build_smiles_tokenizer()
    result_rows = []
    decoy_labels = None
    decoy_smiles = None
    visualisation_batch = []
    vis_batch_idx = 0
    for i, batch in enumerate(test_dataloader):
        if len(batch['smiles'][0]) > 100:
            continue
        if 'smiles' in batch:
            batch = add_smiles_to_batch(batch, tokenizer, max_smiles_len)
        # Get the model output
        result = combined_model.forward(
            batch['protein'], 
            batch['input_ids'], 
            decoy_labels, 
            ligand_voxels=batch['ligand']
            )
        decoy_labels = batch['input_ids'] # use previous batch as decoy labels
        if save_voxels and not skip_visualisation:
            assert len(batch['ligand']) == 1
            os.makedirs(f'{output_dir}/voxels', exist_ok=True)
            poc2mol_output_path = f'{output_dir}/voxels/poc2mol_output_{batch["name"][0]}_{str(int(round(result["poc2mol_loss"], 3) * 1000)).zfill(4)}.npy'
            true_ligand_voxels_path = f'{output_dir}/voxels/true_ligand_voxels_{batch["name"][0]}_{str(int(round(result["poc2mol_loss"], 3) * 1000)).zfill(4)}.npy'
            protein_voxels_path = f'{output_dir}/voxels/protein_voxels_{batch["name"][0]}_{str(int(round(result["poc2mol_loss"], 3) * 1000)).zfill(4)}.npy'
            # save as numpy arrays
            np.save(poc2mol_output_path, result['predicted_ligand_voxels'].detach().float().cpu().numpy())
            np.save(true_ligand_voxels_path, batch['ligand'].detach().float().cpu().numpy())
            np.save(protein_voxels_path, batch['protein'].detach().float().cpu().numpy())

        # convert_smiles_to_rdkit_molecule
        true_smiles = batch['smiles'][0].replace(tokenizer.bos_token, '').replace(tokenizer.eos_token, '').replace("[BOS]", "").replace("[EOS]", "")
        sampled_smiles = result['sampled_smiles'][0].replace("[BOS]", "").replace("[EOS]", "")
        try:
            true_mol = Chem.MolFromSmiles(true_smiles)
            if true_mol is None:
                continue
        except:
            continue
        try:
            sampled_mol = Chem.MolFromSmiles(sampled_smiles)
        except:
            sampled_mol = None

        if sampled_mol is None:
            valid_smiles = 0
        else:
            valid_smiles = 1
            visualisation_batch.append(
                {
                    'true_mol': true_mol,
                    'true_smiles': true_smiles,
                    'sampled_mol': sampled_mol,
                    'sampled_smiles': sampled_smiles,
                    'name': batch['name'][0]
                }
            )
            if len(visualisation_batch) == 9 and not skip_visualisation:
                visualize_2d_molecule_batch(
                    visualisation_batch, 
                    f'{output_dir}/batch_{vis_batch_idx}.png'
                    )
                visualisation_batch = []
                vis_batch_idx += 1
        tanimoto_similarity = get_tanimoto_similarity_from_smiles(true_smiles, sampled_smiles)
        decoy_tanimoto_similarity = get_tanimoto_similarity_from_smiles(decoy_smiles, sampled_smiles)
        true_vs_decoy_tanimoto_similarity = get_tanimoto_similarity_from_smiles(true_smiles, decoy_smiles)
    
        mcs_num_atoms = get_max_common_substructure_num_atoms(true_smiles, sampled_smiles)
        decoy_mcs_num_atoms = get_max_common_substructure_num_atoms(decoy_smiles, sampled_smiles)
        true_vs_decoy_mcs_num_atoms = get_max_common_substructure_num_atoms(true_smiles, decoy_smiles)
        try:
            true_num_heavy_atoms = Chem.MolFromSmiles(true_smiles).GetNumHeavyAtoms()
        except:
            true_num_heavy_atoms = None
            
        # Calculate prop_common_structure for both sampled and decoy
        if sampled_smiles is not None and true_num_heavy_atoms is not None:
            try:
                prop_common_structure = mcs_num_atoms / true_num_heavy_atoms if mcs_num_atoms is not None else None
                decoy_prop_common_structure = decoy_mcs_num_atoms / true_num_heavy_atoms if decoy_mcs_num_atoms is not None else None
            except:
                prop_common_structure = None
                decoy_prop_common_structure = None
        else:
            prop_common_structure = None
            decoy_prop_common_structure = None

        decoy_smiles = true_smiles # use last batch smiles as decoy smiles
        
        result_rows.append(
            {
                'sampled_smiles': sampled_smiles,
                'loss': result['loss'].item(),
                'decoy_loss': result['decoy_loss'].item() if 'decoy_loss' in result else None,
                'name': batch['name'][0],
                'smiles': batch['smiles'][0],
                'valid_smiles': valid_smiles,
                'tanimoto_similarity': tanimoto_similarity,
                'decoy_tanimoto_similarity': decoy_tanimoto_similarity,
                'true_vs_decoy_tanimoto_similarity': true_vs_decoy_tanimoto_similarity,
                'mcs_num_atoms': mcs_num_atoms,
                'decoy_mcs_num_atoms': decoy_mcs_num_atoms,
                'true_vs_decoy_mcs_num_atoms': true_vs_decoy_mcs_num_atoms,
                'prop_common_structure': prop_common_structure,
                'decoy_prop_common_structure': decoy_prop_common_structure,
                'true_num_heavy_atoms': true_num_heavy_atoms,
                'poc2mol_bce': result['poc2mol_bce'],
                'poc2mol_dice': result['poc2mol_dice'],
                'poc2mol_loss': result['poc2mol_loss'],
                'smiles_teacher_forced_accuracy': result['smiles_teacher_forced_accuracy'].item(),
                'decoy_smiles_true_label_teacher_forced_accuracy': result['decoy_smiles_true_label_teacher_forced_accuracy'].item() if 'decoy_smiles_true_label_teacher_forced_accuracy' in result else None,
                'decoy_smiles_decoy_label_teacher_forced_accuracy': result['decoy_smiles_decoy_label_teacher_forced_accuracy'].item() if 'decoy_smiles_decoy_label_teacher_forced_accuracy' in result else None
            }
        )
        df = pd.DataFrame(result_rows)
        df.to_csv(f'{output_dir}/combined_model_results.csv', index=False)

    return result_rows

def main():
    args = parse_args()
    if os.path.exists(f'{args.output_dir}/combined_model_results.csv'):
        df = pd.read_csv(f'{args.output_dir}/combined_model_results.csv')
        generate_plots_from_results_df(df, args.output_dir)
        return

    config = get_config_from_cpt_path(args.ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combined_model = build_combined_model_from_config(
        config, 
        args.ckpt_path,
        eval(args.dtype),
        device
    )
    combined_model.eval()
    complex_dataset_config = config['data']['train_dataset']['poc2mol_output_dataset']['complex_dataset']['config']
    if 'plinder' in args.pdb_dir or 'hiqbind' in args.pdb_dir:
        test_dataloader = build_parquet_test_dataloader(
            complex_dataset_config, 
            args.dtype,
            args.pdb_dir
        )
    else:
        test_dataloader = build_pdb_test_dataloader(
            complex_dataset_config, 
            args.pdb_dir, 
            args.dtype
        )
    results = evaluate_combined_model(
        combined_model, 
        test_dataloader,
        output_dir=args.output_dir,
        save_voxels=True,
        skip_visualisation=args.skip_visualisation
        )
    results_df = pd.DataFrame(results)
    for c in results_df.columns:
        try:
            print(c, round(results_df[c].mean(), 3))
        except:
            pass
    generate_plots_from_results_df(results_df, args.output_dir)




def generate_plots_for_paper(df, output_dir):
    # histogram of hit likelihood versus decoy likelihood (with line showing mean)
    # histogram of sampled mol tanimoto similarity versus decoy tanimoto similarity (with line showing mean)
    # histogram of maximum common substructure of sampled versus decoy
    # histogram of proportion common substructure of sampled versus decoy
    # barplot of means (above) with 95% confidence intervals
    # for each decile of tanimoto similarity and each decile of proportion common structure generate images of
    # the sampled smiles and the true smiles 
    

    pass

if __name__ == "__main__":
    main()