import os
from typing import List, Dict
import copy
import pandas as pd
import numpy as np
import torch
import datetime
import json
import yaml
import argparse
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
from src.evaluation.visual import show_3d_voxel_lig_only
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
"""
This script evaluates a combined model
where we have a checkpoint for the poc2mol model
which has been trained on the HiQBind dataset
then we combine this with a vox2smiles model which has
been trained on the Zinc dataset and the outputs of the 
poc2mol dataset.

For eah model we want to evaluate the following metrics:
- 1:1 hit to decoy enrichment factor (similarity)
- 1 to all hit to decoy similarity enrichment factor
- SMILES ROC AUC score based on likelihoods
- validity 
- proportion of generated molecules are unique
"""


def compute_auc_roc(df):
    y_true = df['is_hit'].values
    y_scores = df['likelihood'].values
    return roc_auc_score(y_true, y_scores)

def compute_auc_pr(df):
    y_true = df['is_hit'].values
    y_scores = df['likelihood'].values
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    return pr_auc

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Poc2Mol model")
    parser.add_argument(
        "--vox2smiles_ckpt_path", 
        type=str, 
        default="logs/CombinedHiQBindCkptFrmPrevCombined/runs/2025-05-06_20-51-46/checkpoints/last.ckpt",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--poc2mol_ckpt_path", 
        type=str, 
        default="logs/poc2mol/runs/2025-04-21_18-13-26/checkpoints/epoch_173.ckpt",
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
        default="torch.float32",
        help="Evaluation dtype"
    )
    parser.add_argument(
        "--skip_visualisation", 
        action="store_true",
    )
    return parser.parse_args()

def build_test_config(complex_dataset_config: DictConfig, dtype: str, system_ids: List[str] = None):
    test_config = copy.deepcopy(complex_dataset_config)
    test_config['random_rotation'] = True
    test_config['random_translation'] = 5.0
    test_config['batch_size'] = 1
    test_config.dtype = dtype
    test_config.remove_hydrogens = True
    test_config.system_ids = system_ids
    return test_config

def build_parquet_test_dataloader(poc2mol_config: DictConfig, dtype: str, pdb_dir: str, system_ids: List[str] = None):
    test_config = build_test_config(poc2mol_config, dtype, system_ids)
    dataset = ParquetDataset(
        config=test_config,
        data_path=pdb_dir,
        use_cluster_member_zero=True
    )
    return DataLoader(dataset, batch_size=1, shuffle=False)


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

def add_smiles_to_batch(batch: dict, tokenizer: PreTrainedTokenizerFast):
    for i, smi in enumerate(batch['smiles']):
        if not smi.startswith(tokenizer.bos_token):
            batch['smiles'][i] = tokenizer.bos_token + smi
        if not smi.endswith(tokenizer.eos_token):
            batch['smiles'][i] = batch['smiles'][i] + tokenizer.eos_token
        
    max_smiles = max([
        len(tokenizer.tokenize(
            smi,
            padding=False,
            truncation=False,
            )) for smi in batch['smiles']
        ])
    
    tokenized = tokenizer(
        batch['smiles'],
        padding='max_length',
        max_length=max_smiles,
        truncation=True,
        return_tensors="pt"
    )
    device  = batch['protein'].device
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    batch.update(tokenized)
    return batch

def get_tanimoto_similarity_from_mol(mol1, mol2):
    if mol1 is None or mol2 is None:
        return None
    morgan_fp1 = AllChem.GetMorganFingerprint(mol1, 2)
    morgan_fp2 = AllChem.GetMorganFingerprint(mol2, 2)
    tanimoto_similarity = DataStructs.TanimotoSimilarity(morgan_fp1, morgan_fp2)
    return tanimoto_similarity

def get_tanimoto_similarity_from_smiles(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        # Return None if molecule creation fails
        if mol1 is None or mol2 is None:
            return None
        tanimoto_similarity = get_tanimoto_similarity_from_mol(mol1, mol2)
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
        skip_visualisation: bool = False,
        test_subset_prefix: str = ""
        ):
    df_savepath = os.path.join(output_dir, f"{test_subset_prefix}combined_model_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
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
            batch = add_smiles_to_batch(batch, tokenizer)
        # Forward pass to obtain Poc2Mol prediction + loss on the *true* SMILES
        result = combined_model.forward(
            protein_voxels=batch['protein'],
            labels=batch['input_ids'],
            ligand_voxels=batch['ligand'],
            sample_smiles=True,
        )


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

        decoy_smiles = true_smiles # use last batch smiles as decoy smiles
        
        result_rows.append(
            {
                'sampled_smiles': sampled_smiles,
                'loss': result['loss'].item(),
                'log_likelihood': -result['loss'].item(),
                'name': batch['name'][0],
                'smiles': batch['smiles'][0].replace("[BOS]", "").replace("[EOS]", ""),
                'valid_smiles': valid_smiles,
                'tanimoto_similarity': tanimoto_similarity,
                'decoy_tanimoto_similarity': decoy_tanimoto_similarity,
                'true_vs_decoy_tanimoto_similarity': true_vs_decoy_tanimoto_similarity,
                'poc2mol_bce': result['poc2mol_bce'],
                'poc2mol_dice': result['poc2mol_dice'],
                'poc2mol_loss': result['poc2mol_loss'],
                'smiles_teacher_forced_accuracy': result['smiles_teacher_forced_accuracy'].item(),
                
            }
        )
        df = pd.DataFrame(result_rows)
        df.to_csv(df_savepath, index=False)

    return result_rows

def summarize_results(results_df, output_dir=None):
    results = dict(
    validity = results_df.valid_smiles.mean(),
    smiles_token_accuracy = results_df.smiles_teacher_forced_accuracy.mean(),
    smiles_loss = results_df.loss.mean(),
    decoy_smiles_loss = results_df.decoy_loss.mean(),
    tanimoto_similarity = results_df.tanimoto_similarity.mean(),
    decoy_tanimoto_similarity = results_df.decoy_tanimoto_similarity.mean(),
    n_mols_gt_0p3_tanimoto = len(results_df[results_df.tanimoto_similarity >= 0.3]),
    tanimoto_enrichment = calculate_enrichment_factor(results_df, target_col="tanimoto_similarity", target_threshold=0.3),
    diversity = len(results_df.sampled_smiles.unique()) / len(results_df),
    n_sampled_mols = len(results_df)
    )
    os.makedirs(output_dir, exist_ok=True)
    results = {k: round(v,4) for k,v in results.items()}
    for k,v in results.items():
        print(f"{k}: {v}")
    if output_dir is not None:
        result_savepath = f"{output_dir}/results_summary.json"
        with open(result_savepath, 'w') as filehandle:
            json.dump(results, filehandle, indent=2)
    return results


def all_decoy_smiles_likelihood_scoring(
        combined_model: CombinedProteinToSmilesModel, 
        test_dataloader: DataLoader,
        df: pd.DataFrame,
        output_dir="evaluation_results",
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get tokenizer configuration from the model config
    config = combined_model.config
    max_smiles_len = config.data.config.max_smiles_len
    
    # Instantiate the tokenizer
    tokenizer = build_smiles_tokenizer()
    results = []
    all_decoy_smiles = df.smiles.values
    os.makedirs(os.path.join(output_dir, "likelihood_scores"), exist_ok=True)
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        likelihoods = []
        decoy_smiles_current_batch = [s for s in all_decoy_smiles if s not in batch['smiles']]
        assert len(decoy_smiles_current_batch) < len(all_decoy_smiles)
        if len(batch['smiles'][0]) > max_smiles_len:
            raise ValueError(f"Smiles length is too long: {len(batch['smiles'][0])}")
        batch = add_smiles_to_batch(batch, tokenizer)

        result = combined_model.forward(
                protein_voxels=batch['protein'], 
                labels=batch['input_ids'], 
                ligand_voxels=None,
                sample_smiles=False,
                )
        likelihood = result['loss'].item()*-1
        predicted_ligand_voxels = result['predicted_ligand_voxels']
        token_accuracy = result['smiles_teacher_forced_accuracy'].item()    
        likelihoods.append({
            'is_hit': 1,
            'likelihood': likelihood,
            'name': batch['name'][0],
            'token_accuracy': token_accuracy,
        })
        for j, decoy_smiles in enumerate(decoy_smiles_current_batch):
            batch['smiles'] = [decoy_smiles]
            batch = add_smiles_to_batch(batch, tokenizer)
        # Get the model output
            decoy_result = combined_model.compute_smiles_metrics(
                ligand_voxels=predicted_ligand_voxels,
                labels=batch['input_ids'],
            )
            decoy_likelihood = decoy_result['loss'].item()*-1
            decoy_token_accuracy = decoy_result['smiles_teacher_forced_accuracy'].item()
            likelihoods.append({
                'is_hit': 0,
                'likelihood': decoy_likelihood,
                'name': batch['name'][0],
                'token_accuracy': decoy_token_accuracy,
            })
        likelihood_output_file = os.path.join(output_dir, "likelihood_scores", f"likelihood_output_{batch['name'][0]}.csv")
        likelihood_df = pd.DataFrame(likelihoods)
        likelihood_df.to_csv(likelihood_output_file, index=False)
        auc_roc_score = compute_auc_roc(likelihood_df)
        auc_pr_score = compute_auc_pr(likelihood_df)
        likelihood_df.sort_values(by='likelihood', ascending=False, inplace=True)
        rank_of_hit = np.where(likelihood_df.is_hit == 1)[0][0]
        print(f"{batch['name'][0]} AUC ROC: {auc_roc_score}, AUC PR: {auc_pr_score}, rank of hit: {rank_of_hit}")
        results.append({
            'system_id': batch['name'][0],
            'auc_roc': auc_roc_score,
            'auc_pr': auc_pr_score,
            'rank_of_hit': rank_of_hit,
            'hit_likelihood': likelihood,
            'hit_teacher_forced_accuracy': result['smiles_teacher_forced_accuracy'].item(),
        })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, "all_decoy_likelihoods.csv"), index=False)
    return results_df



def all_decoy_smiles_likelihood_scoring_batched(
        combined_model: CombinedProteinToSmilesModel,
        test_dataloader: DataLoader,
        df: pd.DataFrame,
        output_dir: str = "evaluation_results",
        smiles_batch_size: int = 128,
        n_pocket_variants: int = 1,
):
    """Batched variant of *all_decoy_smiles_likelihood_scoring* with optional
    pocket augmentation.

    Vox2Smiles model.  Optionally, the evaluation can be repeated
    ``n_pocket_variants`` times for each complex by re-sampling the same
    dataset index, thereby leveraging random rotations / translations inside
    the ``ParquetDataset`` to obtain multiple plausible ligand voxel
    predictions.  The log-likelihood for every candidate SMILES is averaged
    across those variants.

    Note that we manually compute the average log-likelihood for every sample
    since the HuggingFace loss returned by ``VisionEncoderDecoderModel`` is
    averaged across the batch.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Tokeniser and misc. config
    tokenizer = build_smiles_tokenizer()
    config = combined_model.config
    max_smiles_len = config.data.config.max_smiles_len

    all_decoy_smiles = df.smiles.values

    os.makedirs(os.path.join(output_dir, "likelihood_scores"), exist_ok=True)

    results = []  # per-complex summary metrics (AUROC, etc.)

    # Try to infer device from the combined model (fallback to CPU)
    try:
        device = next(combined_model.parameters()).device
    except StopIteration:  # pragma: no cover – should not happen
        device = torch.device("cpu")

    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        system_id = batch['name'][0]

        # Guard against overly long reference SMILES (would violate tokenizer
        # max length assumptions and make comparisons unfair)
        if len(batch['smiles'][0]) > max_smiles_len:
            raise ValueError(
                f"SMILES length for system '{system_id}' exceeds the allowed "
                f"maximum of {max_smiles_len} tokens."
            )

        # ------------------------------------------------------------------
        # 1) Obtain one or more predicted ligand voxel grids by repeatedly
        #    sampling the same pocket (leveraging random transforms inside
        #    the dataset).  The first pass uses the current `batch`, the
        #    remaining passes query `test_dataloader.dataset[i]` directly to
        #    trigger a fresh rotation / translation.
        # ------------------------------------------------------------------
        pocket_voxel_variants = []  # list[(1,C,D,H,W)]

        n_variants = max(1, n_pocket_variants)
        for aug_idx in range(n_variants):
            if aug_idx == 0:
                protein_vox = batch['protein'].to(device)
            else:
                # Re-sample same index to obtain a new random transform.
                aug_sample = test_dataloader.dataset[i]
                protein_vox = aug_sample['protein']
                if protein_vox.dim() == 4:
                    protein_vox = protein_vox.unsqueeze(0)  # add batch dim
                protein_vox = protein_vox.to(device)

            with torch.no_grad():
                poc2mol_out = combined_model.poc2mol_model(protein_vox)
            pocket_voxel_variants.append(poc2mol_out['predicted_ligand_voxels'])
            # img_dir = os.path.join(output_dir, "likelihood_scores", f"pocket_voxel_variants", batch['name'][0])
            # os.makedirs(img_dir, exist_ok=True)
            # show_3d_voxel_lig_only(
            #     poc2mol_out['predicted_ligand_voxels'][0],
            #     angles=[(45,45)],
            #     save_dir=img_dir,
            #     identifier=f"{batch['name'][0]}_pocket_voxel_variant_{aug_idx}",
            #     )
        likelihood_rows = []  # candidate-level results for this pocket

        # ------------------------------------------------------------------
        # 2) Build candidate list: active ligand + decoys (unique)
        # ------------------------------------------------------------------
        active_smiles_raw = batch['smiles'][0]
        active_smiles = active_smiles_raw.replace("[BOS]", "").replace("[EOS]", "")
        decoy_smiles_current = [s for s in all_decoy_smiles if s != active_smiles]
        assert len(decoy_smiles_current) < len(all_decoy_smiles)
        smiles_candidates = [active_smiles] + decoy_smiles_current
        is_hit_flags = [1] + [0] * len(decoy_smiles_current)

        # ------------------------------------------------------------------
        # 3) Iterate over all candidates in mini-batches **for every pocket
        #    variant**, accumulate log-likelihoods / accuracies, then average.
        # ------------------------------------------------------------------

        n_candidates = len(smiles_candidates)
        ll_sums = np.zeros(n_candidates, dtype=np.float32)
        acc_sums = np.zeros(n_candidates, dtype=np.float32)

        for pred_vox in pocket_voxel_variants:
            for start_idx in range(0, n_candidates, smiles_batch_size):
                end_idx = start_idx + smiles_batch_size
                batch_smiles = smiles_candidates[start_idx:end_idx]

                # Ensure BOS/EOS tokens for each SMILES
                batch_smiles_tok = []
                for smi in batch_smiles:
                    if not smi.startswith(tokenizer.bos_token):
                        smi = tokenizer.bos_token + smi
                    if not smi.endswith(tokenizer.eos_token):
                        smi = smi + tokenizer.eos_token
                    batch_smiles_tok.append(smi)

                # Dynamic padding length per mini-batch
                batch_max_len = max(len(s) for s in batch_smiles_tok)
                tokenized = tokenizer(
                    batch_smiles_tok,
                    padding='max_length',
                    max_length=batch_max_len,
                    truncation=True,
                    return_tensors='pt',
                )

                labels = tokenized['input_ids'].to(device)
                pad_id = tokenizer.pad_token_id

                masked_labels = labels.clone()
                masked_labels[masked_labels == pad_id] = -100

                # Duplicate predicted voxels to match candidate batch size
                pred_vox_batch = pred_vox.repeat(labels.size(0), 1, 1, 1, 1)

                with torch.no_grad():
                    vox2_out = combined_model.vox2smiles_model(pred_vox_batch, labels=masked_labels)
                logits = vox2_out.logits  # (B, L, V)

                # a) Log-likelihood per sequence
                log_probs = F.log_softmax(logits, dim=-1)
                gather_idx = labels.clone()
                gather_idx[gather_idx == pad_id] = 0
                token_log_probs = log_probs.gather(-1, gather_idx.unsqueeze(-1)).squeeze(-1)
                valid_mask = labels != pad_id
                token_log_probs = token_log_probs * valid_mask
                seq_lens = valid_mask.sum(dim=1).clamp(min=1)
                seq_log_ll = token_log_probs.sum(dim=1) / seq_lens  # (B,)

                # b) Accuracy
                shift_logits = logits[..., 1:, :]
                shift_labels = masked_labels[..., 1:]
                non_padding_mask = shift_labels != -100
                correct_tokens = (shift_logits.argmax(-1) == shift_labels).float()
                per_seq_acc = (correct_tokens * non_padding_mask).sum(dim=1) / non_padding_mask.sum(dim=1).clamp(min=1)

                # c) Accumulate
                for j in range(labels.size(0)):
                    global_idx = start_idx + j
                    ll_sums[global_idx] += seq_log_ll[j].item()
                    acc_sums[global_idx] += per_seq_acc[j].item()

        # Average across variants
        ll_means = ll_sums / n_variants
        acc_means = acc_sums / n_variants

        for idx in range(n_candidates):
            likelihood_rows.append({
                'is_hit': is_hit_flags[idx],
                'likelihood': ll_means[idx].item() if hasattr(ll_means[idx], 'item') else float(ll_means[idx]),
                'name': system_id,
                'token_accuracy': acc_means[idx].item() if hasattr(acc_means[idx], 'item') else float(acc_means[idx]),
            })

        # ------------------------------------------------------------------
        # 4) Per-pocket evaluation (AUROC, precision-recall, …)
        # ------------------------------------------------------------------
        likelihood_df = pd.DataFrame(likelihood_rows)

        # Persist candidate-level scores for further analysis
        csv_path = os.path.join(
            output_dir,
            "likelihood_scores",
            f"likelihood_output_{system_id}.csv",
        )
        likelihood_df.to_csv(csv_path, index=False)

        # Compute metrics
        auc_roc = compute_auc_roc(likelihood_df)
        auc_pr = compute_auc_pr(likelihood_df)

        likelihood_df_sorted = likelihood_df.sort_values(by='likelihood', ascending=False)
        rank_of_hit = int(np.where(likelihood_df_sorted.is_hit == 1)[0][0])

        print(f"{system_id} AUC ROC: {auc_roc}, AUC PR: {auc_pr}, rank of hit: {rank_of_hit}")

        results.append({
            'system_id': system_id,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'rank_of_hit': rank_of_hit,
            'hit_likelihood': likelihood_df[likelihood_df.is_hit == 1].likelihood.iloc[0],
            'hit_teacher_forced_accuracy': likelihood_df[likelihood_df.is_hit == 1].token_accuracy.iloc[0],
        })

        # Save running summary
        pd.DataFrame(results).to_csv(
            os.path.join(output_dir, "all_decoy_likelihoods_batched.csv"),
            index=False,
        )

    return pd.DataFrame(results)
    

def calculate_enrichment_factor(df, target_col="tanimoto_similarity", target_threshold=0.3):
    model_hitrate = len(df[df[target_col] >= target_threshold]) / len(df)
    baseline_hitrate = len(df[df["decoy_" + target_col]  >= target_threshold] ) / len(df)
    if baseline_hitrate == 0:
        enrichment = 99999
    else:
        enrichment = model_hitrate / baseline_hitrate
    print(
        f"Enrichment factor for {target_col} >= {target_threshold} = {round(enrichment, 3)}"
    )
    return enrichment



def get_plinder_test_split(similarity_df, results_df):
    train_dataset = pd.read_csv("../hiqbind/plixer_train_data.csv")
    plinder_split_df = pd.read_parquet('/mnt/disk2/plinder/2024-06/v2/splits/split.parquet')
    results_df['pdb_id'] = results_df['system_id'].apply(lambda x: x.split("_")[0])
    plinder_split_df['pdb_id'] = plinder_split_df['system_id'].apply(lambda x: x.split("_")[0])
    all_train_clusters = []
    if 'plinder_clusters' not in train_dataset.columns:
        
        train_dataset['pdb_id'] = train_dataset['system_id'].apply(lambda x: x.split("_")[0])
        for i,row in train_dataset.iterrows():
            matches = plinder_split_df[plinder_split_df['pdb_id'] == row['pdb_id']]
            all_train_clusters.extend(list(set(matches['cluster'].values)))
            train_dataset.loc[i, 'plinder_clusters'] = "|".join(list(set(matches['cluster'].values)))
        train_dataset.to_csv("../hiqbind/plixer_train_data.csv", index=False)
    else:
        for i,row in train_dataset.iterrows():
            if not pd.isna(row['plinder_clusters']):
                all_train_clusters.extend(row['plinder_clusters'].split("|"))
    all_train_clusters = list(set(all_train_clusters))
    plinder_test_rows = []
    plinder_test_split = plinder_split_df[~plinder_split_df['cluster'].isin(all_train_clusters)]
    plinder_test_split_pdb_ids = plinder_test_split['pdb_id'].values
    
    for i,row in results_df.iterrows():
        matches = plinder_split_df[plinder_split_df['pdb_id'] == row['pdb_id']]
        if len(matches) > 0 and len(set(matches['cluster'].values) & set(all_train_clusters)) == 0:
            plinder_test_rows.append(row)
    plinder_test_df = pd.DataFrame(plinder_test_rows)
    return plinder_test_df

def compute_all_decoy_similarity_enrichment_factor(results_df, threshold=0.3):
    assert 'smiles' in results_df.columns
    assert 'tanimoto_similarity' in results_df.columns
    for i, row in results_df.iterrows():
        try:
            morgan_fp = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(row['smiles']), 2)
        except:
            morgan_fp = None
        results_df.loc[i, 'morgan_fp'] = morgan_fp
    # results_df['morgan_fp'] = results_df['smiles'].apply(lambda x: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(x), 2))
    all_decoy_similarities = []
    for i, row in results_df.iterrows():
        target_smiles = row['smiles']
        target_fp = row['morgan_fp']
        decoy_smiles = [s for s in results_df['smiles'] if s != target_smiles]
        decoy_fps = [row['morgan_fp'] for i, row in results_df[results_df['smiles'].isin(decoy_smiles)].iterrows() if row['morgan_fp'] is not None]
        decoy_tanimoto_similarities = [DataStructs.TanimotoSimilarity(target_fp, fp) for fp in decoy_fps]
        all_decoy_similarities.extend(decoy_tanimoto_similarities)
    proportion_hits_generated = len(results_df[results_df['tanimoto_similarity'] >= threshold]) / len(results_df)
    proportion_hits_decoy = len([i for sim in all_decoy_similarities if sim >= threshold]) / len(all_decoy_similarities)
    if proportion_hits_decoy == 0:
        enrichment_factor = 99999
    else:
        enrichment_factor = proportion_hits_generated / proportion_hits_decoy
    print(f"All decoy enrichment factor for tanimoto similarity >= {threshold}: {round(enrichment_factor, 3)}")
    return enrichment_factor

def main():
    args = parse_args()
    test_df_paths = [
        # ('chrono_', 'data/test_set_chronological_split.csv'),
        ('plinder_', 'data/test_set_plinder_split.csv'),
        ('seq_sim_', 'data/test_set_seq_sim_split.csv'),
        
    ]
    for split_name, test_df_path in test_df_paths:
        output_dir = os.path.join(args.output_dir, split_name.replace("_", ""))
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(f'{args.output_dir}/combined_model_results_all_decoy_likelihoods.csv'):
            similarity_df = pd.read_csv("../hiqbind/similarity_analysis/test_similarities.csv")
            results_df = pd.read_csv(f'{args.output_dir}/combined_model_results_all_decoy_likelihoods.csv')
            sequence_split_system_ids = similarity_df[
                (similarity_df.max_protein_similarity < 0.3)
                |(similarity_df.max_protein_similarity.isna())
                ].system_id.values
            plinder_test_split = get_plinder_test_split(similarity_df, results_df).copy()
            results_df_sequence = results_df[results_df['system_id'].isin(sequence_split_system_ids)].copy()
            print("\nFull test set:")
            compute_all_decoy_similarity_enrichment_factor(results_df, threshold=0.3)
            metrics = summarize_results(results_df, output_dir=args.output_dir)
            print("\nsequence split:")
            compute_all_decoy_similarity_enrichment_factor(results_df_sequence, threshold=0.3)
            metrics = summarize_results(results_df_sequence, output_dir=os.path.join(args.output_dir, "sequence"))
            print("\nPlinder test split:")
            compute_all_decoy_similarity_enrichment_factor(plinder_test_split, threshold=0.3)
            metrics = summarize_results(plinder_test_split, output_dir=os.path.join(args.output_dir, "plinder"))
            generate_plots_from_results_df(results_df, args.output_dir, vis_deciles=False, similarity_df=similarity_df)
            generate_plots_from_results_df(results_df_sequence, os.path.join(args.output_dir, "sequence"), vis_deciles=False, similarity_df=similarity_df)
            generate_plots_from_results_df(plinder_test_split, os.path.join(args.output_dir, "plinder"), vis_deciles=False, similarity_df=similarity_df)
            return

        config = get_config_from_cpt_path(args.vox2smiles_ckpt_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        combined_model = build_combined_model_from_config(
            config=config, 
            vox2smiles_ckpt_path=args.vox2smiles_ckpt_path,
            poc2mol_ckpt_path=args.poc2mol_ckpt_path,
            dtype=eval(args.dtype),
            device=device
        )
        combined_model.eval()
        combined_model.vox2smiles_model.eval()
        combined_model.poc2mol_model.eval()
        complex_dataset_config = config['data']['train_dataset']['poc2mol_output_dataset']['complex_dataset']['config']
        if 'plinder' in args.pdb_dir or 'hiqbind' in args.pdb_dir:
            test_dataloader = build_parquet_test_dataloader(
                complex_dataset_config, 
                args.dtype,
                args.pdb_dir,
                system_ids=list(pd.read_csv(test_df_path)['system_id'].values)
            )
        else:
            test_dataloader = build_pdb_test_dataloader(
                complex_dataset_config, 
                args.pdb_dir, 
                args.dtype,
            )
        smiles_likelihood_results = all_decoy_smiles_likelihood_scoring_batched(
            combined_model, 
            test_dataloader,
            df = pd.read_csv(test_df_path),
            output_dir=os.path.join(output_dir, "plixer_likelihood_scores"),
            smiles_batch_size=24,
            n_pocket_variants=1,
        )
        results = evaluate_combined_model(
            combined_model, 
            test_dataloader,
            output_dir=output_dir,
            save_voxels=True,
            skip_visualisation=args.skip_visualisation,
            test_subset_prefix=split_name,
            )
        results_df = pd.DataFrame(results)
        for c in results_df.columns:
            try:
                print(c, round(results_df[c].mean(), 3))
            except:
                pass
        metrics = summarize_results(results_df, output_dir=output_dir)
        generate_plots_from_results_df(results_df, output_dir)

if __name__ == "__main__":
    main()