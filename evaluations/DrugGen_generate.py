# from transformers import pipeline

# pipe = pipeline("text-generation", model="alimotahharynia/DrugGen")
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("alimotahharynia/DrugGen")
# model = AutoModelForCausalLM.from_pretrained("alimotahharynia/DrugGen")


# if __name__=="__main__":
#     pass

import os
import torch
import logging
import pandas as pd
import numpy as np
import json
from transformers import AutoTokenizer, GPT2LMHeadModel
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import tqdm
# Import functions from evaluate_combined_vox2smiles.py
from evaluations.evaluate_combined_vox2smiles import (
    get_tanimoto_similarity_from_smiles,
    get_max_common_substructure_num_atoms,
    calculate_enrichment_factor,
    summarize_results,
)

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

# Global logging setup
def setup_logging(output_file):
    log_filename = os.path.splitext(output_file)[0] + ".log"

    logging.getLogger().handlers.clear()

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# Load model and tokenizer
def load_model_and_tokenizer(model_name):
    logging.info(f"Loading model and tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        # Move the model to CUDA if available
        if torch.cuda.is_available():
            logging.info("Moving model to CUDA device.")
            model = model.to("cuda")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model and tokenizer: {e}")
        raise RuntimeError(f"Failed to load model and tokenizer: {e}")

class SMILESGenerator:
    def __init__(self, model=None, tokenizer=None, uniprot_to_sequence=None, output_file="generated_SMILES.txt"):

        self.output_file = output_file

        # Generation parameters
        self.config = {
            "generation_kwargs": {
                "do_sample": True,
                "top_k": 9,
                "max_length": 1024,
                "top_p": 0.9,
                "num_return_sequences": 10
            },
            "max_retries": 30
        }

        self.model = model
        self.tokenizer = tokenizer
        self.uniprot_to_sequence = uniprot_to_sequence

        # Generation device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Adjust generation parameters with token IDs
        self.generation_kwargs = self.config["generation_kwargs"]
        self.generation_kwargs["bos_token_id"] = self.tokenizer.bos_token_id
        self.generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        self.generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

    def score_smiles(self, sequence, smiles):
        with torch.no_grad():
            prompt = f"<|startoftext|><P>{sequence}<L>{smiles}<|endoftext|>"
            mask_prompt = f"<|startoftext|><P>{sequence}<L>"
            encoded_prompt = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
            encoded_mask_prompt = self.tokenizer(mask_prompt, return_tensors="pt")["input_ids"].to(self.device)
            labels = encoded_prompt.clone()
            labels[0,:encoded_mask_prompt.shape[-1]] = -100
            if encoded_prompt.shape[-1] > 1024:
                return None
            loss = self.model.forward(encoded_prompt, labels=labels).loss
            return loss.item() * -1
    
    
    def generate_smiles(self, sequence, num_generated):
        generated_smiles_set = set()
        prompt = f"<|startoftext|><P>{sequence}<L>"
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        retries = 0

        logging.info(f"Generating SMILES for sequence: {sequence[:10]}...")

        sum_length = 0
        num_gen = 0

        while len(generated_smiles_set) < num_generated:
            if retries >= self.config["max_retries"]:
                logging.warning("Max retries reached. Returning what has been generated so far.")
                break
            if encoded_prompt.shape[-1] > 1024:
                logging.warning(f"Encoded prompt is longer than max length, shape: {encoded_prompt.shape}")
                return ['']
            sample_outputs = self.model.generate(encoded_prompt, **self.generation_kwargs)
            for sample_output in sample_outputs:
                output_decode = self.tokenizer.decode(sample_output, skip_special_tokens=False)
                try:
                    generated_smiles = output_decode.split("<L>")[1].split("<|endoftext|>")[0]
                    if generated_smiles not in generated_smiles_set:
                        sum_length += len(generated_smiles)
                        num_gen += 1
                        print(len(generated_smiles), generated_smiles)
                        generated_smiles_set.add(generated_smiles)
                except (IndexError, AttributeError) as e:
                    logging.warning(f"Failed to parse SMILES due to error: {str(e)}. Skipping.")

            retries += 1

        logging.info(f"SMILES generation for sequence completed. Generated {len(generated_smiles_set)} SMILES.")
        if num_gen > 0:
            logging.info(f"Average SMILES length: {sum_length / num_gen:.2f}")
        return list(generated_smiles_set)
        
    
    
    def generate_all_decoy_likelihoods(self, df, save_dir, k=1000):
        all_auc_roc_scores = []
        all_auc_pr_scores = []
        rank_scores = []
        results = []
        os.makedirs(save_dir, exist_ok=True)
        for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            output_file = os.path.join(save_dir, f"{row['system_id']}_likelihoods.csv")
            if os.path.exists(output_file):
                likelihood_df = pd.read_csv(output_file)
            else:
                likelihoods = []
                likelihood = self.score_smiles(row['protein_sequence'], row['smiles'])
                if likelihood is None:
                    continue
                likelihoods.append({
                    'is_hit': 1,
                    'likelihood': likelihood,
                    })
                for j, decoy_row in df.iterrows():
                    if i == j:
                        continue
                    likelihood = self.score_smiles(row['protein_sequence'], decoy_row['smiles'])
                    if likelihood is None:
                        continue
                    likelihoods.append({
                        'is_hit': 0,
                        'likelihood': likelihood
                        })
                likelihood_df = pd.DataFrame(likelihoods)
                likelihood_df.to_csv(output_file, index=False)
            auc_roc_score = compute_auc_roc(likelihood_df)
            auc_pr_score = compute_auc_pr(likelihood_df)
            all_auc_roc_scores.append(auc_roc_score)
            all_auc_pr_scores.append(auc_pr_score)
            likelihood_df.sort_values(by='likelihood', ascending=False, inplace=True)
            rank_of_hit = np.where(likelihood_df.is_hit == 1)[0][0]
            rank_scores.append(rank_of_hit) 
            print(f"{row['system_id']} AUC ROC: {auc_roc_score}, AUC PR: {auc_pr_score}, rank of hit: {rank_of_hit}")
            results.append({
                'system_id': row['system_id'],
                'auc_roc': auc_roc_score,
                'auc_pr': auc_pr_score,
                'rank_of_hit': rank_of_hit
            })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(save_dir, "auc_roc_results.csv"), index=False)
        return results_df
    
    
    def generate_likelihoods(self, df):
        for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            if i == 0 or ('likelihood' in row and not pd.isna(row.likelihood)):
                continue
            decoy_smiles = df.iloc[i-1]['smiles']
            sequence = row['protein_sequence']
            smiles = row['smiles']
            try:
                true_likelihood = self.score_smiles(sequence, smiles)
                decoy_likelihood = self.score_smiles(sequence, decoy_smiles)
            except Exception as e:
                logging.warning(f"Error scoring SMILES for {row['system_id']}: {e}")
                true_likelihood = None
                decoy_likelihood = None
                return df
            df.loc[i, 'likelihood'] = true_likelihood
            df.loc[i, 'decoy_likelihood'] = decoy_likelihood
        return df
    
    def generate_smiles_data(self, list_of_sequences=None, num_generated=2):
        sequences_input = []

        if list_of_sequences:
            sequences_input.extend(list_of_sequences)

        data = []
        for sequence in sequences_input:
            smiles = self.generate_smiles(sequence, num_generated)
            uniprot_id = next((uid for uid, seq in self.uniprot_to_sequence.items() if seq == sequence), None)
            data.append({"system_id": uniprot_id, "sequence": sequence, "SMILES": smiles[-1]})


        logging.info(f"Completed SMILES generation for {len(data)} entries.")
        return pd.DataFrame(data)

def build_plixer_test_dataset(result_path="evaluation_results/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol/combined_model_results.csv"):
    # columns are system_id,protein_sequence,smiles
    test_df = pd.read_csv("../hiqbind/plixer_test_data.csv")
    plixer_test_results = pd.read_csv(result_path)
    for i, row in plixer_test_results.iterrows():
        plixer_test_results.loc[i, 'protein_sequence'] = test_df[test_df.system_id == row['name']].iloc[0]['protein_sequence']
        plixer_test_results.loc[i, 'system_id'] = row['name']
    return plixer_test_results

def compute_all_decoy_similarity_enrichment_factor(plixer_test_df, results_df, threshold=0.3):
    for i, row in plixer_test_df.iterrows():
        true_smiles = row.smiles.replace("[BOS]","").replace("[EOS]", "")
        try:
            morgan_fp = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(true_smiles), 2)
        except:
            morgan_fp = None
        plixer_test_df.loc[i, 'morgan_fp'] = morgan_fp
    all_decoy_similarities = []
    for i, row in plixer_test_df.iterrows():
        target_smiles = row['smiles']
        target_fp = row['morgan_fp']
        decoy_smiles = [s for s in plixer_test_df['smiles'] if s != target_smiles]
        decoy_fps = [row['morgan_fp'] for i, row in plixer_test_df[plixer_test_df['smiles'].isin(decoy_smiles)].iterrows() if row['morgan_fp'] is not None]
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

def analyse_df(
    df, 
    plixer_csv="evaluation_results/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol/combined_model_results.csv",
    output_dir="evaluation_results/druggen_analysis_v2",
    experiment_name="",
    ):
    logging.info("Analyzing generated SMILES against true molecules...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the plixer dataframe to get true SMILES
    plixer_df = pd.read_csv(plixer_csv)
    
    # Prepare results collection
    result_rows = []
    decoy_smiles = None  # Will use previous row as decoy
    all_decoys = []
    # Process each row in the dataframe
    for i, row in df.iterrows():
        generated_smiles = row.SMILES
        
        # Find the matching true SMILES from plixer dataframe
        plixer_match = plixer_df[plixer_df.name == row.system_id]
        if len(plixer_match) == 0:
            logging.warning(f"No match found for system_id {row.system_id} in plixer data")
            continue
            
        true_smiles = plixer_match.iloc[0].smiles.replace(
            "[BOS]", ""
        ).replace(
            "[EOS]", ""
        )
        
        # Check SMILES validity
        try:
            true_mol = Chem.MolFromSmiles(true_smiles)
            if true_mol is None:
                logging.warning(f"Invalid true SMILES for {row.system_id}: {true_smiles}")
                continue
        except Exception as e:
            logging.warning(f"Error processing true SMILES for {row.system_id}: {e}")
            continue
            
        try:
            sampled_mol = Chem.MolFromSmiles(generated_smiles)
        except Exception as e:
            sampled_mol = None
            logging.warning(f"Error processing generated SMILES for {row.system_id}: {e}")
            
        # Validity check
        valid_smiles = 1 if sampled_mol is not None else 0
        
        # Calculate similarities when possible
        tanimoto_similarity = get_tanimoto_similarity_from_smiles(true_smiles, generated_smiles)
        decoy_tanimoto_similarity = get_tanimoto_similarity_from_smiles(decoy_smiles, generated_smiles) if decoy_smiles else None
        true_vs_decoy_tanimoto_similarity = get_tanimoto_similarity_from_smiles(true_smiles, decoy_smiles) if decoy_smiles else None
        
        placeholder_loss = float('nan')
        
        # Store result for this row
        result_rows.append({
            'system_id': row.system_id,
            'sampled_smiles': generated_smiles,
            'loss': placeholder_loss,
            'decoy_loss': placeholder_loss,
            'log_likelihood': -placeholder_loss,
            'decoy_log_likelihood': -placeholder_loss,
            'name': row.system_id,
            'smiles': true_smiles,
            'valid_smiles': valid_smiles,
            'tanimoto_similarity': tanimoto_similarity,
            'decoy_tanimoto_similarity': decoy_tanimoto_similarity,
            'true_vs_decoy_tanimoto_similarity': true_vs_decoy_tanimoto_similarity,
            'sequence': row.sequence,
            'smiles_likelihood_auc_roc': row.auc_roc,
        })
        
        # Use this row's true SMILES as decoy for next row
        decoy_smiles = true_smiles
    
    # Create dataframe from results
    results_df = pd.DataFrame(result_rows)
    
    # Save results
    results_df.to_csv(f'{output_dir}/generation_{experiment_name}_druggen_results.csv', index=False)
    all_decoy_enrichment_factor = compute_all_decoy_similarity_enrichment_factor(
        plixer_test_df=plixer_df, 
        results_df=results_df, 
        threshold=0.3
        )
    summary_savepath = f"{output_dir}/druggen_{experiment_name}_results_summary.json"
    drug_summarize_results(results_df, summary_savepath, all_decoy_enrichment=all_decoy_enrichment_factor)    
    return results_df

# Custom summarize_results function because the imported one might have different parameters
def drug_summarize_results(results_df, savepath=None, all_decoy_enrichment=None):
    """Summarize the evaluation results."""
    if len(results_df) == 0:
        logging.warning("No valid results to summarize")
        return {}
    
    # Calculate metrics
    results = dict(
        validity = results_df.valid_smiles.mean(),
        tanimoto_similarity = results_df.tanimoto_similarity.mean(),
        decoy_tanimoto_similarity = results_df.decoy_tanimoto_similarity.mean() if 'decoy_tanimoto_similarity' in results_df.columns else None,
        n_mols_gt_0p3_tanimoto = len(results_df[results_df.tanimoto_similarity >= 0.3]),
        tanimoto_enrichment = calculate_enrichment_factor(results_df, target_col="tanimoto_similarity", target_threshold=0.3) 
                              if 'decoy_tanimoto_similarity' in results_df.columns else None,
        proportion_mols_are_unique = len(results_df.sampled_smiles.unique()) / len(results_df),
        smiles_likelihood_auc_roc = results_df.smiles_likelihood_auc_roc.mean()
    )
    if all_decoy_enrichment is not None:
        results["all_decoy_similarity_enrichment"] = all_decoy_enrichment
    # Round and clean up results
    results = {k: round(v, 4) if isinstance(v, (int, float)) and not pd.isna(v) else v 
               for k, v in results.items() if v is not None}
    
    # Print results
    for k, v in results.items():
        print(f"{k}: {v}")
        
    # Save results if output directory is provided
    if savepath is not None:
        with open(savepath, 'w') as filehandle:
            json.dump(results, filehandle, indent=2)
            
    return results

# Main function for inference
def run_inference(num_generated=1, output_file="evaluation_results/druggen_v2/drug_gen_on_plixer_test_set_w_likelihoods.csv"):
    # Setup logging
    output_dir=os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_file)
    model_name = "alimotahharynia/DrugGen"
    model, tokenizer = load_model_and_tokenizer(model_name)
    for ds_path in [
        "evaluation_results/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol/combined_model_results_20250514_190923.csv",
        "evaluation_results/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol/plinder_test_split_results.csv",
        "evaluation_results/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol/seq_sim_test_split_results.csv"
    ]:
        experiment_name = os.path.basename(ds_path).replace(".csv", "")
        print(f"\n\nRunning experiment: {experiment_name}")
        experiment_output_file = output_file.replace(".csv", f"_{experiment_name}.csv")
        dataset = build_plixer_test_dataset(
            result_path=ds_path
        )
        id_to_sequence = {row["system_id"]: row["protein_sequence"] for i, row in dataset.iterrows()}

        # Initialize the generator
        generator = SMILESGenerator(model, tokenizer, id_to_sequence, output_file=experiment_output_file)
        # if os.path.exists(experiment_output_file):
        #     dataset = pd.read_csv(experiment_output_file)
        likelihoods_df = generator.generate_all_decoy_likelihoods(dataset, save_dir=f"{output_dir}/{experiment_name}_all_decoy_likelihoods")
        logging.info("Starting SMILES generation process...")
        if os.path.exists(experiment_output_file):
            df = pd.read_csv(experiment_output_file, sep="\t")
        else:
            # Generate SMILES data
            df = generator.generate_smiles_data(
                list_of_sequences=id_to_sequence.values(),
                num_generated=num_generated
            )

            # Save the output
            df.to_csv(experiment_output_file, sep="\t", index=False)
            print(f"Generated SMILES saved to {experiment_output_file}")
        # add auc roc score to generations df
        for i, row in df.iterrows():
            match = likelihoods_df[likelihoods_df.system_id==row.system_id]
            if len(match==1):
                df.loc[i, 'auc_roc'] = match.iloc[0].auc_roc
        analyse_df(df, plixer_csv=ds_path, output_dir=output_dir, experiment_name=experiment_name)
        

if __name__=="__main__":
    run_inference()