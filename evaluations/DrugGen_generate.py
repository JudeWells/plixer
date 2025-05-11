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
# Import functions from evaluate_combined_vox2smiles.py
from evaluations.evaluate_combined_vox2smiles import (
    get_tanimoto_similarity_from_smiles,
    get_max_common_substructure_num_atoms,
    calculate_enrichment_factor,
    calculate_likelihood_enrichment_factor,
    summarize_results
)

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

    def generate_smiles_data(self, list_of_sequences=None, num_generated=10):
        sequences_input = []

        if list_of_sequences:
            sequences_input.extend(list_of_sequences)

        data = []
        for sequence in sequences_input:
            smiles = self.generate_smiles(sequence, num_generated)
            uniprot_id = next((uid for uid, seq in self.uniprot_to_sequence.items() if seq == sequence), None)
            data.append({"system_id": uniprot_id, "sequence": sequence, "SMILES": smiles[0]})


        logging.info(f"Completed SMILES generation for {len(data)} entries.")
        return pd.DataFrame(data)

def build_plixer_test_dataset():
    # columns are system_id,protein_sequence,smiles
    test_df = pd.read_csv("../hiqbind/plixer_test_data.csv")
    plixer_test_results = pd.read_csv("evaluation_results/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol/combined_model_results.csv")
    test_df = test_df[test_df.system_id.isin(plixer_test_results.name)]
    return test_df

def analyse_df(
    df, 
    plixer_csv="evaluation_results/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol/combined_model_results.csv",
    output_dir="evaluation_results/druggen_analysis"
    ):
    logging.info("Analyzing generated SMILES against true molecules...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the plixer dataframe to get true SMILES
    plixer_df = pd.read_csv(plixer_csv)
    
    # Prepare results collection
    result_rows = []
    decoy_smiles = None  # Will use previous row as decoy
    
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
        
        # Calculate maximum common substructure
        mcs_num_atoms = get_max_common_substructure_num_atoms(true_smiles, generated_smiles)
        decoy_mcs_num_atoms = get_max_common_substructure_num_atoms(decoy_smiles, generated_smiles) if decoy_smiles else None
        true_vs_decoy_mcs_num_atoms = get_max_common_substructure_num_atoms(true_smiles, decoy_smiles) if decoy_smiles else None
        
        # Get number of heavy atoms in true molecule
        try:
            true_num_heavy_atoms = true_mol.GetNumHeavyAtoms()
        except:
            true_num_heavy_atoms = None
            
        # Calculate proportion of common structure
        if valid_smiles and true_num_heavy_atoms is not None and true_num_heavy_atoms > 0:
            prop_common_structure = mcs_num_atoms / true_num_heavy_atoms if mcs_num_atoms is not None else None
            decoy_prop_common_structure = decoy_mcs_num_atoms / true_num_heavy_atoms if decoy_mcs_num_atoms is not None else None
        else:
            prop_common_structure = None
            decoy_prop_common_structure = None
            
        # For metrics like loss/log-likelihood where we would typically have model outputs,
        # we'll use placeholder values since we're only working with final SMILES strings
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
            'mcs_num_atoms': mcs_num_atoms,
            'decoy_mcs_num_atoms': decoy_mcs_num_atoms,
            'true_vs_decoy_mcs_num_atoms': true_vs_decoy_mcs_num_atoms,
            'prop_common_structure': prop_common_structure,
            'decoy_prop_common_structure': decoy_prop_common_structure,
            'true_num_heavy_atoms': true_num_heavy_atoms,
            'sequence': row.sequence
        })
        
        # Use this row's true SMILES as decoy for next row
        decoy_smiles = true_smiles
    
    # Create dataframe from results
    results_df = pd.DataFrame(result_rows)
    
    # Save results
    results_df.to_csv(f'{output_dir}/druggen_results.csv', index=False)
    
    # Summarize results using the imported function
    drug_summarize_results(results_df, output_dir)
    
    return results_df

# Custom summarize_results function because the imported one might have different parameters
def drug_summarize_results(results_df, output_dir=None):
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
        proportion_mols_are_unique = len(results_df.sampled_smiles.unique()) / len(results_df)
    )
    
    # Include likelihood enrichment only if we have those values
    if not pd.isna(results_df.log_likelihood).all() and 'decoy_log_likelihood' in results_df.columns:
        results['likelihood_enrichment'] = calculate_likelihood_enrichment_factor(
            results_df, target_col="log_likelihood", top_k=min(100, len(results_df))
        )
    
    # Round and clean up results
    results = {k: round(v, 4) if isinstance(v, (int, float)) and not pd.isna(v) else v 
               for k, v in results.items() if v is not None}
    
    # Print results
    for k, v in results.items():
        print(f"{k}: {v}")
        
    # Save results if output directory is provided
    if output_dir is not None:
        result_savepath = f"{output_dir}/results_summary.json"
        with open(result_savepath, 'w') as filehandle:
            json.dump(results, filehandle, indent=2)
            
    return results

# Main function for inference
def run_inference(num_generated=1, output_file="evaluation_results/drug_gen_on_plixer_test_set.csv"):
    # Setup logging
    setup_logging(output_file)
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)

    else:
        model_name = "alimotahharynia/DrugGen"
        model, tokenizer = load_model_and_tokenizer(model_name)

        dataset = build_plixer_test_dataset()
        id_to_sequence = {row["system_id"]: row["protein_sequence"] for i, row in dataset.iterrows()}

        # Initialize the generator
        generator = SMILESGenerator(model, tokenizer, id_to_sequence, output_file=output_file)
        logging.info("Starting SMILES generation process...")
        
        # Generate SMILES data
        df = generator.generate_smiles_data(
            list_of_sequences=id_to_sequence.values(),
            num_generated=num_generated
        )

        # Save the output
        df.to_csv(output_file, sep="\t", index=False)
        print(f"Generated SMILES saved to {output_file}")
    analyse_df(df)
    

if __name__=="__main__":
    run_inference()