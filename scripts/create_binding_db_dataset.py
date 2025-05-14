import pandas as pd
import os
import tempfile
import subprocess
from tqdm import tqdm
import numpy as np
from Bio.PDB import PDBParser, PPBuilder
import json
import glob
import logging
from pathlib import Path
import shutil

from create_hiqbind_dataset import build_mmseqs_database, batch_search_protein_similarities

"""
Goal is to create a test dataset using BindingDB where we 
find protein systems that have nothing similar in the training 
set. There are 3,012,191 entries in BindingDB

As a first filter we remove all the entries where one of the PDB IDs
matches a PDB ID in the training set

Next we accumulate all of the sequences in BindingDB (from all chains longer than 50 residues)
we create a mmseqsDB of all chains (one entry per chain)
and search this against all of the chains in the training dataset

if any chains have a sequence identity of greater than 30% to the training set at 50% coverage
then we remove the system from BindingDB
The remaining rows are then used as a strict test split.

BindingDB columns:
BindingDB Reactant_set_id
Ligand SMILES
Ligand InChI
Ligand InChI Key
BindingDB MonomerID
BindingDB Ligand Name
Target Name
Target Source Organism According to Curator or DataSource
Ki (nM)
IC50 (nM)
Kd (nM)
EC50 (nM)
kon (M-1-s-1)
koff (s-1)
pH
Temp (C)
Curation/DataSource
Article DOI
Link to Ligand-Target Pair in BindingDB
Ligand HET ID in PDB
PDB ID(s) for Ligand-Target Complex
PubChem CID
PubChem SID
ChEBI ID of Ligand
ChEMBL ID of Ligand
DrugBank ID of Ligand
IUPHAR_GRAC ID of Ligand
KEGG ID of Ligand
ZINC ID of Ligand
Number of Protein Chains in Target (>1 implies a multichain complex)
BindingDB Target Chain Sequence
PDB ID(s) of Target Chain
UniProt (SwissProt) Recommended Name of Target Chain
UniProt (SwissProt) Entry Name of Target Chain
UniProt (SwissProt) Primary ID of Target Chain
UniProt (SwissProt) Secondary ID(s) of Target Chain
UniProt (SwissProt) Alternative ID(s) of Target Chain
UniProt (TrEMBL) Submitted Name of Target Chain
UniProt (TrEMBL) Entry Name of Target Chain
UniProt (TrEMBL) Primary ID of Target Chain
UniProt (TrEMBL) Secondary ID(s) of Target Chain
UniProt (TrEMBL) Alternative ID(s) of Target Chain
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def system_id_pdb_id(system_id):
    return system_id.split("_")[0]

def extract_sequences_from_bindingdb(bindingdb_df):
    """Extract protein sequences from BindingDB DataFrame."""
    sequences = {}
    
    # Function to process a single sequence field
    def process_sequence_field(row, sequence_col, id_prefix):
        sequence = row[sequence_col]
        if isinstance(sequence, str) and len(sequence) >= 50:
            # Create a unique ID for each sequence
            seq_id = f"{id_prefix}_{row['BindingDB Reactant_set_id']}"
            sequences[seq_id] = sequence
    
    # Process all protein sequence columns
    for i in range(50):  # There are up to 50 chains in the BindingDB data
        seq_col = f"BindingDB Target Chain Sequence{'.'+str(i) if i > 0 else ''}"
        if seq_col in bindingdb_df.columns:
            for idx, row in bindingdb_df.iterrows():
                process_sequence_field(row, seq_col, f"chain_{i}")
    
    log.info(f"Extracted {len(sequences)} sequences from BindingDB")
    return sequences

def aggregate_pdb_ids(bindingdb_df):
    pdb_id_cols = [c for c in bindingdb_df.columns if "PDB ID(s) of Target Chain" in c]
    for i, row in bindingdb_df.iterrows():
        pdb_ids = []
        for pdb_id_col in pdb_id_cols:
            if pd.notna(row[pdb_id_col]) and isinstance(row[pdb_id_col], str) and len(row[pdb_id_col]) > 0:
                pdb_ids.extend(row[pdb_id_col].split(","))
        pdb_ids = list(set(pdb_ids))
        bindingdb_df.at[i, "pdb_ids"] = ",".join(pdb_ids)
    return bindingdb_df

def remove_rows_where_pdb_id_is_in_training_set(df, train_pdb_ids):
    
    def has_training_pdb(pdb_ids_str):
        if not isinstance(pdb_ids_str, str):
            return False
        pdb_ids = pdb_ids_str.lower().split(',')
        return any(pdb_id.strip() in train_pdb_ids for pdb_id in pdb_ids)
        
    pdb_id_col = "pdb_ids"
    filtered_df = df[~df[pdb_id_col].apply(has_training_pdb)]
    log.info(f"After PDB ID filtering: {len(filtered_df)}/{len(df)} entries remaining")
    return filtered_df

def perform_mmseqs_similarity_filtering(
        bindingdb_df, 
        train_db, 
        tmpdir, 
        similarity_threshold=30, 
        output_dir="../BindingDB/filtered_chunks",
        c=0.1,
        min_alnlen=50
        ):
    """
    Filter BindingDB entries based on sequence similarity to training set.
    Returns the filtered DataFrame and a DataFrame with similarity scores.
    
    Args:
        bindingdb_df: DataFrame containing BindingDB entries
        train_db: Path to the pre-built training database
        tmpdir: Directory for temporary files
        similarity_threshold: Maximum allowed sequence identity percentage
        output_dir: Directory to save output files
        
    Returns:
        filtered_df: DataFrame with entries below similarity threshold
        similarities_df: DataFrame with similarity scores
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract protein sequences from BindingDB chunk
    bindingdb_seqs = extract_sequences_from_bindingdb(bindingdb_df)
    
    if not bindingdb_seqs:
        log.warning("No valid sequences found in this chunk")
        # Return an empty similarity DataFrame with the right structure
        similarity_df = pd.DataFrame(columns=["query", "target", "pident", "alnlen", "qlen", "tlen"])
        bindingdb_df["max_sequence_identity"] = 0.0
        return bindingdb_df, similarity_df
    
    # Build MMSEQS database only for the current chunk sequences
    bindingdb_db = build_mmseqs_database(bindingdb_seqs, tmpdir, "bindingdb_db")
    
    # Perform MMSEQS search against the pre-built training database
    similarities_df = batch_search_protein_similarities(bindingdb_db, train_db, tmpdir, c=0.1)
    similarities_df = similarities_df[similarities_df["alnlen"] >= min_alnlen]
    # Create a mapping from sequence IDs to BindingDB entry IDs
    entry_to_max_identity = {}
    
    if not similarities_df.empty:
        similarities_df["BindingDB Reactant_set_id"] = similarities_df["query"].apply(lambda x: x.split("_")[-1])
        similarity_lookup = similarities_df[["BindingDB Reactant_set_id", "pident"]]
        similarity_lookup = similarity_lookup.groupby("BindingDB Reactant_set_id").max().reset_index()
        entry_to_max_identity = dict(zip(similarity_lookup["BindingDB Reactant_set_id"].astype(int), similarity_lookup["pident"]))


    
    # Add the maximum sequence identity to the DataFrame
    bindingdb_df["max_sequence_identity"] = bindingdb_df["BindingDB Reactant_set_id"].map(
        lambda x: entry_to_max_identity.get(x, 0.0)
    )
    
    # Filter out entries with sequence identity above the threshold
    filtered_df = bindingdb_df[bindingdb_df["max_sequence_identity"] <= similarity_threshold]
    
    log.info(f"After sequence similarity filtering: {len(filtered_df)}/{len(bindingdb_df)} entries remaining")
    
    return filtered_df, similarities_df

def main():
    # Load training data
    train_save_path = '../hiqbind/plixer_train_data.csv'
    log.info(f"Loading training data from {train_save_path}")
    train_df = pd.read_csv(train_save_path)
    train_df["pdb_id"] = train_df["system_id"].apply(system_id_pdb_id)
    
    # Create a set of PDB IDs in the training set for quick filtering
    train_pdb_ids = set(train_df["pdb_id"].str.lower())
    log.info(f"Found {len(train_pdb_ids)} unique PDB IDs in the training set")
    
    # Extract protein sequences from training set
    log.info("Extracting protein sequences from training set")
    train_seqs = {row["system_id"]: row["protein_sequence"] for _, row in train_df.iterrows() if pd.notna(row["protein_sequence"])}
    
    # Create output directories
    output_base_dir = "../BindingDB"
    filtered_chunks_dir = os.path.join(output_base_dir, "filtered_chunks")
    similarity_data_dir = os.path.join(output_base_dir, "similarity_data")
    mmseqs_dir = os.path.join(output_base_dir, "mmseqs_db")
    os.makedirs(filtered_chunks_dir, exist_ok=True)
    os.makedirs(similarity_data_dir, exist_ok=True)
    os.makedirs(mmseqs_dir, exist_ok=True)
    
    # Load BindingDB data
    bindingdb_path = "../BindingDB/BindingDB_All.tsv"
    if not os.path.exists(bindingdb_path):
        log.error(f"BindingDB file not found at {bindingdb_path}")
        return
    
    log.info(f"Processing BindingDB data from {bindingdb_path}")
    
    # Check for any previously processed chunks
    existing_chunks = glob.glob(os.path.join(filtered_chunks_dir, "chunk_*.csv"))
    start_chunk = len(existing_chunks)
    log.info(f"Found {start_chunk} previously processed chunks")
    
    # Process the data in chunks
    filtered_chunks = []
    chunk_size = 10000
    chunk_id = start_chunk

    # Create a persistent directory for the training database
    # Build the training database once in a dedicated location
    train_db_path = os.path.join(mmseqs_dir, "train_db")
    
    # Check if training database already exists
    if os.path.exists(train_db_path) and os.path.exists(f"{train_db_path}.dbtype"):
        log.info(f"Using existing training database at {train_db_path}")
        train_db = train_db_path
    else:
        # Create a temporary directory for building the training database
        with tempfile.TemporaryDirectory() as tmp_train_dir:
            log.info("Building training sequence database with MMSEQS")
            train_db = build_mmseqs_database(train_seqs, tmp_train_dir, "train_db")
            
            # Copy the training database to the persistent location
            for f in glob.glob(f"{train_db}*"):
                dst = os.path.join(mmseqs_dir, os.path.basename(f))
                shutil.copy2(f, dst)
            
            # Update the path to the persistent location
            train_db = train_db_path
            log.info(f"Training database built and stored at {train_db}")
    
    # Process chunks with a temporary directory for each chunk
    for chunk in tqdm(pd.read_csv(bindingdb_path, sep='\t', chunksize=chunk_size, 
                                  skiprows=range(1, chunk_id*chunk_size) if chunk_id > 0 else None), 
                      desc="Processing BindingDB chunks"):
        log.info(f"Processing chunk {chunk_id} with {len(chunk)} entries")
        try:
            # Aggregate PDB IDs
            chunk = aggregate_pdb_ids(chunk)
            
            # First filter: Remove entries with PDB IDs in training set
            chunk = remove_rows_where_pdb_id_is_in_training_set(chunk, train_pdb_ids)
            
            if len(chunk) > 0:
                # Create a temporary directory for this chunk processing
                with tempfile.TemporaryDirectory() as chunk_tmpdir:
                    # Second filter: Remove entries with similar protein sequences
                    filtered_chunk, similarities_df = perform_mmseqs_similarity_filtering(
                        chunk, train_db, chunk_tmpdir, similarity_threshold=30, output_dir=filtered_chunks_dir
                    )
                    
                    # Save the filtered chunk
                    chunk_path = os.path.join(filtered_chunks_dir, f"chunk_{chunk_id}.csv")
                    filtered_chunk.to_csv(chunk_path, index=False)
                    log.info(f"Saved filtered chunk to {chunk_path}")
                    
                    # Save similarity data
                    if not similarities_df.empty:
                        sim_path = os.path.join(similarity_data_dir, f"similarity_chunk_{chunk_id}.csv")
                        similarities_df.to_csv(sim_path, index=False)
                        log.info(f"Saved similarity data to {sim_path}")
                    
                    # Append to filtered chunks list
                    filtered_chunks.append(filtered_chunk)
            
            chunk_id += 1
        except Exception as e:
            chunk_id += 1
            log.error(f"Error processing chunk {chunk_id}: {e}")
            continue
    
    # Combine all filtered chunks
    if filtered_chunks:
        final_df = pd.concat(filtered_chunks, ignore_index=True)
        log.info(f"Final dataset after filtering: {len(final_df)} entries")
        
        # Save the final dataset
        output_path = os.path.join(output_base_dir, "bindingdb_test_set.csv")
        final_df.to_csv(output_path, index=False)
        log.info(f"Saved final test dataset to {output_path}")
        
        # Save a separate file with just the sequence identity information
        seq_identity_df = final_df[["BindingDB Reactant_set_id", "max_sequence_identity"]]
        seq_identity_path = os.path.join(output_base_dir, "bindingdb_sequence_identity.csv")
        seq_identity_df.to_csv(seq_identity_path, index=False)
        log.info(f"Saved sequence identity data to {seq_identity_path}")
    else:
        log.warning("No data remained after filtering")

if __name__ == "__main__":
    main()