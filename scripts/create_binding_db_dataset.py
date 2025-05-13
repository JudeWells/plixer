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
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def system_id_pdb_id(system_id):
    return system_id.split("_")[0]

def extract_protein_sequence(pdb_path: str) -> str:
    """Extract the amino-acid sequence from a PDB file (first model, all chains)."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("struct", pdb_path)
    except Exception as e:
        log.error(f"Error parsing PDB file {pdb_path}: {e}")
        return ""
    ppb = PPBuilder()
    seq = ""
    for pp in ppb.build_peptides(structure):
        seq += str(pp.get_sequence())
    return seq

def build_mmseqs_database(sequences: dict[str, str], tmp_dir: str, db_prefix: str) -> str:
    """Build an MMSEQS database from a dictionary of sequences.
    
    Args:
        sequences: Dictionary mapping sequence IDs to sequences
        tmp_dir: Temporary directory for MMSEQS files
        db_prefix: Prefix for the output database files
        
    Returns:
        Path to the created database
    """
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Write sequences to fasta file
    fasta_path = os.path.join(tmp_dir, f"{db_prefix}.fasta")
    with open(fasta_path, "w") as f:
        for seq_id, seq in sequences.items():
            if seq and len(seq) >= 50:  # Only include sequences longer than 50 residues
                f.write(f">{seq_id}\n{seq}\n")
    
    # Create MMSEQS database
    db_path = os.path.join(tmp_dir, db_prefix)
    cmd = [
        "mmseqs", "createdb",
        fasta_path,
        db_path,
        "--compressed", "1"
    ]
    subprocess.run(cmd, check=True)
    
    return db_path

def batch_search_protein_similarities(query_db: str, target_db: str, tmp_dir: str) -> pd.DataFrame:
    """Perform batch MMSEQS search between query and target databases.
    
    Args:
        query_db: Path to query database
        target_db: Path to target database
        tmp_dir: Temporary directory for MMSEQS files
        
    Returns:
        DataFrame with search results
    """
    # Run MMSEQS search
    search_prefix = os.path.join(tmp_dir, "search_out")
    cmd = [
        "mmseqs", "search",
        query_db,
        target_db,
        search_prefix,
        tmp_dir,
        "--min-seq-id", "0.3",   # 30% sequence identity threshold
        "-c", "0.5",            # 50% coverage threshold
        "--threads", "20",
        "--remove-tmp-files", "0",  # Keep temporary files for convertalis
        "-v", "1",
    ]
    subprocess.run(cmd, check=True)
    
    # Convert search results to BLAST format
    result_file = os.path.join(tmp_dir, "results.m8")
    cmd = [
        "mmseqs", "convertalis",
        query_db,
        target_db,
        search_prefix,
        result_file,
        "--format-output", "query,target,pident,qcov,qlen,alnlen",
        "--threads", "20"
    ]
    subprocess.run(cmd, check=True)
    
    # Parse results
    similarities = []
    colnames = ["query", "target", "pident", "qcov", "qlen", "alnlen"]
    if os.path.exists(result_file):
        with open(result_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == len(colnames):
                    new_row = dict(zip(colnames, parts))
                    similarities.append(new_row)
    df = pd.DataFrame(similarities)
    if not df.empty:
        df["pident"] = df["pident"].astype(float)
        df["qcov"] = df["qcov"].astype(float)
        df["qlen"] = df["qlen"].astype(int)
        df["alnlen"] = df["alnlen"].astype(int)
    return df

def extract_sequences_from_bindingdb(bindingdb_df):
    """Extract protein sequences from BindingDB DataFrame."""
    sequences = {}
    
    # Function to process a single sequence field
    def process_sequence_field(row, sequence_col, id_prefix):
        sequence = row[sequence_col]
        if isinstance(sequence, str) and len(sequence) >= 50:
            # Create a unique ID for each sequence
            seq_id = f"{id_prefix}_{row.name}"
            sequences[seq_id] = sequence
    
    # Process all protein sequence columns
    for i in range(50):  # There are up to 50 chains in the BindingDB data
        seq_col = f"BindingDB Target Chain Sequence{'.'+str(i) if i > 0 else ''}"
        if seq_col in bindingdb_df.columns:
            for idx, row in tqdm(bindingdb_df.iterrows(), desc=f"Processing {seq_col}", total=len(bindingdb_df)):
                process_sequence_field(row, seq_col, f"chain_{i}")
    
    log.info(f"Extracted {len(sequences)} sequences from BindingDB")
    return sequences

def aggregate_pdb_ids(bindingdb_df):
    pdb_id_cols = [c for c in bindingdb_df.columns if "PDB ID(s) of Target Chain" in c]
    for i, row in bindingdb_df.iterrows():
        pdb_ids = []
        for pdb_id_col in pdb_id_cols:
            if pd.notna(row[pdb_id_col]) and len(row[pdb_id_col]) > 0:
                pdb_ids.extend(row[pdb_id_col].split(","))
        pdb_ids = list(set(pdb_ids))
        bindingdb_df.at[i, "pdb_ids"] = ",".join(pdb_ids)
    return bindingdb_df

def main():
    # Load training data
    train_save_path = '../hiqbind/plixer_train_data.csv'
    log.info(f"Loading training data from {train_save_path}")
    train_df = pd.read_csv(train_save_path)
    train_df["pdb_id"] = train_df["system_id"].apply(system_id_pdb_id)
    
    # Create a set of PDB IDs in the training set for quick filtering
    train_pdb_ids = set(train_df["pdb_id"].str.lower())
    log.info(f"Found {len(train_pdb_ids)} unique PDB IDs in the training set")
    
    # Load BindingDB data
    bindingdb_path = "../BindingDB/BindingDB_All.tsv"
    if not os.path.exists(bindingdb_path):
        log.error(f"BindingDB file not found at {bindingdb_path}")
        return
    
    log.info(f"Loading BindingDB data from {bindingdb_path}")
    # Using a chunksize to handle the large file
    chunks = []
    for chunk in tqdm(pd.read_csv(bindingdb_path, sep='\t', chunksize=100000), desc="Loading BindingDB data"):
        chunks.append(chunk)
        break
    bindingdb_df = pd.concat(chunks)
    log.info(f"Loaded {len(bindingdb_df)} entries from BindingDB")
    bindingdb_df = aggregate_pdb_ids(bindingdb_df)
    
    # First filter: Remove entries with PDB IDs in the training set
    pdb_id_col = "pdb_ids"
    if pdb_id_col in bindingdb_df.columns:
        # Function to check if any PDB ID from an entry matches training set PDB IDs
        def has_training_pdb(pdb_ids_str):
            if not isinstance(pdb_ids_str, str):
                return False
            pdb_ids = pdb_ids_str.lower().split(',')
            return any(pdb_id.strip() in train_pdb_ids for pdb_id in pdb_ids)
        
        # Filter out entries with matching PDB IDs
        filtered_df = bindingdb_df[~bindingdb_df[pdb_id_col].apply(has_training_pdb)]
        log.info(f"After PDB ID filtering: {len(filtered_df)}/{len(bindingdb_df)} entries remaining")
    else:
        filtered_df = bindingdb_df
        log.warning(f"PDB ID column '{pdb_id_col}' not found in BindingDB data")
    
    # Second filter: Remove entries with protein sequences similar to training set
    with tempfile.TemporaryDirectory() as tmpd:
        log.info("Extracting protein sequences from BindingDB")
        bindingdb_seqs = extract_sequences_from_bindingdb(filtered_df)
        
        log.info("Extracting protein sequences from training set")
        train_seqs = {row["system_id"]: row["protein_sequence"] for _, row in train_df.iterrows() if pd.notna(row["protein_sequence"])}
        
        log.info("Building MMSEQS databases")
        bindingdb_db = build_mmseqs_database(bindingdb_seqs, tmpd, "bindingdb_db")
        train_db = build_mmseqs_database(train_seqs, tmpd, "train_db")
        
        log.info("Performing MMSEQS search to find similar proteins")
        similarities_df = batch_search_protein_similarities(bindingdb_db, train_db, tmpd)
        
        if similarities_df.empty:
            log.info("No similar proteins found between BindingDB and training set")
            similar_entries = set()
        else:
            # Extract the entry IDs that have similar proteins
            similar_entries = set()
            for query in similarities_df["query"]:
                # Extract the original row ID from the query ID
                parts = query.split('_')
                if len(parts) >= 2:
                    try:
                        entry_id = int(parts[-1])
                        similar_entries.add(entry_id)
                    except ValueError:
                        continue
            
            log.info(f"Found {len(similar_entries)} BindingDB entries with similar proteins to training set")
        
        # Filter out entries with similar proteins
        final_df = filtered_df[~filtered_df.index.isin(similar_entries)]
        log.info(f"Final dataset after protein similarity filtering: {len(final_df)}/{len(filtered_df)} entries")
        
        # Save the final dataset
        output_path = "../BindingDB/bindingdb_test_set.csv"
        final_df.to_csv(output_path, index=False)
        log.info(f"Saved final test dataset to {output_path}")
        
        # Save summary statistics
        summary = {
            "original_entries": len(bindingdb_df),
            "after_pdb_filtering": len(filtered_df),
            "final_entries": len(final_df),
            "filtered_by_pdb": len(bindingdb_df) - len(filtered_df),
            "filtered_by_protein_similarity": len(filtered_df) - len(final_df)
        }
        
        with open("bindingdb_filtering_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        log.info("Saved filtering summary to bindingdb_filtering_summary.json")

if __name__ == "__main__":
    main()




