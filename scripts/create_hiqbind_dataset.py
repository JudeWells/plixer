"""
Created by Jude Wells 2025-04-21

HiQBind was downloaded from:
https://figshare.com/articles/dataset/BioLiP2-Opt_Dataset/27430305?file=52379423
on 2025-04-21

We create a chronological split of the data 2020, 2021 and 2022 are used for testing
all previous results are used for training and validation.

Independent of the (train + val) / test split, we also do clustering on both the ligand and the 
protein sequence. We choose 10% of the protein clusters to be the validation set

To avoid reppetition during training we sample from combined cluster where the combined cluster is defined
as the combination of the protein cluster and the ligand cluster.

For protein clustering: MMSEQS easy cluster with 30% sequence identity 50% coverage.

For the ligand clustering: EFCP finger prints and tanimoto


For the test dataset we will do a non-redundancy reduction which considers both the protein and the ligand.
For this we sample 1 test example from for each protein-ligand cluster pair.

The validation set will consis of 10% of the training dataset clusters (based on protein sequence only).

We will exclude the polymer subset of the dataset for now only using:
../hiqbind/raw_data_hiq_sm
and ../hiqbind/hiqbind_metadata.csv

Metadata colummns:
PDBID,Resolution,Year,Ligand Name,Ligand Chain,Ligand Residue Number,Binding Affinity Measurement,Binding Affinity Sign,Binding Affinity Value,Binding Affinity Unit,Log Binding Affinity,Binding Affinity Source,Binding Affinity Annotation,Protein UniProtID,Protein UniProtName,Ligand SMILES,Ligand MW,Ligand LogP,Ligand TPSA,Ligand NumRotBond,Ligand NumHeavyAtoms,Ligand NumHDon,Ligand NumHAcc,Ligand QED

../hiqbind/raw_data_hiq_sm$ ls | head
10gs
13gs
181l

Note that some entries have multiple chains:
cd 7du9
(VenvVoxDiff) judewells@kaspian:/mnt/disk2/VoxelDiffOuter/hiqbind/raw_data_hiq_sm/7du9$ ls
7du9_Q4J_A_1101  7du9_Q4J_B_1101

cd 7du9_Q4J_A_1101/
(VenvVoxDiff) judewells@kaspian:/mnt/disk2/VoxelDiffOuter/hiqbind/raw_data_hiq_sm/7du9/7du9_Q4J_A_1101$ ls
7du9_Q4J_A_1101_hetatm.pdb          7du9_Q4J_A_1101_protein_hetatm.pdb
7du9_Q4J_A_1101_ligand.pdb          7du9_Q4J_A_1101_protein.pdb
7du9_Q4J_A_1101_ligand_refined.sdf  7du9_Q4J_A_1101_protein_refined.pdb

Use the refined version of the PDB and the the sdf file.
"""
import os
import subprocess
import glob
import pandas as pd
import argparse
import logging
import shutil
from tqdm import tqdm
from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
from Bio.PDB import PDBParser, PPBuilder
from collections import defaultdict
import json
import tempfile
from docktgrid.molparser import MolecularData
from src.data.docktgrid_mods import MolecularParserWrapper
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def run_mmseqs_easy_cluster(fasta_file, out_prefix, min_seq_id=0.9, coverage=0.8, threads=20):
    """
    Run mmseqs easy-cluster on a fasta file with specified sequence identity and coverage thresholds.
    Returns the path to the representative sequences fasta file.
    """
    cmd = [
        "mmseqs", "easy-cluster",
        fasta_file,
        out_prefix,
        out_prefix,
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--threads", str(threads),
        "--remove-tmp-files", "1",
        "--cluster-mode", "1",
        "-v", "1",  # 0=silent, 1=errors, 2=warnings
    ]
    print(f"Running mmseqs: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def parse_mmseqs_clusters(cluster_tsv: str):
    """Parse mmseqs easy-cluster tsv file and return mapping from sequence id to cluster id (representative).

    The tsv produced by mmseqs easy-cluster has two whitespace‑separated columns:
        <seq_id>  <cluster_rep>
    where <cluster_rep> is the representative sequence for the cluster.
    """
    mapping = {}
    if cluster_tsv is None or not os.path.exists(cluster_tsv):
        return mapping
    with open(cluster_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            cluster_rep, seq_id = parts[0], parts[1]
            mapping[seq_id] = cluster_rep
    return mapping


def process_system(system_dir: str, parser: MolecularParserWrapper, min_atoms: int = 10):
    """Process a HiQBind system, returning MolecularData objects and SMILES.

    Ensures exactly one ligand per system.
    Returns (protein_data, ligand_data, smiles) or (None, None, None) if unsuitable.
    """
    # Locate protein file
    pdb_files = glob.glob(os.path.join(system_dir, "*_protein_refined.pdb"))
    if not pdb_files:
        pdb_files = glob.glob(os.path.join(system_dir, "*_protein.pdb"))
    if len(pdb_files) != 1:
        assert not pdb_files, f"Expected 1 protein file, got {len(pdb_files)} for {system_dir}"
        return None, None, None
    protein_path = pdb_files[0]

    # Locate single ligand file
    ligand_files = glob.glob(os.path.join(system_dir, "*_ligand_refined.sdf"))
    if not ligand_files:
        ligand_files = glob.glob(os.path.join(system_dir, "*_ligand.sdf"))
    if not ligand_files:
        ligand_files = glob.glob(os.path.join(system_dir, "*_ligand.pdb"))
    assert len(ligand_files) == 1, f"Expected exactly 1 ligand file for {system_dir}"
    ligand_file = ligand_files[0]

    # Parse ligand with RDKit
    if ligand_file.endswith(".sdf"):
        suppl = Chem.SDMolSupplier(ligand_file, removeHs=False)
        mol = suppl[0] if suppl and len(suppl) > 0 else None
    else:
        mol = Chem.MolFromPDBFile(ligand_file, removeHs=False)
    if mol is None:
        return None, None, None

    if mol.GetNumAtoms() < min_atoms:
        return None, None, None

    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))

    # Convert ligand to temporary PDB for parser
    with tempfile.TemporaryDirectory() as tmpd:
        lig_pdb = os.path.join(tmpd, "ligand.pdb")
        Chem.MolToPDBFile(mol, lig_pdb)
        protein_data: MolecularData = parser.parse_file(protein_path, '.pdb')
        ligand_data: MolecularData = parser.parse_file(lig_pdb, '.pdb')

    return protein_data, ligand_data, smiles


def compute_cluster_weights(df: pd.DataFrame):
    """Compute inverse frequency weights by cluster."""
    cluster_counts = df["cluster"].value_counts().astype(float)
    return 1.0 / cluster_counts


def extract_protein_sequence(pdb_path: str) -> str:
    """Extract the amino‑acid sequence from a PDB file (first model, all chains)."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("struct", pdb_path)
    except Exception:
        return ""
    ppb = PPBuilder()
    seq = ""
    for pp in ppb.build_peptides(structure):
        seq += str(pp.get_sequence())
    return seq


def build_fasta(seqs: dict[str, str], fasta_path: str):
    """Write sequences dict{name:sequence} to fasta file."""
    with open(fasta_path, "w") as fh:
        for name, seq in seqs.items():
            if seq:
                fh.write(f">{name}\n{seq}\n")


def run_protein_clustering(raw_dir: str, tmp_dir: str) -> dict[str, str]:
    """Extract sequences, run mmseqs clustering, return mapping system_id -> protein cluster rep."""
    os.makedirs(tmp_dir, exist_ok=True)
    fasta_path = "../hiqbind/hiqbind_proteins.fasta"
    if not os.path.exists(fasta_path):
        seqs = {}
        for system_dir in glob.glob(os.path.join(raw_dir, "*/*")):
            if not os.path.isdir(system_dir):
                continue
            pdb_files = glob.glob(os.path.join(system_dir, "*_protein_refined.pdb"))
            if not pdb_files:
                pdb_files = glob.glob(os.path.join(system_dir, "*_protein.pdb"))
            if not pdb_files:
                continue
            seq = extract_protein_sequence(pdb_files[0])
            if seq:
                seqs[os.path.basename(system_dir)] = seq
        build_fasta(seqs, fasta_path)
    # Run mmseqs
    out_prefix = os.path.join(tmp_dir, "mmseqs_out")
    run_mmseqs_easy_cluster(fasta_path, out_prefix, min_seq_id=0.3, coverage=0.5)
    cluster_tsv = out_prefix + "_cluster.tsv"
    
    mapping_raw = parse_mmseqs_clusters(cluster_tsv)
    # Map representative ids to incremental cluster numbers
    rep_to_idx = {}
    system_to_cluster = {}
    for sid, rep in mapping_raw.items():
        if rep not in rep_to_idx:
            rep_to_idx[rep] = len(rep_to_idx)
        system_to_cluster[sid] = rep_to_idx[rep]
    return system_to_cluster


def cluster_ligands(smiles_list: list[str], cutoff: float = 0.3) -> dict[str, int]:
    """Cluster ligands with Butina on ECFP4 fingerprints."""
    unique_smiles = list({s for s in smiles_list if s})
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024) for s in unique_smiles]
    # Compute distance matrix (1‑tan).
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)
    smi_to_cluster = {}
    for cidx, cluster in enumerate(clusters):
        for idx in cluster:
            smi_to_cluster[unique_smiles[idx]] = cidx
    # Any smile that failed to parse gets own cluster
    for s in smiles_list:
        if s not in smi_to_cluster:
            smi_to_cluster[s] = len(smi_to_cluster)
    return smi_to_cluster


def create_indices(input_dir: str):
    """Generate cluster/file mapping indices inside *input_dir*/indices using same logic as Plinder script."""
    for split_dir in ["train", "val", "test"]:
        output_dir = os.path.join(input_dir, split_dir, "indices")
        os.makedirs(output_dir, exist_ok=True)
        parquet_files = glob.glob(os.path.join(input_dir, split_dir, "*.parquet"))
        if not parquet_files:
            return
        all_samples, file_indices = [], []
        cluster_samples = defaultdict(list)
        for file_idx, file_path in enumerate(parquet_files):
            df = pd.read_parquet(file_path)
            for row_idx, row in df.iterrows():
                cluster_id = row["cluster"]
                all_samples.append({"system_id": row["system_id"], "cluster": cluster_id})
                file_indices.append(file_idx)
                cluster_samples[cluster_id].append({"file_idx": file_idx, "system_id": row["system_id"], "row_idx": int(row_idx)})
        
        with open(os.path.join(output_dir, "global_index.json"), "w") as f:
            json.dump({"samples": all_samples, "file_indices": file_indices}, f, indent=2)
        with open(os.path.join(output_dir, "cluster_index.json"), "w") as f:
            json.dump(cluster_samples, f, indent=2)
        with open(os.path.join(output_dir, "file_mapping.json"), "w") as f:
            json.dump({i: os.path.basename(p) for i, p in enumerate(parquet_files)}, f, indent=2)
        with open(os.path.join(output_dir, "index_summary.json"), "w") as f:
            json.dump({"total_samples": len(all_samples), "total_clusters": len(cluster_samples), "total_files": len(parquet_files)}, f, indent=2)


def build_numeric_protein_clusters(raw_dir: str, tsv_path: str | None) -> dict[str, int]:
    """Load existing mmseqs cluster mapping or build it if TSV not present.

    Returns dict system_id -> numeric protein_cluster_id."""
    if tsv_path and os.path.exists(tsv_path):
        mapping_raw = parse_mmseqs_clusters(tsv_path)
        rep_to_idx: dict[str, int] = {}
        numeric_map: dict[str, int] = {}
        for sid, rep in mapping_raw.items():
            if rep not in rep_to_idx:
                rep_to_idx[rep] = len(rep_to_idx)
            numeric_map[sid] = rep_to_idx[rep]
        return numeric_map

    tmp_dir = Path(tsv_path).parent / "mmseqs_tmp" if tsv_path else Path("./mmseqs_tmp")
    numeric_map = run_protein_clustering(raw_dir, str(tmp_dir))

    if tsv_path:
        generated_tsv = str(Path(tmp_dir) / "mmseqs_out_cluster.tsv")
        if os.path.exists(generated_tsv):
            shutil.copy(generated_tsv, tsv_path)
    return numeric_map


def build_ligand_clusters(meta_df: pd.DataFrame, smiles_col: str = "Ligand SMILES", out_json: str | None = None) -> dict[str, int]:
    """Load ligand clusters mapping if json exists else build and save."""
    if out_json and os.path.exists(out_json):
        with open(out_json) as f:
            mapping = json.load(f)
        return {k: int(v) for k, v in mapping.items()}

    smiles_list = meta_df[smiles_col].dropna().astype(str).tolist()
    mapping = cluster_ligands(smiles_list)

    if out_json:
        with open(out_json, "w") as f:
            json.dump(mapping, f, indent=2)
    return mapping


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


def batch_search_protein_similarities(query_db: str, target_db: str, tmp_dir: str) -> dict[str, dict[str, float]]:
    """Perform batch MMSEQS search between query and target databases.
    
    Args:
        query_db: Path to query database
        target_db: Path to target database
        tmp_dir: Temporary directory for MMSEQS files
        
    Returns:
        Dictionary mapping query IDs to dictionaries of target IDs and their similarities
    """
    # Run MMSEQS search
    search_prefix = os.path.join(tmp_dir, "search_out")
    cmd = [
        "mmseqs", "search",
        query_db,
        target_db,
        search_prefix,
        tmp_dir,
        "--min-seq-id", "0.1",
        "-c", "0.5",
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
    df["pident"] = df["pident"].astype(float)
    df["qcov"] = df["qcov"].astype(float)
    df["qlen"] = df["qlen"].astype(int)
    df["alnlen"] = df["alnlen"].astype(int)
    return df


def calculate_similarity_scores(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate maximum similarity scores for test set entries relative to training set using batch MMSEQS search."""
    with tempfile.TemporaryDirectory() as tmpd:
        # Build sequence dictionaries
        test_seqs = {row["system_id"]: row["protein_sequence"] for _, row in test_df.iterrows() if row["protein_sequence"]}
        train_seqs = {row["system_id"]: row["protein_sequence"] for _, row in train_df.iterrows() if row["protein_sequence"]}
        
        # Build MMSEQS databases
        test_db = build_mmseqs_database(test_seqs, tmpd, "test_db")
        train_db = build_mmseqs_database(train_seqs, tmpd, "train_db")
        
        # Perform batch search
        sim_df = batch_search_protein_similarities(test_db, train_db, tmpd)
        max_protein_sims = sim_df.groupby("query")["pident"].max().to_dict()
        
        # Calculate ligand similarities (keeping existing ligand similarity calculation)
        max_ligand_sims = {}
        for _, test_row in tqdm(test_df.iterrows(), desc="Calculating ligand similarities"):
            test_id = test_row["system_id"]
            test_smiles = test_row["smiles"]
            if test_id not in max_protein_sims:
                max_protein_sims[test_id] = 0.0
            ligand_sims = []
            for _, train_row in train_df.iterrows():
                train_smiles = train_row["smiles"]
                if train_smiles:
                    sim = calculate_ligand_similarity(test_smiles, train_smiles)
                    ligand_sims.append(sim)
            max_ligand_sims[test_id] = max(ligand_sims) if ligand_sims else 0.0
        
        # Add similarity scores to test dataframe
        test_df["max_protein_similarity"] = test_df["system_id"].map(max_protein_sims)
        test_df["max_ligand_similarity"] = test_df["system_id"].map(max_ligand_sims)
        
        return test_df


def calculate_protein_similarity(seq1: str, seq2: str) -> float:
    """Calculate sequence similarity between two protein sequences using MMSEQS."""
    with tempfile.TemporaryDirectory() as tmpd:
        # Write sequences to temporary fasta files
        seq1_path = os.path.join(tmpd, "seq1.fasta")
        seq2_path = os.path.join(tmpd, "seq2.fasta")
        with open(seq1_path, "w") as f:
            f.write(f">seq1\n{seq1}\n")
        with open(seq2_path, "w") as f:
            f.write(f">seq2\n{seq2}\n")
        
        # Run MMSEQS search
        out_prefix = os.path.join(tmpd, "mmseqs_out")
        cmd = [
            "mmseqs", "search",
            seq1_path,
            seq2_path,
            out_prefix,
            tmpd,
            "--min-seq-id", "0.0",
            "-c", "0.0",
            "--threads", "1",
            "--remove-tmp-files", "1",
            "-v", "1",
        ]
        subprocess.run(cmd, check=True)
        
        # Parse results
        result_file = out_prefix + ".m8"
        if not os.path.exists(result_file):
            return 0.0
        
        with open(result_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    return float(parts[2])  # Return sequence identity
        return 0.0


def calculate_ligand_similarity(smiles1: str, smiles2: str) -> float:
    """Calculate Tanimoto similarity between two ligands using ECFP4 fingerprints."""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def main():
    parser = argparse.ArgumentParser(description="Create HiQBind parquet dataset compatible with ParquetDataset")
    parser.add_argument("--raw_dir", default="../hiqbind/raw_data_hiq_sm", help="Directory containing raw HiQBind structures")
    parser.add_argument("--metadata", default="../hiqbind/hiqbind_metadata.csv", help="Path to HiQBind metadata CSV")
    parser.add_argument("--output_dir", default="../hiqbind/parquet", help="Directory to save parquet batches")
    parser.add_argument("--min_atoms", type=int, default=10, help="Minimum heavy atoms for ligand")
    parser.add_argument("--entries_per_file", type=int, default=32, help="Maximum number of rows per output parquet file")
    parser.add_argument("--mmseqs_cluster_tsv", default="../hiqbind/protein_clusters.tsv", help="Path to mmseqs cluster TSV (will be generated if absent)")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of train clusters to use as validation")
    parser.add_argument("--test_year_start", type=int, default=2020, help="Year (inclusive) for test split")
    parser.add_argument("--calculate_similarities", action="store_true", help="Calculate similarity scores for test set")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.metadata):
        raise FileNotFoundError(f"Metadata CSV not found at {args.metadata}")
    meta_df = pd.read_csv(args.metadata)

    pdb_col = "PDBID"
    year_col = "Year"

    pdb_to_year = dict(zip(meta_df[pdb_col].astype(str).str.lower(), meta_df[year_col].astype(int)))
    parser = MolecularParserWrapper()

    protein_cluster_map = build_numeric_protein_clusters(args.raw_dir, args.mmseqs_cluster_tsv)

    ligand_cluster_json = os.path.join(args.output_dir, "ligand_clusters.json")
    ligand_cluster_map = build_ligand_clusters(meta_df, out_json=ligand_cluster_json)
    unique_cluster_ids = np.array(list(set(protein_cluster_map.values())))
    n_val_cluster_ids = len(unique_cluster_ids) // 10
    np.random.seed(42)
    validation_cluster_ids = np.random.choice(unique_cluster_ids, size=n_val_cluster_ids, replace=False)
    batch_counters = {"train": 0, "val": 0, "test": 0}
    buffers = {"train": [], "val": [], "test": []}
    saved_files = []

    def flush_split(split_name: str):
        if not buffers[split_name]:
            return
        outdir = os.path.join(args.output_dir, split_name)
        os.makedirs(outdir, exist_ok=True)
        batch_idx = batch_counters[split_name]
        out_path = os.path.join(outdir, f"{split_name}_batch_{batch_idx:04d}.parquet")
        pd.DataFrame(buffers[split_name]).to_parquet(out_path, index=False)
        log.info(f"Flushed {len(buffers[split_name])} rows to {out_path}")
        saved_files.append(out_path)
        buffers[split_name].clear()
        batch_counters[split_name] += 1

    pdb_dirs = [d for d in glob.glob(os.path.join(args.raw_dir, "*")) if os.path.isdir(d)]
    log.info(f"Found {len(pdb_dirs)} PDB parent directories in {args.raw_dir}")

    for pdb_dir in tqdm(pdb_dirs, desc="Processing PDB folders"):
        pdb_code = os.path.basename(pdb_dir).lower()
        subdirs = [sd for sd in glob.glob(os.path.join(pdb_dir, "*")) if os.path.isdir(sd)]
        for system_dir in subdirs:
            system_id = os.path.basename(system_dir)
            try:
                # Determine cluster from protein mapping (requires SMILES after parsed)
                protein_data, ligand_data, smiles = process_system(system_dir, parser, min_atoms=args.min_atoms)
                if protein_data is None:
                    continue

                protein_cluster_id = protein_cluster_map.get(system_id, protein_cluster_map.get(pdb_code, -1))
                ligand_cluster_id = ligand_cluster_map.get(smiles, -1)

                # Determine split based on year
                year = pdb_to_year.get(pdb_code)
                if year is None or year < args.test_year_start:
                    if protein_cluster_id in validation_cluster_ids:
                        split = "val"
                    else:
                        split = "train"
                else:
                    split = "test"

                # Flatten coords and shapes
                protein_coords_flat = protein_data.coords.float().numpy().astype(np.float16).flatten()
                ligand_coords_flat = ligand_data.coords.float().numpy().astype(np.float16).flatten()

                row = {
                    "system_id": system_id,
                    "smiles": smiles,
                    "protein_coords": protein_coords_flat,
                    "protein_coords_shape": protein_data.coords.shape,
                    "protein_element_symbols": protein_data.element_symbols.astype('U'),
                    "ligand_coords": ligand_coords_flat,
                    "ligand_coords_shape": ligand_data.coords.shape,
                    "ligand_element_symbols": ligand_data.element_symbols.astype('U'),
                    "split": split,
                    "cluster": f'P{protein_cluster_id}L{ligand_cluster_id}',
                    "protein_cluster_id": int(protein_cluster_id),
                    "ligand_cluster_id": int(ligand_cluster_id),
                }
                buffers[split].append(row)
                if len(buffers[split]) >= args.entries_per_file:
                    flush_split(split)
                
            except Exception as e:
                log.error(f"Error processing {system_dir}: {e}")

    # Flush any remaining rows
    for s in ["train", "val", "test"]:
        flush_split(s)

    if not saved_files:
        log.error("No parquet files written. Exiting")
        return

    create_indices(args.output_dir)

    index_df = pd.DataFrame({"parquet_file": saved_files, "split": [os.path.basename(f).split("_")[0] for f in saved_files]})
    index_path = os.path.join(args.output_dir, "index.csv")
    index_df.to_csv(index_path, index=False)
    log.info(f"Created index file with {len(saved_files)} parquet files at {index_path}")
    
    log.info("Dataset creation complete")

if __name__ == "__main__":
    main()