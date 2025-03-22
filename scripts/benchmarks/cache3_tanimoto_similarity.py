from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_morgan_fingerprint(mol, radius=2, nBits=2048):
    """Calculate Morgan fingerprint for a molecule."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

def calculate_tanimoto_similarity(fp1, fp2):
    """Calculate Tanimoto similarity between two fingerprints."""
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def get_metrics(df, score_column, hit_column='hit'):
    """Calculate various metrics for the predictions.
    
    Args:
        df: DataFrame containing the predictions
        score_column: Name of column containing the scores
        hit_column: Name of column containing the hit labels (True/False)
    
    Returns:
        dict: Dictionary containing various metrics
    """
    metrics = {}
    
    # Basic statistics
    metrics['mean_score'] = df[score_column].mean()
    metrics['mean_score_hits'] = df[df[hit_column] == True][score_column].mean()
    metrics['mean_score_misses'] = df[df[hit_column] == False][score_column].mean()
    dropna_df = df.dropna(subset=[score_column])
    # AUC-ROC
    metrics['auc_roc'] = roc_auc_score(dropna_df[hit_column], dropna_df[score_column])
    
    # Top/bottom examples
    sorted_df = dropna_df.sort_values(score_column, ascending=False)
    metrics['top_examples'] = sorted_df.head(10)
    metrics['bottom_examples'] = sorted_df.tail(10)
    
    return metrics

def print_metrics(metrics, score_name="score"):
    """Print metrics in a formatted way."""
    print(f"\nMetrics for {score_name}:")
    print(f"Mean score: {metrics['mean_score']:.4f}")
    print(f"Mean score for hits: {metrics['mean_score_hits']:.4f}")
    print(f"Mean score for misses: {metrics['mean_score_misses']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    
    print(f"\nTop 10 highest {score_name}:")
    for _, row in metrics['top_examples'].iterrows():
        print(f"{row['smiles']}: {row[score_name]:.4f}, hit: {row['hit']}")
    
    print(f"\nBottom 10 lowest {score_name}:")
    for _, row in metrics['bottom_examples'].iterrows():
        print(f"{row['smiles']}: {row[score_name]:.4f}, hit: {row['hit']}")

if __name__ == "__main__":
    # Reference molecule
    lrh0003_smiles = "O=C1CC=CN1Nc1ncnc2[nH]c3cc(F)ccc3c12"
    lrh0003_mol = Chem.MolFromSmiles(lrh0003_smiles)
    lrh0003_fp = calculate_morgan_fingerprint(lrh0003_mol)
    
    # Load CSV file
    csv_path = "/mnt/disk2/VoxelDiffOuter/VoxelDiff2/data/cache_round1_smiles_all_out_hits_and_others.csv"
    df = pd.read_csv(csv_path)
    
    # Calculate Tanimoto similarities
    similarities = []
    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = calculate_morgan_fingerprint(mol)
            sim = calculate_tanimoto_similarity(fp, lrh0003_fp)
            similarities.append(sim)
        else:
            similarities.append(None)
    
    df['tanimoto_similarity'] = similarities
    
    # Calculate and print metrics
    metrics = get_metrics(df, 'tanimoto_similarity')
    print_metrics(metrics, score_name="tanimoto_similarity")
    
    # Save results
    output_path = "tanimoto_similarity_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
