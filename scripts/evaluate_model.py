#!/usr/bin/env python
"""
Script to evaluate the end-to-end model on a test set.
"""

import os
import argparse
import torch
import glob
import json
import pandas as pd
from rdkit import Chem
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.models.poc2smiles import CombinedProteinToSmilesModel
from src.data.poc2smiles.data_module import ProteinLigandSmilesDataModule
from src.data.common.voxelization.config import Poc2MolDataConfig, Vox2SmilesDataConfig
from src.utils.metrics import calculate_metrics, calculate_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the end-to-end model on a test set")
    
    # Paths
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    
    # Evaluation parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate per protein")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = CombinedProteinToSmilesModel.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.cuda()
    
    # Create data configs
    poc2mol_config = Poc2MolDataConfig(
        vox_size=0.75,
        box_dims=[24.0, 24.0, 24.0],
        random_rotation=False,
        random_translation=0.0,
        max_atom_dist=32.0,
        batch_size=args.batch_size,
    )
    
    vox2smiles_config = Vox2SmilesDataConfig(
        vox_size=0.75,
        box_dims=[24.0, 24.0, 24.0],
        random_rotation=False,
        random_translation=0.0,
        max_atom_dist=32.0,
        batch_size=args.batch_size,
        max_smiles_len=200,
    )
    
    # Create data module
    data_module = ProteinLigandSmilesDataModule(
        poc2mol_config=poc2mol_config,
        vox2smiles_config=vox2smiles_config,
        pdb_dir=args.test_dir,
        val_pdb_dir=args.test_dir,
    )
    
    # Setup data module
    data_module.setup("test")
    
    # Get test dataloader
    test_loader = data_module.test_dataloader()
    
    # Evaluate model
    all_results = []
    all_generated_smiles = []
    all_true_smiles = []
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        protein_voxels = batch["protein"].cuda()
        ligand_voxels = batch["ligand"].cuda()
        true_smiles = batch.get("smiles_str", [])
        names = batch["name"]
        
        batch_results = []
        
        with torch.no_grad():
            # Generate multiple samples per protein
            for i in range(len(protein_voxels)):
                protein_voxel = protein_voxels[i:i+1]
                true_ligand_voxel = ligand_voxels[i:i+1]
                name = names[i]
                
                # Get true SMILES
                true_smile = true_smiles[i] if i < len(true_smiles) else ""
                
                # Generate samples
                sample_smiles = []
                
                for _ in range(args.num_samples):
                    output = model(protein_voxel)
                    generated_ligand_voxel = output["ligand_voxels"]
                    generated_smile = model.vox2smiles_model.generate_smiles(
                        generated_ligand_voxel, 
                        temperature=args.temperature
                    )[0]
                    sample_smiles.append(generated_smile)
                
                # Calculate metrics for this protein
                validity = calculate_validity(sample_smiles)
                uniqueness = calculate_uniqueness(sample_smiles)
                
                # Calculate similarity to true SMILES if available
                similarity = 0.0
                if true_smile:
                    for smile in sample_smiles:
                        mol = Chem.MolFromSmiles(smile)
                        if mol is not None:
                            true_mol = Chem.MolFromSmiles(true_smile)
                            if true_mol is not None:
                                similarity += calculate_similarity(smile, true_smile)
                    
                    if validity > 0:
                        similarity /= (validity * len(sample_smiles))
                
                # Save results
                result = {
                    "name": name,
                    "true_smiles": true_smile,
                    "generated_smiles": sample_smiles,
                    "validity": validity,
                    "uniqueness": uniqueness,
                    "similarity": similarity,
                }
                
                batch_results.append(result)
                all_generated_smiles.extend(sample_smiles)
                if true_smile:
                    all_true_smiles.append(true_smile)
        
        all_results.extend(batch_results)
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(all_generated_smiles, all_true_smiles)
    
    # Save results
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "overall_metrics": overall_metrics,
            "per_protein_results": all_results,
        }, f, indent=2)
    
    # Create summary CSV
    summary_data = []
    for result in all_results:
        summary_data.append({
            "name": result["name"],
            "validity": result["validity"],
            "uniqueness": result["uniqueness"],
            "similarity": result["similarity"],
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(args.output_dir, "evaluation_summary.csv"), index=False)
    
    # Print overall metrics
    print("Overall Metrics:")
    for metric_name, metric_value in overall_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Create visualizations
    create_visualizations(all_results, overall_metrics, args.output_dir)


def create_visualizations(results, overall_metrics, output_dir):
    """
    Create visualizations of the evaluation results.
    
    Args:
        results: List of per-protein results
        overall_metrics: Dictionary of overall metrics
        output_dir: Path to output directory
    """
    # Create directory for visualizations
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Extract metrics
    validities = [result["validity"] for result in results]
    uniquenesses = [result["uniqueness"] for result in results]
    similarities = [result["similarity"] for result in results]
    
    # Create histogram of validity
    plt.figure(figsize=(10, 6))
    plt.hist(validities, bins=10, alpha=0.7)
    plt.axvline(overall_metrics["validity"], color="red", linestyle="dashed", linewidth=2)
    plt.xlabel("Validity")
    plt.ylabel("Count")
    plt.title(f"Distribution of Validity (Overall: {overall_metrics['validity']:.4f})")
    plt.savefig(os.path.join(vis_dir, "validity_histogram.png"), dpi=300)
    plt.close()
    
    # Create histogram of uniqueness
    plt.figure(figsize=(10, 6))
    plt.hist(uniquenesses, bins=10, alpha=0.7)
    plt.axvline(overall_metrics["uniqueness"], color="red", linestyle="dashed", linewidth=2)
    plt.xlabel("Uniqueness")
    plt.ylabel("Count")
    plt.title(f"Distribution of Uniqueness (Overall: {overall_metrics['uniqueness']:.4f})")
    plt.savefig(os.path.join(vis_dir, "uniqueness_histogram.png"), dpi=300)
    plt.close()
    
    # Create histogram of similarity
    if "avg_similarity" in overall_metrics:
        plt.figure(figsize=(10, 6))
        plt.hist(similarities, bins=10, alpha=0.7)
        plt.axvline(overall_metrics["avg_similarity"], color="red", linestyle="dashed", linewidth=2)
        plt.xlabel("Similarity to True Ligand")
        plt.ylabel("Count")
        plt.title(f"Distribution of Similarity (Overall: {overall_metrics['avg_similarity']:.4f})")
        plt.savefig(os.path.join(vis_dir, "similarity_histogram.png"), dpi=300)
        plt.close()
    
    # Create scatter plot of validity vs. uniqueness
    plt.figure(figsize=(10, 6))
    plt.scatter(validities, uniquenesses, alpha=0.7)
    plt.xlabel("Validity")
    plt.ylabel("Uniqueness")
    plt.title("Validity vs. Uniqueness")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(vis_dir, "validity_vs_uniqueness.png"), dpi=300)
    plt.close()
    
    # Create scatter plot of validity vs. similarity
    if "avg_similarity" in overall_metrics:
        plt.figure(figsize=(10, 6))
        plt.scatter(validities, similarities, alpha=0.7)
        plt.xlabel("Validity")
        plt.ylabel("Similarity to True Ligand")
        plt.title("Validity vs. Similarity")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(vis_dir, "validity_vs_similarity.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    main() 