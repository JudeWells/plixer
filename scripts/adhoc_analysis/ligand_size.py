import os
import pandas as pd
from tqdm import tqdm

def filter_large_ligands(input_dir, output_csv, atom_threshold=50):
    results = []

    # Iterate over all parquet files in the input directory
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
    for pq_file in tqdm(parquet_files, desc="Processing parquet files"):
        file_path = os.path.join(input_dir, pq_file)
        df = pd.read_parquet(file_path)

        # Process each row in the dataframe
        for _, row in df.iterrows():
            num_atoms = len(row['ligand_element_symbols'])
            if num_atoms > atom_threshold:
                results.append({
                    'system_id': row['system_id'],
                    'cluster_id': row['cluster_id'],
                    'num_atoms': num_atoms,
                    'smiles': row['smiles']
                })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved {len(results)} entries to {output_csv}")

if __name__ == "__main__":
    input_directory = "../plinder/train_arrays"
    output_csv_file = "large_ligands.csv"
    filter_large_ligands(input_directory, output_csv_file, atom_threshold=29)