import os
import json
import pandas as pd
from tqdm import tqdm

def filter_parquets(input_dir, output_dir, indices_dir, atom_threshold=60):
    os.makedirs(output_dir, exist_ok=True)
    new_indices_dir = os.path.join(output_dir, 'indices')
    os.makedirs(new_indices_dir, exist_ok=True)

    # Load existing indices
    with open(os.path.join(indices_dir, 'cluster_index.json'), 'r') as f:
        cluster_index = json.load(f)

    with open(os.path.join(indices_dir, 'file_mapping.json'), 'r') as f:
        file_mapping = json.load(f)

    # Create a mapping from system_id to cluster_id once
    system_id_to_cluster = {}
    for cluster_id, samples in cluster_index.items():
        for sample in samples:
            system_id_to_cluster[sample['system_id']] = cluster_id

    # Initialize new indices
    new_cluster_index = {}
    new_file_mapping = {}
    new_global_index = []
    new_index = {}
    summary = {'total_clusters': 0, 'total_samples': 0}

    file_counter = 0
    global_row_counter = 0
    dropped_rows = 0
    total_rows = 0

    for file_idx_str, rel_path in tqdm(file_mapping.items(), desc="Processing parquet files"):
        input_path = os.path.join(input_dir, rel_path)
        df = pd.read_parquet(input_path)
        total_rows += len(df)

        # Filter rows based on ligand atom count
        df['ligand_atom_count'] = df['ligand_element_symbols'].apply(len)
        filtered_df = df[df['ligand_atom_count'] <= atom_threshold].drop(columns=['ligand_atom_count']).reset_index(drop=True)
        dropped_rows += len(df) - len(filtered_df)

        if filtered_df.empty:
            continue  # Skip empty files after filtering

        # Save filtered parquet
        new_rel_path = f"filtered_{file_counter}.parquet"
        output_path = os.path.join(output_dir, new_rel_path)
        filtered_df.to_parquet(output_path, index=False)

        # Update file mapping
        new_file_mapping[file_counter] = new_rel_path

        # Update indices using the precomputed system_id to cluster_id mapping
        for row_idx, row in filtered_df.iterrows():
            system_id = row['system_id']
            cluster_id = system_id_to_cluster.get(system_id)

            if cluster_id is None:
                continue  # Skip if system_id not found in original indices

            if cluster_id not in new_cluster_index:
                new_cluster_index[cluster_id] = []
                summary['total_clusters'] += 1

            sample_info = {'file_idx': file_counter, 'system_id': system_id, 'row_idx': row_idx}
            new_cluster_index[cluster_id].append(sample_info)

            new_global_index.append({'file_idx': file_counter, 'row_idx': row_idx})
            new_index[str(global_row_counter)] = sample_info
            global_row_counter += 1
            summary['total_samples'] += 1

        file_counter += 1

    # Save new indices
    with open(os.path.join(new_indices_dir, 'cluster_index.json'), 'w') as f:
        json.dump(new_cluster_index, f, indent=2)

    with open(os.path.join(new_indices_dir, 'file_mapping.json'), 'w') as f:
        json.dump(new_file_mapping, f, indent=2)

    with open(os.path.join(new_indices_dir, 'global_index.json'), 'w') as f:
        json.dump(new_global_index, f, indent=2)

    with open(os.path.join(new_indices_dir, 'index.json'), 'w') as f:
        json.dump(new_index, f, indent=2)

    with open(os.path.join(new_indices_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Filtered data and indices saved to {output_dir}")
    print(f"Total rows: {total_rows}, Dropped rows: {dropped_rows}")

if __name__ == "__main__":
    input_directory = "/mnt/disk2/VoxelDiffOuter/plinder/train_arrays"
    output_directory = "/mnt/disk2/VoxelDiffOuter/plinder/train_arrays_filtered"
    indices_directory = "/mnt/disk2/VoxelDiffOuter/plinder/train_arrays/indices"

    filter_parquets(input_directory, output_directory, indices_directory, atom_threshold=60)
