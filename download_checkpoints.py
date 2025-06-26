import os
from huggingface_hub import hf_hub_download

REPO_ID = "judewells/plixer_v1"
# The local directory to save the checkpoints.
# The script will create subdirectories inside this folder to match the repo structure.
LOCAL_DIR = "checkpoints"

FILES_TO_DOWNLOAD = [
    "combined_protein_to_smiles/config.yaml",
    "combined_protein_to_smiles/epoch_000.ckpt",
    "poc_vox_to_mol_vox/config.yaml",
    "poc_vox_to_mol_vox/epoch_173.ckpt",
]

def download_checkpoints():
    """Downloads all necessary checkpoint files from the Hugging Face Hub."""
    print(f"Downloading checkpoints from repo: {REPO_ID}")
    
    for file_path in FILES_TO_DOWNLOAD:
        print(f"Downloading {file_path}...")
        try:
            # hf_hub_download will save the file to `local_dir/file_path`
            hf_hub_download(
                repo_id=REPO_ID,
                filename=file_path,
                repo_type="model",
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            print(f"Error downloading {file_path}: {e}")
            
    print("\nAll checkpoints downloaded successfully!")
    print(f"Files are located in the '{LOCAL_DIR}' directory, preserving the original structure.")

if __name__ == "__main__":
    download_checkpoints() 