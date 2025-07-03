```
██████╗      ██╗         ██╗    ██╗   ██╗     ███████╗    ██████╗     
██   ██╗     ██║         ██║    ╚██╗ ██╔╝     ██╔════╝    ██   ██╗    
██████╔╝     ██║         ██║     ╚████╔╝      █████╗      ██████╔╝    
██╔═══╝      ██║         ██║     ██╔═██╗      ██╔══╝      ██╔══██╗    
██║          ███████╗    ██║    ██╔╝  ██╗     ███████╗    ██║  ██║    
╚═╝          ╚══════╝    ╚═╝    ╚═╝   ╚═╝     ╚══════╝    ╚═╝  ╚═╝
                                                         
```
1.  **Clone the repository:**
    ```bash
    git clone git@github.com:JudeWells/plixer.git
    cd plixer
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Download Model Checkpoints:**
    The necessary model checkpoints are hosted on Hugging Face Hub. Run the following script to download them into the `checkpoints/` directory.
    ```bash
    python download_checkpoints.py
    ```

4. **Check instalation by running**


# Poc2Smiles: End-to-End Protein-to-SMILES Model



This repository contains three models:

1. **Poc2Mol**: Generates voxelized ligands from protein voxel inputs
2. **Vox2Smiles**: Decodes voxelized ligands into SMILES strings
3. **CombinedProtein2Smiles** combines 1 & 2 for end-to-end pipeline

## Project Structure

```
├── configs/                  # Configuration files
│   ├── data/                 # Data configuration
│   ├── model/                # Model configuration
│   └── experiments/          # Launch training runs with this
├── src/                      # Source code
│   ├── data/                 # Data processing modules
│   │   ├── common/           # Common data utilities
│   │   │   ├── tokenizers/   # SMILES tokenizers
│   │   │   └── voxelization/ # Unified voxelization code
│   │   ├── poc2mol/          # Poc2Mol data modules
│   │   ├── vox2smiles/       # Vox2Smiles data modules
│   │   └── poc2smiles/       # Combined data modules
│   ├── models/               # Model definitions
│   └── utils/                # Utility functions
└── scripts/                  # Random scripts
```

## Installation

```bash
# Clone the repository
git clone https://github.com/judewells/plixer
cd plixer

# Install dependencies
pip install -r requirements.txt

```
Modify this file in your dock2grid install if you want to use float16 (optional):
`venv/lib/python3.11/site-packages/docktgrid/config.py`
```python
DTYPE = torch.bfloat16
```

## Usage

### Training poc2mol
```bash
python src/train.py experiment=example_train_poc2mol_plinder
```

### Training vox2smiles on zinc molecules

```bash
python src/train.py experiment=example_train_vox2smiles_zinc

```

### Fine-tuning Vox2Smiles on Poc2Mol Outputs AND zinc

```bash
python src/train.py experiment=train_vox2smiles_combined
```

### Generating SMILES Strings from Ligand Voxels

```bash
# Generate SMILES strings from protein voxels
python src/main.py --config-name=generate
```

## Model Architecture

### Poc2Mol

The Poc2Mol model is a 3D U-Net that takes protein voxels as input and generates ligand voxels as output. It uses a residual U-Net architecture.

### Vox2Smiles

The Vox2Smiles model is a Vision Transformer (ViT) that takes ligand voxels as input and generates SMILES strings as output. It uses a transformer encoder to process the voxel patches and a GPT-style decoder to generate the SMILES tokens.

### Combined Model

The combined CombinedProtein2Smiles model connects the Poc2Mol and Vox2Smiles models in an end-to-end fashion. It takes protein voxels as input, generates ligand voxels using Poc2Mol, and then generates SMILES strings using Vox2Smiles.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The voxelization code is based on [DockTGrid](https://github.com/example/docktgrid). 