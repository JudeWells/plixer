# Poc2Smiles: End-to-End Protein-to-SMILES Model

This repository contains an end-to-end model for generating SMILES strings directly from protein voxel inputs. It integrates two models:

1. **Poc2Mol**: Generates voxelized ligands from protein voxel inputs
2. **Vox2Smiles**: Decodes voxelized ligands into SMILES strings

## Project Structure

```
├── configs/                  # Configuration files
│   ├── data/                 # Data configuration
│   ├── model/                # Model configuration
│   └── training/             # Training configuration
├── src/                      # Source code
│   ├── data/                 # Data processing modules
│   │   ├── common/           # Common data utilities
│   │   │   ├── tokenizers/   # SMILES tokenizers
│   │   │   └── voxelization/ # Unified voxelization code
│   │   ├── poc2mol/          # Poc2Mol data modules
│   │   ├── vox2smiles/       # Vox2Smiles data modules
│   │   └── poc2smiles/       # Combined data modules
│   ├── models/               # Model definitions
│   ├── training/             # Training scripts
│   └── utils/                # Utility functions
└── scripts/                  # Utility scripts
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/poc2smiles.git
cd poc2smiles

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the End-to-End Model

```bash
# Train the end-to-end model
python src/main.py mode=train_end2end training=train_end2end data=poc2smiles_data
```

### Fine-tuning Vox2Smiles on Poc2Mol Outputs

```bash
# Fine-tune Vox2Smiles on Poc2Mol outputs
python src/main.py mode=finetune_vox2smiles training=finetune_vox2smiles data=unified_data
```

### Generating SMILES Strings from Protein Voxels

```bash
# Generate SMILES strings from protein voxels
python src/main.py --config-name=generate
```

## Model Architecture

### Poc2Mol

The Poc2Mol model is a 3D U-Net that takes protein voxels as input and generates ligand voxels as output. It uses a residual U-Net architecture with skip connections to preserve spatial information.

### Vox2Smiles

The Vox2Smiles model is a Vision Transformer (ViT) that takes ligand voxels as input and generates SMILES strings as output. It uses a transformer encoder to process the voxel patches and a transformer decoder to generate the SMILES tokens.

### Combined Model

The combined Poc2Smiles model connects the Poc2Mol and Vox2Smiles models in an end-to-end fashion. It takes protein voxels as input, generates ligand voxels using Poc2Mol, and then generates SMILES strings using Vox2Smiles.

## Data Pipeline

1. **Protein-Ligand Complexes**: The input data consists of protein-ligand complexes in PDB and MOL2 formats.
2. **Voxelization**: The proteins and ligands are voxelized using a unified voxelization module.
3. **SMILES Tokenization**: The SMILES strings are tokenized using a pre-trained tokenizer.
4. **Training**: The model is trained to generate SMILES strings from protein voxels.

## Metrics

The model is evaluated using the following metrics:

- **Validity**: Percentage of generated SMILES strings that are valid molecules.
- **Uniqueness**: Percentage of unique molecules among the valid ones.
- **Novelty**: Percentage of generated molecules that are not in the training set.
- **Similarity**: Average Tanimoto similarity between generated molecules and reference molecules.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The Poc2Mol model is based on [VoxelDiff](https://github.com/example/voxeldiff).
- The Vox2Smiles model is based on [VoxelToSMILES](https://github.com/example/voxeltosmiles).
- The voxelization code is based on [DockTGrid](https://github.com/example/docktgrid). 