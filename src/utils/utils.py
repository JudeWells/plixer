import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple
import os
import yaml
import numpy as np
import torch
from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(
    metric_dict: Dict[str, Any], metric_name: Optional[str]
) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value

def get_config_from_cpt_path(cpt_path: str) -> DictConfig:
    cpt_dir = os.path.dirname(cpt_path)
    # Try a few common locations for the config file
    potential_config_paths = [
        os.path.join(cpt_dir, "../.hydra/config.yaml"),
        os.path.join(cpt_dir, "config.yaml"),
    ]
    
    for config_path in potential_config_paths:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return DictConfig(config)
    
    raise FileNotFoundError(f"Could not find a config.yaml for checkpoint {cpt_path}")


def build_combined_model_from_config(
        config: DictConfig, 
        vox2smiles_ckpt_path: str,
        poc2mol_ckpt_path: str,
        dtype: torch.dtype,
        device: torch.device
    ):
    from src.models.poc2mol import Poc2Mol
    from src.models.vox2smiles import VoxToSmilesModel
    from src.models.poc2smiles import CombinedProteinToSmilesModel

    poc2mol_model = Poc2Mol.load_from_checkpoint(poc2mol_ckpt_path)

    vox2smiles_model = VoxToSmilesModel.load_from_checkpoint(vox2smiles_ckpt_path)

    combined_model = CombinedProteinToSmilesModel(
        poc2mol_model=poc2mol_model,
        vox2smiles_model=vox2smiles_model,
        config=config
    )

    combined_model = combined_model.to(dtype)
    combined_model.to(device)
    return combined_model


# def build_combined_model_from_config(
#         config: DictConfig, 
#         ckpt_path: str,
#         dtype: torch.dtype,
#         device: torch.device
#     ):
#     from src.models.poc2mol import Poc2Mol
#     from src.models.vox2smiles import VoxToSmilesModel
#     from src.models.poc2smiles import CombinedProteinToSmilesModel

#     # This function assumes the checkpoint is for the CombinedProteinToSmilesModel
#     # and that the component models (Poc2Mol, VoxToSmiles) are defined within it.
#     # Lightning will load the weights for the sub-models automatically.
    
#     # We need to initialize the sub-models to pass them to the combined model's constructor.
#     # The weights will be overwritten by the checkpoint's state_dict.
#     poc2mol_model = Poc2Mol(config.model.poc2mol_model.config)
#     vox2smiles_model = VoxToSmilesModel(config.model.vox2smiles_model.config)

#     model = CombinedProteinToSmilesModel.load_from_checkpoint(
#         checkpoint_path=ckpt_path,
#         poc2mol_model=poc2mol_model,
#         vox2smiles_model=vox2smiles_model,
#         config=config.model,
#         override_optimizer_on_load=True,
#     )

#     model = model.to(dtype)
#     model.to(device)
#     model.eval()
#     return model

def load_model(vox2smiles_ckpt_path, poc2mol_ckpt_path, device, dtype=torch.float32):
    """High-level wrapper to load a model for inference."""
    config = get_config_from_cpt_path(vox2smiles_ckpt_path)
    model = build_combined_model_from_config(
        config=config,
        vox2smiles_ckpt_path=vox2smiles_ckpt_path,
        poc2mol_ckpt_path=poc2mol_ckpt_path,
        dtype=dtype,
        device=device,
    )
    return model, config


# ============== Inference Utilities ==============
from rdkit import Chem
from src.data.common.voxelization.config import Poc2MolDataConfig
from src.data.common.voxelization.voxelizer import UnifiedVoxelGrid
from docktgrid.molparser import MolecularParser
from docktgrid.molecule import MolecularData, ptable, MolecularComplex

def get_center_from_ligand(ligand_path):
    if ligand_path.endswith('.sdf'):
        suppl = Chem.SDMolSupplier(ligand_path, removeHs=False)
        mol = next(suppl)
    elif ligand_path.endswith('.mol2'):
        mol = Chem.MolFromMol2File(ligand_path, removeHs=False)
    else:
        raise ValueError("Ligand file must be .sdf or .mol2")

    if mol is None:
        if ligand_path.endswith('mol2'):
            return mol2_center_parser(ligand_path)
        else:
            raise ValueError("Could not read ligand file.")
    
    conformer = mol.GetConformer()
    positions = conformer.GetPositions()
    center = positions.mean(axis=0)
    return center

def voxelize_protein(protein_pdb_path, center_coords, config: Poc2MolDataConfig):
    parser = MolecularParser()
    protein_data = parser.parse_file(protein_pdb_path, ext='.pdb')
    
    # Ensure protein data has the correct dtype
    protein_data.coords = protein_data.coords.to(config.dtype)

    # Create a dummy ligand with one carbon atom at the center.
    ligand_coords = torch.from_numpy(center_coords).unsqueeze(1).to(config.dtype)
    dummy_ligand_data = MolecularData(
        molecule_object=None,
        coords=ligand_coords,
        element_symbols=np.array(['C'])
    )
    
    # The molparser is not used when MolecularData objects are passed.
    complex_obj = MolecularComplex(protein_file=protein_data, ligand_file=dummy_ligand_data, molparser=None)
    
    # Manually set the correct dtype for attributes that might have been created with a default dtype.
    complex_obj.ligand_center = complex_obj.ligand_center.to(config.dtype)
    complex_obj.coords = complex_obj.coords.to(config.dtype)
    complex_obj.vdw_radii = complex_obj.vdw_radii.to(config.dtype)

    voxelizer = UnifiedVoxelGrid(config)
    voxel = voxelizer.voxelize(complex_obj)
    
    # Separate protein and ligand channels
    protein_channels_count = len(config.protein_channels)
    protein_voxel = voxel[:protein_channels_count]
    
    return protein_voxel.unsqueeze(0) # Add batch dimension
    
def mol2_center_parser(mol2_path):
    """
    Parses a MOL2 file to find the center of the first molecule.

    This function manually parses a MOL2 file to avoid dependency on RDKit for this specific task.
    It reads the atom coordinates from the first `@<TRIPOS>ATOM` section and computes their centroid.

    Args:
        mol2_path (str): The path to the MOL2 file.

    Returns:
        np.ndarray: A numpy array of shape (3,) containing the (x, y, z) coordinates of the centroid.
    
    Raises:
        ValueError: If no atoms are found in the MOL2 file.
    """
    coords = []
    in_atom_section = False
    with open(mol2_path, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line == '@<TRIPOS>ATOM':
                in_atom_section = True
                continue
            
            if in_atom_section:
                if stripped_line.startswith('@<TRIPOS>'):
                    break  # End of ATOM section for the first molecule
                
                parts = stripped_line.split()
                if len(parts) >= 5:
                    try:
                        x = float(parts[2])
                        y = float(parts[3])
                        z = float(parts[4])
                        coords.append([x, y, z])
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue

    if not coords:
        raise ValueError(f"Could not find any atoms in the MOL2 file: {mol2_path}")

    return np.array(coords).mean(axis=0)