import torch
import docktgrid.config as docktgrid_config
from src.data.common.voxelization.config import VoxelizationConfig

def initialize_docktgrid(config: VoxelizationConfig = None):
    """
    Initialize docktgrid with the correct configuration.
    
    This function can be called at the start of your application to ensure
    docktgrid uses the correct data type without requiring users to manually
    edit the installed package.
    
    Args:
        config: VoxelizationConfig object. If None, uses default float16.
    """
    if config is None:
        # Default to float16 if no config provided
        docktgrid_config.DTYPE = torch.float16
    else:
        docktgrid_config.DTYPE = config.dtype
    
    return docktgrid_config.DTYPE
