import torch
import torch.nn as nn
from lightning import LightningModule
from src.models.pytorch3dunet_lib.unet3d.buildingblocks import ResNetBlockSE, Encoder, Decoder
from src.models.pytorch3dunet_lib.unet3d.losses import get_loss_criterion

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

def create_structural_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding,
                             layer_order, num_groups, dropout_prob, is3d=True):
    """
    Create encoder path for structural autoencoder. Each encoder reduces spatial dimensions by 2x.
    
    Args:
        in_channels (int): Number of input channels
        f_maps (list): List of feature maps for each encoder level
        basic_module: Basic conv module (ResNetBlock, ResNetBlockSE etc.)
        conv_kernel_size (int): Kernel size for convolutions
        conv_padding (int): Padding for convolutions
        layer_order (str): Order of operations in conv blocks
        num_groups (int): Number of groups for group norm
        dropout_prob (float): Dropout probability
        is3d (bool): Whether to use 3D convolutions
    """
    encoders = []
    
    # First encoder without pooling
    encoder = Encoder(in_channels, f_maps[0],
                     apply_pooling=False,
                     basic_module=basic_module,
                     conv_layer_order=layer_order,
                     conv_kernel_size=conv_kernel_size,
                     num_groups=num_groups,
                     padding=conv_padding,
                     dropout_prob=dropout_prob,
                     is3d=is3d)
    encoders.append(encoder)
    
    # Remaining encoders with pooling
    for i in range(1, len(f_maps)):
        encoder = Encoder(f_maps[i-1], f_maps[i],
                        basic_module=basic_module,
                        conv_layer_order=layer_order,
                        conv_kernel_size=conv_kernel_size,
                        num_groups=num_groups,
                        pool_kernel_size=2,  # Always reduce spatial dims by 2
                        padding=conv_padding,
                        dropout_prob=dropout_prob,
                        is3d=is3d)
        encoders.append(encoder)

    return nn.ModuleList(encoders)

def create_structural_decoders(f_maps, basic_module, conv_kernel_size, conv_padding,
                             layer_order, num_groups, dropout_prob, is3d=True):
    """
    Create decoder path for structural autoencoder. Each decoder increases spatial dimensions by 2x.
    No skip connections are used to ensure all information flows through the bottleneck.
    
    Args:
        f_maps (list): List of feature maps for each decoder level (in reverse order)
        basic_module: Basic conv module (ResNetBlock, ResNetBlockSE etc.)
        conv_kernel_size (int): Kernel size for convolutions
        conv_padding (int): Padding for convolutions
        layer_order (str): Order of operations in conv blocks
        num_groups (int): Number of groups for group norm
        dropout_prob (float): Dropout probability
        is3d (bool): Whether to use 3D convolutions
    """
    # Reverse feature maps to match encoder path
    f_maps = list(reversed(f_maps))
    decoders = []
    
    # Create decoders - each one upsamples by 2x
    for i in range(len(f_maps) - 1):
        in_channels = f_maps[i]
        out_channels = f_maps[i + 1]
        
        decoder = Decoder(in_channels, out_channels,
                         basic_module=basic_module,
                         conv_layer_order=layer_order,
                         conv_kernel_size=conv_kernel_size,
                         num_groups=num_groups,
                         padding=conv_padding,
                         scale_factor=2,  # Always increase spatial dims by 2
                         upsample='deconv',  # Use transposed conv for upsampling
                         dropout_prob=dropout_prob,
                         is3d=is3d,
                         skip_connections=False)  # No skip connections
        decoders.append(decoder)

    return nn.ModuleList(decoders)

class StructuralAutoEncoder(LightningModule):
    """
    Autoencoder for learning structural embeddings from 3D voxel data.
    Uses encoder/decoder architecture without skip connections for dimensionality reduction.
    """
    def __init__(
        self,
        in_channels: int,
        f_maps: int = 64,
        latent_dim: int = 512,
        num_levels: int = 5,
        layer_order: str = 'gcr',
        num_groups: int = 8,
        conv_padding: int = 1,
        dropout_prob: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        loss=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        
        # Create encoder path
        self.encoders = create_structural_encoders(
            in_channels=in_channels,
            f_maps=f_maps,
            basic_module=ResNetBlockSE,
            conv_kernel_size=3,
            conv_padding=conv_padding,
            layer_order=layer_order,
            num_groups=num_groups,
            dropout_prob=dropout_prob,
            is3d=True
        )
        
        # Add final conv to reduce to latent dimension
        self.to_latent = nn.Conv3d(f_maps[-1], latent_dim, kernel_size=2, stride=2)
        
        # # Create decoder path (in reverse order of encoder f_maps)
        # decoder_f_maps = list(reversed(f_maps))
        
        # First decoder layer: transform from latent to first decoder size
        # Use ConvTranspose3d to upsample from 1x1x1 to 2x2x2
        self.from_latent = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, f_maps[-1], 
                             kernel_size=2, stride=2),
            nn.GroupNorm(num_groups, f_maps[-1]),
            nn.ReLU(inplace=True)
        )
        
        # Create decoder path
        self.decoders = create_structural_decoders(
            f_maps=f_maps,
            basic_module=ResNetBlockSE,
            conv_kernel_size=3,
            conv_padding=conv_padding,
            layer_order=layer_order,
            num_groups=num_groups,
            dropout_prob=dropout_prob,
            is3d=True
        )
        
        # Final conv to map back to input channels
        self.final_conv = nn.Conv3d(f_maps[0], in_channels, kernel_size=1)
        
        self.loss_fn = get_loss_criterion(loss) if loss else nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay

    def encode(self, x):
        """Encode input to latent representation"""
        for encoder in self.encoders:
            x = encoder(x)
        return self.to_latent(x)

    def decode(self, z):
        """Decode latent representation back to original space"""
        # First upsample from latent space to initial decoder size
        x = self.from_latent(z)
        
        # Then apply decoder blocks
        for decoder in self.decoders:
            x = decoder(None, x)  # Pass None as encoder features to avoid skip connections
            
        return self.final_conv(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def training_step(self, batch, batch_idx):
        x = batch["protein"]  # Use protein channels as input
        recon, z = self(x)
        loss = self.loss_fn(recon, x)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["protein"]
        recon, z = self(x)
        loss = self.loss_fn(recon, x)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer

    def get_embedding(self, x):
        """Get latent embedding for input"""
        return self.encode(x)
    

if __name__ == "__main__":
    # Example usage
    
    loss_config = {
        'name': 'BCEDiceLoss',
        'weight': None,
        'normalization': None,
        'alpha': 0.2, # BCE weight
        'beta': 1 # DICE weight
    }
    model = StructuralAutoEncoder(
        in_channels=14,  # Number of protein channels
        f_maps=64,       # Initial feature maps
        latent_dim=512,  # Size of latent embedding
        num_levels=5,    # Number of down/up-sampling levels
        loss=loss_config  # Use MSE loss for reconstruction
    )

    # Get embeddings
    x = torch.randn(1, 14, 32, 32, 32)  # Example input
    z = model.get_embedding(x)  # Get latent embedding
    recon = model.decode(z)    # Reconstruct input
    bp=1