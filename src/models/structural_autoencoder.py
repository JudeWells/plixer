import torch
import torch.nn as nn
from lightning import LightningModule
from src.models.pytorch3dunet_lib.unet3d.buildingblocks import ResNetBlockSE, create_encoders, create_decoders
from src.models.pytorch3dunet_lib.unet3d.losses import get_loss_criterion

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

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
        self.encoders = create_encoders(
            in_channels=in_channels,
            f_maps=f_maps,
            basic_module=ResNetBlockSE,
            conv_kernel_size=3,
            conv_padding=conv_padding,
            layer_order=layer_order,
            num_groups=num_groups,
            pool_kernel_size=2,
            dropout_prob=dropout_prob,
            is3d=True,
            conv_upscale=2
        )
        
        # Add final conv to reduce to latent dimension
        self.to_latent = nn.Conv3d(f_maps[-1], latent_dim, kernel_size=1)
        
        # Create decoder path (in reverse order of encoder f_maps)
        decoder_f_maps = list(reversed(f_maps))
        
        # Initial conv to expand from latent
        self.from_latent = nn.Conv3d(latent_dim, decoder_f_maps[0], kernel_size=1)
        
        # Create decoder path
        self.decoders = create_decoders(
            f_maps=decoder_f_maps,
            basic_module=ResNetBlockSE,
            conv_kernel_size=3,
            conv_padding=conv_padding,
            layer_order=layer_order,
            num_groups=num_groups,
            upsample='deconv',
            dropout_prob=dropout_prob,
            is3d=True
        )
        
        # Final conv to map back to input channels
        self.final_conv = nn.Conv3d(decoder_f_maps[-1], in_channels, kernel_size=1)
        
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
        x = self.from_latent(z)
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