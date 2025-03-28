import os
from typing import Any, Dict, Optional

import torch
import numpy as np
from lightning import LightningModule
import torch.nn.functional as F
import math

from src.models.pytorch3dunet import ResidualUNetSE3D
from src.models.pytorch3dunet_lib.unet3d.buildingblocks import ResNetBlockSE, ResNetBlock
from transformers.optimization import get_scheduler

from src.evaluation.visual import show_3d_voxel_lig_only, visualise_batch

from src.models.pytorch3dunet_lib.unet3d.losses import get_loss_criterion

class DiffusionConfig:
    def __init__(
        self,
        enabled: bool = False,
        timesteps: int = 1000,
        beta_schedule: str = 'linear',
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.enabled = enabled
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end

class ResUnetConfig:
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        final_sigmoid: bool = False,
        f_maps: int = 64,
        layer_order: str = 'gcr',
        num_groups: int = 8,
        num_levels: int = 5,
        conv_padding: int = 1,
        conv_upscale: int = 2,
        upsample: str = 'default',
        dropout_prob: float = 0.1,
        basic_module: ResNetBlock = ResNetBlockSE,
        diffusion: DiffusionConfig = DiffusionConfig(),
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_sigmoid = final_sigmoid
        self.f_maps = f_maps
        self.layer_order = layer_order
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.conv_padding = conv_padding
        self.conv_upscale = conv_upscale
        self.upsample = upsample
        self.dropout_prob = dropout_prob
        self.basic_module = basic_module
        self.diffusion = diffusion

class Poc2Mol(LightningModule):
    def __init__(
        self,
        loss,
        config: ResUnetConfig,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        scheduler_name: str = None,
        num_training_steps: Optional[int] = 100000,
        num_warmup_steps: int = 0,
        num_decay_steps: int = 0,
        img_save_dir: str = None,
        scheduler_kwargs: Dict[str, Any] = None,
        matmul_precision: str = 'high',
        compile: bool = False,
        override_optimizer_on_load: bool = False,
        visualise_val: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
    
        # Create the model
        self.model = ResidualUNetSE3D(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            final_sigmoid=config.final_sigmoid,
            f_maps=config.f_maps,
            layer_order=config.layer_order,
            num_groups=config.num_groups,
            num_levels=config.num_levels,
            conv_padding=config.conv_padding,
            conv_upscale=config.conv_upscale,
            upsample=config.upsample,
            dropout_prob=config.dropout_prob,
            basic_module=config.basic_module,
        )
        
        # Convert model to the specified dtype
        self.model = self.model.to(dtype=dtype)
        
        # Other initializations...
        with_logits = not config.final_sigmoid
        self.loss = get_loss_criterion(loss, with_logits=with_logits)
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs
        self.num_decay_steps = num_decay_steps
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.img_save_dir = img_save_dir
        torch.set_float32_matmul_precision(matmul_precision)
        self.override_optimizer_on_load = override_optimizer_on_load
        self.visualise_val = visualise_val
        self.config = config
        
        # Add diffusion-specific initialization if enabled
        if config.diffusion.enabled:
            self.timesteps = config.diffusion.timesteps
            
            # Calculate beta schedule (create directly with correct dtype)
            if config.diffusion.beta_schedule == 'linear':
                beta = torch.linspace(
                    config.diffusion.beta_start,
                    config.diffusion.beta_end,
                    self.timesteps,
                    dtype=dtype  # Use the parameter directly
                )
            elif config.diffusion.beta_schedule == 'cosine':
                # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
                steps = self.timesteps + 1
                x = torch.linspace(0, self.timesteps, steps, dtype=dtype)
                alphas_cumprod = torch.cos(((x / self.timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
                alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
                betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
                beta = torch.clip(betas, 0.0001, 0.9999)
            else:
                raise NotImplementedError(f"Beta schedule {config.diffusion.beta_schedule} not implemented")
            
            # Calculate diffusion parameters and register as buffers
            alpha = 1 - beta
            alpha_bar = torch.cumprod(alpha, dim=0)
            
            # Register tensors as buffers so they'll be properly moved to the right device with the model
            self.register_buffer('beta', beta)
            self.register_buffer('alpha', alpha)
            self.register_buffer('alpha_bar', alpha_bar)
            self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
            self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1 - alpha_bar))
            
            # Add time embedding layer if using diffusion
            self.time_mlp = torch.nn.Sequential(
                torch.nn.Linear(16, 32),
                torch.nn.SiLU(),
                torch.nn.Linear(32, 32),
            )

    def add_noise(self, x, t):
        """Add noise to input x at timestep t."""
        noise = torch.randn_like(x)
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1, 1)
        return sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise, noise

    def forward(self, prot_vox, ligand=None, t=None):
        if not self.config.diffusion.enabled:
            return self.model(x=prot_vox, labels=ligand)
        
        batch_size = prot_vox.shape[0]
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=prot_vox.device)
        
        # Add noise to ligand
        noisy_ligand, noise = self.add_noise(ligand, t)
        
        # Create time embeddings
        t_emb = self.get_timestep_embedding(t, embedding_dim=16)
        
        # Concatenate protein and noisy ligand along channel dimension
        x = torch.cat([prot_vox, noisy_ligand], dim=1)
        
        # The UNet will process the input and predict the noise
        pred_noise = self.model(x=x, t_emb=None)
        
        # Ensure pred_noise has the same shape as noise
        if pred_noise.shape != noise.shape:
            raise ValueError(f"Shape mismatch: pred_noise {pred_noise.shape} vs noise {noise.shape}")
        
        return pred_noise, noise  # Return both predicted and target noise

    def training_step(self, batch, batch_idx):
        if "load_time" in batch:
            self.log("train/load_time", batch["load_time"].mean(), on_step=True, on_epoch=False, prog_bar=True)

        if not self.config.diffusion.enabled:
            # Original non-diffusion training logic
            outputs = self(batch["protein"], labels=batch["ligand"])
            loss = self.loss(outputs, batch["ligand"])
            if isinstance(loss, dict):
                running_loss = 0
                for k,v in loss.items():
                    self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=False)
                    self.log(f"train/batch_{k}", v, on_step=True, on_epoch=False, prog_bar=False)
                    running_loss += v
                loss = running_loss
            
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/batch_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
            self.log_channel_means(batch, outputs)
            
            if batch_idx == 0:
                # apply sigmoid to outputs for visualisation
                outputs_for_viz = torch.sigmoid(outputs.detach())
                visualise_batch(batch["ligand"], outputs_for_viz, batch["name"], save_dir=self.img_save_dir, batch=str(batch_idx))
        else:
            # Diffusion training logic
            pred_noise, target_noise = self(batch["protein"], batch["ligand"])
            
            # Use the diffusion loss
            loss = self.loss(pred_noise, target_noise)
            
            # Handle dict-type loss similar to non-diffusion case
            if isinstance(loss, dict):
                running_loss = 0
                for k,v in loss.items():
                    self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=False)
                    self.log(f"train/batch_{k}", v, on_step=True, on_epoch=False, prog_bar=False)
                    running_loss += v
                loss = running_loss
            
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/batch_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
            
            # Log noise prediction metrics
            with torch.no_grad():
                noise_error = (pred_noise - target_noise).abs().mean()
                self.log("train/noise_error", noise_error, on_step=False, on_epoch=True)
            
            if batch_idx == 0 and loss < 0.5 and self.trainer.current_epoch % 10 == 0:
                self.visualize_diffusion_step(batch, save_dir=self.img_save_dir)
        
        return loss

    def validation_step(self, batch, batch_idx):
        if not self.config.diffusion.enabled:
            # Original validation logic
            outputs = self(batch["protein"], labels=batch["ligand"])
            loss = self.loss(outputs, batch["ligand"])
            if isinstance(loss, dict):
                running_loss = 0
                for k,v in loss.items():
                    self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False)
                    running_loss += v
                loss = running_loss
            save_dir = f"{self.img_save_dir}/val"
            if self.visualise_val and batch_idx in [0, 50, 100]:
                outputs_for_viz = torch.sigmoid(outputs.detach())
                lig, pred, names = batch["ligand"][:1], outputs_for_viz[:1], batch["name"][:1]

                visualise_batch(
                    lig,
                    pred,
                    names,
                    save_dir=save_dir,
                    batch=str(batch_idx)
                )

            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        else:
            # Diffusion validation logic
            pred_noise, target_noise = self(batch["protein"], batch["ligand"])
            
            # Use the diffusion loss
            loss = self.loss(pred_noise, target_noise)
            
            # Handle dict-type loss similar to non-diffusion case
            if isinstance(loss, dict):
                running_loss = 0
                for k,v in loss.items():
                    self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False)
                    running_loss += v
                loss = running_loss
            
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            
            if self.visualise_val and batch_idx in [0]:
                # Sample some denoised predictions for visualization
                with torch.no_grad():
                    denoised = self.sample(batch["protein"][:1])
                    if torch.isnan(denoised).any():
                        print(f"Warning: NaN values detected in denoised output at batch {batch_idx}")
                    else:
                        visualise_batch(
                            batch["ligand"][:1],
                            denoised,
                            batch["name"][:1],
                            save_dir=f"{self.img_save_dir}/val",
                            batch=str(batch_idx)
                        )
            
            return loss

    def test_step(self, batch, batch_idx):
        outputs = self(batch["protein"], labels=batch["ligand"])
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-5,
        )
        if self.scheduler_name is not None:
            scheduler = get_scheduler(
                self.scheduler_name,
                optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps,
                scheduler_specific_kwargs=self.scheduler_kwargs,
            )

        return [optimizer], [scheduler]
    
    
    def on_load_checkpoint(self, checkpoint):
        """Handle checkpoint loading, optionally overriding optimizer and scheduler states.

        If override_optimizer_on_load is True, we'll remove the optimizer and
        lr_scheduler states from the checkpoint, forcing Lightning to create new ones
        based on the current config hyperparameters.
        """
        if self.override_optimizer_on_load:
            if "optimizer_states" in checkpoint:
                print(
                    "Overriding optimizer state from checkpoint with current config values"
                )
                del checkpoint["optimizer_states"]

            if "lr_schedulers" in checkpoint:
                print(
                    "Overriding lr scheduler state from checkpoint with current config values"
                )
                del checkpoint["lr_schedulers"]

            # Set a flag to tell Lightning not to expect optimizer states
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []
    
    def log_channel_means(self, batch, outputs):
        n_lig_channels = batch['ligand'].shape[1]
        self.log_dict({
            f"channel_mean/ligand_{channel}": batch['ligand'][:,channel,...].mean().detach().item() for channel in range(n_lig_channels)
            })
        self.log_dict({
            f"channel_mean/pred_ligand_{channel}": outputs[:,channel,...].mean().detach().item() for channel in range(n_lig_channels)
        })
        n_prot_channels = batch['protein'].shape[1]
        self.log_dict({
            f"channel_mean/protein_{channel}": batch['protein'][:,channel,...].mean().detach().item() for channel in range(n_prot_channels)
        })

    @torch.no_grad()
    def sample(self, prot_vox, steps=None, save_intermediates=False):
        """Sample from the diffusion model."""
        if not self.config.diffusion.enabled:
            return self.forward(prot_vox)
            
        steps = steps or self.timesteps
        batch_size = prot_vox.shape[0]
        device = prot_vox.device
        
        # Start from pure noise
        x = torch.randn((batch_size, self.config.out_channels) + prot_vox.shape[2:], device=device)
        intermediates = [x.cpu()] if save_intermediates else None
        
        # Gradually denoise
        for t in reversed(range(steps)):
            if torch.isnan(x).any():
                print(f"Warning: NaN values detected in denoised output at step {t}")
                break
            t_batch = (torch.ones(batch_size, device=device) * t).long()
            
            # Predict noise
            pred_noise = self(prot_vox, x, t_batch)[0]  # Only take prediction, not target
            if torch.isnan(pred_noise).any():
                print(f"Warning: NaN values detected in predicted noise at step {t}")
            # Clip predicted noise to prevent extreme values
            pred_noise = torch.clamp(pred_noise, -100, 100)
            
            # Update sample using more stable formula
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]
            
            # Add small epsilon to avoid division by zero
            eps = 1e-8
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            
            # More numerically stable implementation
            normalization_factor = 1 / torch.sqrt(alpha_t + eps)
            noise_weight = beta_t / torch.sqrt(1 - alpha_bar_t + eps)

            x = normalization_factor * (x - noise_weight * pred_noise)
            
            if t > 0:  # Only add noise for t > 0
                x = x + torch.sqrt(beta_t) * noise
            
            # Clip x to prevent extreme values that could lead to NaNs in next iteration
            x = torch.clamp(x, -100, 100)
            
            # Check for NaN and replace with zeros if found
            if torch.isnan(x).any():
                print(f"Warning: NaN values detected in denoised output at step {t}")
                nan_mask = torch.isnan(x)
                x[nan_mask] = 0.0
            
            if save_intermediates:
                intermediates.append(torch.sigmoid(x.detach().cpu()))
        
        # Final sigmoid to get probabilities if needed

        x = torch.sigmoid(x)
        
        return (x, intermediates) if save_intermediates else x

    def get_timestep_embedding(self, timesteps, embedding_dim=128):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: (torch.Tensor) tensor of timesteps [B]
        :param embedding_dim: (int) dimension of the embeddings to create
        :return: (torch.Tensor) [B, embedding_dim]
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad if needed
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def visualize_diffusion_step(self, batch, save_dir=None):
        """Visualize the diffusion steps - shows original, noisy, and predicted samples."""
        if not self.config.diffusion.enabled or save_dir is None:
            return
            
        # Create directory if it doesn't exist
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            # Get a few samples for visualization
            prot = batch["protein"][:1]
            lig = batch["ligand"][:1]
            names = batch["name"][:1]
            
            # Sample intermediate steps
            _, denoising_steps = self.sample(prot, steps=self.timesteps//10, save_intermediates=True)
            self.log("denoising_steps_sampled", len(denoising_steps))
            if len(denoising_steps) >= (self.timesteps//10)-1:
                # Visualize original ligand
                visualise_batch(
                    lig,
                    lig,  # Show original as both input and output
                    names,
                    save_dir=save_dir,
                    batch="original"
                )
                
                # Visualize first noisy sample
                visualise_batch(
                    lig,
                    denoising_steps[0],
                    names,
                    save_dir=save_dir,
                    batch="noisy_t0"
                )
                
                # Visualize middle step
                mid_idx = len(denoising_steps) // 2
                visualise_batch(
                    lig,
                    denoising_steps[mid_idx],
                    names,
                    save_dir=save_dir,
                    batch=f"noisy_t{mid_idx}"
                )
                
                # Visualize final denoised result
                visualise_batch(
                    lig,
                    denoising_steps[-2],
                    names,
                    save_dir=save_dir,
                    batch="denoised"
                )
    
        @torch.no_grad()
        def sample2(self, prot_vox, steps=None, save_intermediates=False):
            """Sample from the diffusion model."""
            if not self.config.diffusion.enabled:
                return self.forward(prot_vox)
                
            steps = steps or self.timesteps
            batch_size = prot_vox.shape[0]
            device = prot_vox.device
            
            # Start from pure noise
            x = torch.randn((batch_size, self.config.out_channels) + prot_vox.shape[2:], device=device)
            intermediates = [x.cpu()] if save_intermediates else None
            
            # Gradually denoise
            for t in reversed(range(steps)):
                if torch.isnan(x).any():
                    print(f"Warning: NaN values detected in denoised output at step {t}")
                    break
                
                # Predict noise
                pred_noise = self(prot_vox, x, None)[0]  # Only take prediction, not target
                if torch.isnan(pred_noise).any():
                    print(f"Warning: NaN values detected in predicted noise at step {t}")
                # Clip predicted noise to prevent extreme values
                pred_noise = torch.clamp(pred_noise, -100, 100)
                    
                # More numerically stable implementation
                x = x - pred_noise
                
                
                # Clip x to prevent extreme values that could lead to NaNs in next iteration
                x = torch.clamp(x, -100, 100)
                
                # Check for NaN and replace with zeros if found
                if torch.isnan(x).any():
                    print(f"Warning: NaN values detected in denoised output at step {t}")
                    nan_mask = torch.isnan(x)
                    x[nan_mask] = 0.0
                
                if save_intermediates:
                    intermediates.append(torch.sigmoid(x.detach().cpu()))
            
            # Final sigmoid to get probabilities if needed

            x = torch.sigmoid(x)
            
            return (x, intermediates) if save_intermediates else x