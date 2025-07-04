import os
from typing import Any, Dict, Optional

import torch
import numpy as np
from lightning import LightningModule
import torch.nn.functional as F

from src.models.pytorch3dunet import ResidualUNetSE3D
from src.models.pytorch3dunet_lib.unet3d.buildingblocks import ResNetBlockSE, ResNetBlock
from transformers.optimization import get_scheduler
from torch.optim.lr_scheduler import StepLR

from src.evaluation.visual import show_3d_voxel_lig_only, visualise_batch

from src.models.pytorch3dunet_lib.unet3d.losses import get_loss_criterion, compute_per_channel_dice


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

class Poc2Mol(LightningModule):
    def __init__(
        self,
        config: ResUnetConfig,
        loss="BCEDiceLoss",
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        scheduler: Optional[Dict[str, Any]] = None,
        num_training_steps: Optional[int] = 100000,
        num_warmup_steps: int = 0,
        num_decay_steps: int = 0,
        img_save_dir: str = None,
        matmul_precision: str = 'high',
        override_optimizer_on_load: bool = False,
        visualise_train: bool = True,
        visualise_val: bool = True,
        n_samples_for_visualisation: int = 2,
        unmasking_strategy: str = "random",  # "confidence" or "random"
        # ---------------- Critic specific ------------------
        use_critic: bool = False,
        critic_loss_weight: float = 1.0,
        critic_incorrect_threshold: float = 0.7,
        # Determines the maximum number of voxels that can be intentionally
        # corrupted for critic training.  It is expressed as a fraction of the
        # *current* number of masked voxels (1.0 = up to n_masked).  The actual
        # number is sampled uniformly between 0 and this maximum on every
        # iteration.
        max_corrupt_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        assert config.final_sigmoid == False, "final_sigmoid must be False"
        self.model = ResidualUNetSE3D(
            # Protein channels + 9 ligand channels + 1 mask channel
            in_channels=config.in_channels + config.out_channels + 1,
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
        self.loss = get_loss_criterion(loss, with_logits=True)
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler or {}
        self.num_decay_steps = num_decay_steps
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.img_save_dir = img_save_dir
        torch.set_float32_matmul_precision(matmul_precision)
        self.override_optimizer_on_load = override_optimizer_on_load
        self.visualise_val = visualise_val
        self.visualise_train = visualise_train
        self.n_samples_for_visualisation = n_samples_for_visualisation
        # Strategy for selecting which masked voxels to reveal during sampling
        assert unmasking_strategy in {"confidence", "random"}, (
            "unmasking_strategy must be either 'confidence' or 'random'"
        )
        self.unmasking_strategy = unmasking_strategy
        self.residual_unet_config = config

        # ---------------- Diffusion hyper-parameters ----------------
        # total number of un-masking iterations during training / sampling
        self.num_diffusion_steps = 10
        # threshold used when turning continuous occupancies into binary one-hot
        self.discretize_threshold = 0.5
        # Probability below this is interpreted as EMPTY during sampling
        self.empty_threshold = 0.5

        # ---------------- Critic initialisation ----------------
        self.use_critic = use_critic
        self.critic_loss_weight = critic_loss_weight
        self.critic_incorrect_threshold = critic_incorrect_threshold
        self.max_corrupt_ratio = max_corrupt_ratio

        if self.use_critic:
            # Duplicate architecture but with single output channel (incorrect / correct)
            self.critic = ResidualUNetSE3D(
                in_channels=config.in_channels + config.out_channels + 1,
                out_channels=1,
                final_sigmoid=False,
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

            # Running estimates (per timestep) of how many voxels are re-masked
            # by the critic during *inference*.  We store count, mean and the
            # "M2" value required for Welford's online variance algorithm.  All
            # tensors have shape ``[T+1]`` so they can be directly indexed by
            # timestep ``t``.
            self.register_buffer(
                "critic_rm_count",
                torch.zeros(self.num_diffusion_steps + 1, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "critic_rm_mean",
                torch.zeros(self.num_diffusion_steps + 1),
                persistent=False,
            )
            self.register_buffer(
                "critic_rm_M2",
                torch.zeros(self.num_diffusion_steps + 1),
                persistent=False,
            )

    # ------------------------------------------------------------------
    # Helper functions for discrete masking diffusion
    # ------------------------------------------------------------------

    @staticmethod
    def _discretise_ligand(ligand: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Turn continuous 9-channel ligand occupancies into binary one-hot.

        A voxel is considered *occupied* by the winning atom type if its value
        is >= *threshold*; otherwise the voxel is set to all-zeros (empty).
        """
        # ligand: [B, 9, D, H, W]
        max_vals, max_idx = ligand.max(dim=1, keepdim=True)
        one_hot = torch.zeros_like(ligand)
        one_hot.scatter_(1, max_idx, 1.0)
        occupied_mask = (max_vals >= threshold).float()
        one_hot = one_hot * occupied_mask
        return one_hot

    def _generate_random_mask(self, ligand_onehot: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Sample voxels uniformly (occupied **and** empty) for masking.

        For each sample *b* we mask a fraction `t_b / T` of **all** voxels so
        that training matches inference, where the model initially sees only
        mask tokens.
        """
        B, _, D, H, W = ligand_onehot.shape
        device = ligand_onehot.device

        total_vox = D * H * W
        # create base random numbers once to vectorise the operation
        rand = torch.rand((B, D, H, W), device=device)
        # per-sample masking threshold = frac = t/T
        frac = timesteps.float().view(B, 1, 1, 1) / float(self.num_diffusion_steps)
        mask = rand < frac  # bool tensor

        # guarantee that at least one voxel is masked so loss isn't NaN
        for b in range(B):
            if not mask[b].any():
                # randomly set one voxel
                idx = torch.randint(0, total_vox, (1,), device=device)
                z = idx // (H * W)
                y = (idx % (H * W)) // W
                x = idx % W
                mask[b, z, y, x] = True
        return mask

    @staticmethod
    def _apply_mask(ligand_onehot: torch.Tensor, mask_pos: torch.Tensor) -> torch.Tensor:
        """Return [B, 10, D, H, W] tensor with ligand channels + mask channel."""
        ligand_masked = ligand_onehot.clone()
        ligand_masked = ligand_masked * (~mask_pos.unsqueeze(1)).float()
        mask_channel = mask_pos.unsqueeze(1).float()
        return torch.cat([ligand_masked, mask_channel], dim=1)

    @torch.no_grad()
    def sample_ligand(self, prot_vox: torch.Tensor, num_steps: int = None, topk_fraction: float = 0.2):
        """Iterative de-masking inference.

        Starts from a fully masked ligand and reveals voxels greedily in *num_steps*
        iterations using the model's current predictions.
        """
        num_steps = num_steps or self.num_diffusion_steps
        B, P, D, H, W = prot_vox.shape
        C = self.residual_unet_config.out_channels  # 9 atom types
        device = prot_vox.device

        ligand_part = torch.zeros((B, C, D, H, W), device=device)
        mask_channel = torch.ones((B, 1, D, H, W), device=device)
        ligand_with_mask = torch.cat([ligand_part, mask_channel], dim=1)

        for step_idx in range(num_steps):
            model_input = torch.cat([prot_vox, ligand_with_mask], dim=1)
            logits = self.model(model_input)
            probs = torch.sigmoid(logits)  # [B,9,D,H,W]

            # Atom-wise max prob and index
            conf_atom, atom_idx = probs.max(dim=1)  # [B,D,H,W]
            # Probability that voxel is empty = 1 - sum(atom_probs)
            empty_prob = (1.0 - probs.sum(dim=1)).clamp(min=0.0, max=1.0)  # [B,D,H,W]

            # Confidence used for ranking: best between atoms and empty
            conf_total = torch.maximum(conf_atom, empty_prob)  # [B,D,H,W]

            mask_positions = ligand_with_mask[:, -1] > 0.5  # [B,D,H,W]

            for b in range(B):
                vox_coords = mask_positions[b].nonzero(as_tuple=False)
                if vox_coords.numel() == 0:
                    continue

                n_to_unmask = max(1, int(topk_fraction * vox_coords.shape[0]))

                if self.unmasking_strategy == "confidence":
                    # Select voxels with highest confidence according to the model
                    conf_b = conf_total[b][mask_positions[b]]
                    topk_idx = torch.topk(conf_b, k=n_to_unmask, largest=True).indices
                    selected = vox_coords[topk_idx]
                else:  # "random"
                    perm = torch.randperm(vox_coords.shape[0], device=device)
                    selected = vox_coords[perm[:n_to_unmask]]

                # zero out any previous content
                ligand_with_mask[b, :-1, selected[:, 0], selected[:, 1], selected[:, 2]] = 0

                # Iterate over selected voxels and decide empty vs atom prediction
                for idx_i in range(selected.shape[0]):
                    z, y, x = selected[idx_i]
                    best_ch = atom_idx[b, z, y, x].item()
                    best_conf_atom = probs[b, best_ch, z, y, x].item()
                    best_conf_empty = empty_prob[b, z, y, x].item()

                    if best_conf_empty >= best_conf_atom and best_conf_empty >= self.empty_threshold:
                        # leave all-zero → empty prediction
                        pass
                    elif best_conf_atom >= self.empty_threshold:
                        ligand_with_mask[b, best_ch, z, y, x] = 1.0  # assign atom
                    # otherwise low confidence for both, keep empty
                    ligand_with_mask[b, -1, z, y, x] = 0  # clear mask regardless

            # ---------------- Critic re-masking (inference) ----------------
            if self.use_critic:
                critic_input = torch.cat([prot_vox, ligand_with_mask], dim=1)
                critic_logits = self.critic(critic_input)
                critic_probs = torch.sigmoid(critic_logits).squeeze(1)  # [B,D,H,W]

                # Identify voxels the critic believes are incorrect *and* are
                # currently unmasked (mask channel == 0)
                current_unmasked = ligand_with_mask[:, -1] < 0.5  # bool
                re_mask_positions = (critic_probs > self.critic_incorrect_threshold) & current_unmasked

                # Apply re-masking: clear ligand channels & set mask channel
                # to 1 at those positions.
                ligand_with_mask[:, :-1][re_mask_positions.unsqueeze(1).expand(-1, C, -1, -1, -1)] = 0
                ligand_with_mask[:, -1][re_mask_positions] = 1.0

            # ---------------- Logging masked proportion ----------------
            try:
                masked_frac = (ligand_with_mask[:, -1] > 0.5).float().mean().item()
                self.log(f"infer/masked_frac_t{step_idx+1}", masked_frac, on_step=True, on_epoch=True, prog_bar=False)
            except Exception:
                # In case self.log is unavailable (e.g. outside Lightning trainer context)
                pass

        # ---------------- Final unconditional unmasking ----------------
        remaining_masks = ligand_with_mask[:, -1] > 0.5  # bool
        self.log("infer/masked_frac_final", remaining_masks.float().mean().item(), on_step=True, on_epoch=True, prog_bar=False)
        if remaining_masks.any():
            # Second pass of the generator to fill *all* remaining masked voxels.
            model_input = torch.cat([prot_vox, ligand_with_mask], dim=1)
            logits = self.model(model_input)
            probs = torch.sigmoid(logits)

            # Prepare helper tensors
            conf_atom, atom_idx = probs.max(dim=1)
            empty_prob = (1.0 - probs.sum(dim=1)).clamp(min=0.0, max=1.0)

            coords = remaining_masks.nonzero(as_tuple=False)  # [N,4] (B,z,y,x)
            for coord in coords:
                b, z, y, x = coord.tolist()
                best_ch = atom_idx[b, z, y, x].item()
                best_conf_atom = probs[b, best_ch, z, y, x].item()
                best_conf_empty = empty_prob[b, z, y, x].item()

                # Decide assignment (same rule as earlier)
                if best_conf_empty >= best_conf_atom and best_conf_empty >= self.empty_threshold:
                    # leave empty (all zeros)
                    pass
                elif best_conf_atom >= self.empty_threshold:
                    ligand_with_mask[b, best_ch, z, y, x] = 1.0
                # Clear mask channel in all cases
                ligand_with_mask[b, -1, z, y, x] = 0.0

            # Log final masked fraction (should be zero)
            try:
                final_mask_frac = (ligand_with_mask[:, -1] > 0.5).float().mean().item()
                self.log("infer/masked_frac_final", final_mask_frac, on_step=True, on_epoch=False, prog_bar=False)
            except Exception:
                pass

            # Safety check – no masks should remain
            assert not (ligand_with_mask[:, -1] > 0.5).any(), "sample_ligand must return with zero masks remaining"

        return ligand_with_mask[:, :-1]  # drop mask channel

    def forward(self, prot_vox, labels=None):
        """Run the UNet and (optionally) compute loss.

        Parameters
        ----------
        prot_vox : torch.Tensor
            Protein voxel tensor of shape ``[B, C, X, Y, Z]``.
        labels : torch.Tensor | None
            Ground-truth *continuous* ligand voxels (9 channels).  If supplied
            the method runs a single diffusion training step; otherwise it
            performs iterative de-masking sampling.

        Returns
        -------
        Dict[str, torch.Tensor]
            The returned dictionary always contains at least the following keys

            ``predicted_ligand_logits``
                The raw UNet output (logits).

            ``predicted_ligand_voxels``
                Sigmoid-activated version of the logits which can be treated as
                probabilities/occupancies.

            ``loss`` *(optional)*
                Total loss (only present when *labels* is given).

            In training mode (i.e. when *labels* are provided) the dictionary
            additionally contains one entry per individual loss component (e.g.
            ``bce`` and ``dice``).
        """
        # ---------------- Inference path ----------------
        if labels is None:
            pred_vox = self.sample_ligand(prot_vox)
            return {
                "predicted_ligand_logits": torch.logit(torch.clamp(pred_vox, 1e-6, 1 - 1e-6)),
                "predicted_ligand_voxels": pred_vox,
            }

        # ---------------- Training / validation path ----------------
        # 1) discretise ligand
        ligand_onehot = self._discretise_ligand(labels, threshold=self.discretize_threshold)

        # 2) sample t ∈ [1,T]
        B = ligand_onehot.size(0)
        device = ligand_onehot.device
        t = torch.randint(1, self.num_diffusion_steps + 1, (B,), device=device)

        # 3) mask occupied voxels + extra masks based on critic statistics
        base_mask = self._generate_random_mask(ligand_onehot, t)  # [B,D,H,W] bool

        if self.use_critic:
            extra_counts = self._sample_extra_mask_counts(t)  # [B]
            # Add the requested number of additional masked voxels per sample
            mask_pos = base_mask.clone()
            B, D, H, W = mask_pos.shape
            for b in range(B):
                n_extra = extra_counts[b].item()
                if n_extra == 0:
                    continue
                # Identify currently *unmasked* voxels
                available = (~mask_pos[b]).nonzero(as_tuple=False)
                if available.size(0) == 0:
                    continue
                n_extra = min(n_extra, available.size(0))
                perm = torch.randperm(available.size(0), device=prot_vox.device)
                extra_sel = available[perm[:n_extra]]
                mask_pos[b, extra_sel[:, 0], extra_sel[:, 1], extra_sel[:, 2]] = True
        else:
            mask_pos = base_mask

        # 4) create masked ligand input (add mask channel)
        ligand_masked = self._apply_mask(ligand_onehot, mask_pos)  # [B,10,D,H,W]

        # 5) model input = protein + ligand_masked
        model_input = torch.cat([prot_vox, ligand_masked], dim=1)
        pred_logits = self.model(model_input)
        pred_vox = torch.sigmoid(pred_logits)

        # 6) loss on masked voxels only
        loss_dict = self._masked_bce_dice_loss(pred_logits, ligand_onehot, mask_pos)
        total_loss = sum(loss_dict.values())

        # ---------------- Critic training ----------------
        if self.use_critic:
            with torch.no_grad():
                # Detach predictions before using them to corrupt the ligand so
                # the generator is *not* updated through the critic pathway.
                pred_onehot = self._discretise_ligand(torch.sigmoid(pred_logits).detach(), threshold=self.discretize_threshold)

            ligand_corrupted = self._corrupt_ligand(ligand_onehot, pred_onehot, mask_pos)

            # Critic input: protein + (corrupted ligand + mask channel)
            ligand_corrupted_with_mask = self._apply_mask(ligand_corrupted, mask_pos)
            critic_input = torch.cat([prot_vox, ligand_corrupted_with_mask], dim=1)

            critic_logits = self.critic(critic_input)
            # Ground-truth incorrect mask: 1 if ligand_corrupted differs from target
            incorrect_gt = (ligand_corrupted != ligand_onehot).any(dim=1, keepdim=True).float()

            unmasked_mask = (~mask_pos).unsqueeze(1).float()

            critic_bce = F.binary_cross_entropy_with_logits(
                critic_logits, incorrect_gt, reduction='none'
            )
            critic_bce = (critic_bce * unmasked_mask).sum() / unmasked_mask.sum().clamp_min(1.0)

            # Update running statistics of re-masked voxels (how many the critic
            # *would* re-mask).  We threshold the critic prediction to compute a
            # count for statistics only – this has no effect on gradients.
            with torch.no_grad():
                critic_probs = torch.sigmoid(critic_logits)
                re_mask_pred = (critic_probs > self.critic_incorrect_threshold) & (unmasked_mask > 0)
                re_mask_counts = re_mask_pred.sum(dim=(1, 2, 3, 4)).long()  # [B]
                critic_remask_prop = re_mask_pred.sum().item() / max(1, unmasked_mask.sum().item())
                re_mask_accuracy = ((re_mask_pred == incorrect_gt)*unmasked_mask).sum().item() / max(1, unmasked_mask.sum().item())
                self._update_critic_stats(t, re_mask_counts)

            # Combine losses
            loss_dict["critic_bce"] = critic_bce
            total_loss = total_loss + self.critic_loss_weight * critic_bce

        output = {
            "predicted_ligand_logits": pred_logits,
            "predicted_ligand_voxels": pred_vox,
            "critic_remask_prop": critic_remask_prop,
            "critic_remask_accuracy": re_mask_accuracy,
            "loss": total_loss,
        }
        if isinstance(loss_dict, dict):
            output.update(loss_dict)
        return output

    def training_step(self, batch, batch_idx):
        if "load_time" in batch:
            self.log("train/load_time", batch["load_time"].mean().item(), on_step=True, on_epoch=False, prog_bar=True)

        outputs = self(batch["protein"], labels=batch["ligand"])
        # ``outputs`` is a dictionary – extract what we need
        pred_vox = outputs["predicted_ligand_voxels"]
        loss = outputs["loss"]

        # Log each individual loss component (except the prediction tensor)
        for k, v in outputs.items():
            if k in {"predicted_ligand_voxels", "predicted_ligand_logits"}:
                continue
            # Per-batch logging
            self.log(f"train/batch_{k}", v, on_step=True, on_epoch=False, prog_bar=(k=="loss"))
            # Per-epoch logging
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=(k=="loss"))

        # Channel statistics
        self.log_channel_means(batch, pred_vox)

        # Visualisation
        if batch_idx == 0 and self.visualise_train:
            try:
                outputs_for_viz = pred_vox[:self.n_samples_for_visualisation].float().detach().cpu().numpy()
                visualise_batch(
                    batch["ligand"][:self.n_samples_for_visualisation], 
                    outputs_for_viz[:self.n_samples_for_visualisation], 
                    batch["name"][:self.n_samples_for_visualisation], 
                    save_dir=self.img_save_dir, 
                    batch=str(batch_idx) + "_train"
                )
            except Exception as e:
                print(f"Error visualising batch {batch_idx}: {e}")

        return loss

    def validation_step(self, batch, batch_idx):
        # ------------ masked loss (same as training) -------------
        outputs_masked = self(batch["protein"], labels=batch["ligand"])
        masked_loss = outputs_masked["loss"]
        self.log(f"val/loss", masked_loss, on_step=False, on_epoch=True, prog_bar=True)  # always log this for checkpoints etc.

        for k, v in outputs_masked.items():
            if k in {"predicted_ligand_voxels", "predicted_ligand_logits"}:
                continue
            self.log(f"val/masked_{k}", v, on_step=False, on_epoch=True, prog_bar=(k=="loss"))

        # ------------ full diffusion evaluation (every 10 batches) ------------------
        if batch_idx % 10 == 0:
            with torch.no_grad():
                pred_vox_full = self.sample_ligand(batch["protein"])  # [B,9,...]

            target_onehot_full = self._discretise_ligand(batch["ligand"], threshold=self.discretize_threshold)
            full_loss_dict = self._full_bce_dice_loss(pred_vox_full, target_onehot_full)

            for k, v in full_loss_dict.items():
                self.log(f"val/full_{k}", v, on_step=False, on_epoch=True, prog_bar=(k=="loss"))

            # Visualisation using full predictions
            if self.visualise_val and batch_idx in [0, 50, 100]:
                try:
                    save_dir = f"{self.img_save_dir}/val" if self.img_save_dir else None
                    outputs_for_viz = pred_vox_full[:self.n_samples_for_visualisation].float().detach().cpu().numpy()
                    lig, pred, names = batch["ligand"][:self.n_samples_for_visualisation], outputs_for_viz[:self.n_samples_for_visualisation], batch["name"][:self.n_samples_for_visualisation]

                    visualise_batch(
                        lig,
                        pred,
                        names,
                        save_dir=save_dir,
                        batch=str(batch_idx) + "_val"
                    )
                except Exception as e:
                    print(f"Error visualising batch {batch_idx}: {e}")

        return masked_loss

    def test_step(self, batch, batch_idx):
        outputs = self(batch["protein"], labels=batch["ligand"])
        return outputs["loss"]

    def configure_optimizers(self) -> Dict[str, Any]:

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-5,
        )

        # Resolve scheduler configuration
        scheduler_config = self.scheduler_config.copy()

        if not scheduler_config:
            # No scheduler requested – return optimizer only
            return {"optimizer": optimizer}

        scheduler_type = scheduler_config.get("type", "step")

        if scheduler_type == "step":
            step_size = scheduler_config.get("step_size", 100)
            gamma = scheduler_config.get("gamma", 0.997)
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": scheduler_config.get("interval", "step"),
                    "frequency": scheduler_config.get("frequency", 1),
                },
            }
        else:
            # Use HuggingFace get_scheduler for advanced schedulers
            num_training_steps = self.trainer.estimated_stepping_batches
            num_warmup_steps = scheduler_config.get("num_warmup_steps", 0)

            if isinstance(num_warmup_steps, float) and 0 <= num_warmup_steps < 1:
                num_warmup_steps = int(num_training_steps * num_warmup_steps)

            scheduler_specific_kwargs = {}

            if scheduler_type == "cosine_with_restarts":
                scheduler_specific_kwargs["num_cycles"] = scheduler_config.get("num_cycles", 1)
            elif scheduler_type == "cosine_with_min_lr":
                scheduler_specific_kwargs["min_lr_rate"] = scheduler_config.get("min_lr_rate", 0.1)

            scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=scheduler_specific_kwargs,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": scheduler_config.get("interval", "step"),
                    "frequency": scheduler_config.get("frequency", 1),
                },
            }

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
    
    def log_channel_means(self, batch, pred_vox):
        """Log mean values for each channel of protein, true ligand and prediction."""
        n_lig_channels = batch['ligand'].shape[1]
        self.log_dict({
            f"channel_mean/ligand_{channel}": batch['ligand'][:, channel, ...].mean().detach().item()
            for channel in range(n_lig_channels)
        })
        self.log_dict({
            f"channel_mean/pred_ligand_{channel}": pred_vox[:, channel, ...].mean().detach().item()
            for channel in range(n_lig_channels)
        })
        n_prot_channels = batch['protein'].shape[1]
        self.log_dict({
            f"channel_mean/protein_{channel}": batch['protein'][:, channel, ...].mean().detach().item()
            for channel in range(n_prot_channels)
        })

    # ------------------------------------------------------------------
    # Masked BCE + Dice combined loss
    # ------------------------------------------------------------------
    def _masked_bce_dice_loss(self, pred_logits: torch.Tensor, target_onehot: torch.Tensor, mask_pos: torch.Tensor):
        """Compute BCE-Dice only over *masked* voxels.

        Parameters
        ----------
        pred_logits : [B, 9, D, H, W]
        target_onehot : [B, 9, D, H, W]
        mask_pos : [B, D, H, W] (bool)
        """
        mask = mask_pos.unsqueeze(1).float()  # broadcast over channel

        # ------------------ BCE ------------------
        bce = F.binary_cross_entropy_with_logits(pred_logits, target_onehot, reduction='none')
        bce = (bce * mask).sum() / mask.sum().clamp_min(1.0)

        # ------------------ Dice -----------------
        pred_prob = torch.sigmoid(pred_logits)
        dice_coeff = compute_per_channel_dice(pred_prob * mask, target_onehot * mask)
        dice = 1.0 - dice_coeff.mean()

        return {"bce": bce, "dice": dice}

    # ------------------------------------------------------------------
    # Full-grid BCE + Dice for end-to-end diffusion evaluation
    # ------------------------------------------------------------------
    def _full_bce_dice_loss(self, pred_prob: torch.Tensor, target_onehot: torch.Tensor):
        """Loss over the entire grid (no masking).

        Parameters
        ----------
        pred_prob : [B, 9, D, H, W] (sigmoid outputs or binary 0/1)
        target_onehot : [B, 9, D, H, W]
        """
        eps = 1e-6
        bce = F.binary_cross_entropy_with_logits(torch.logit(pred_prob.clamp(eps, 1 - eps)), target_onehot)
        dice_coeff = compute_per_channel_dice(pred_prob, target_onehot)
        dice = 1.0 - dice_coeff.mean()
        return {"bce": bce, "dice": dice, "loss": bce + dice}

    # ------------------------------------------------------------------
    # Critic helper utilities
    # ------------------------------------------------------------------

    def _sample_extra_mask_counts(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Sample additional mask counts based on the running critic statistics.

        The returned tensor has shape ``[B]`` and indicates how many *extra*
        voxels should be added to the original random mask so that the
        distribution of masked voxels during training matches that encountered
        when using the critic at inference time.
        """
        if not self.use_critic:
            return torch.zeros_like(timesteps, dtype=torch.long)

        device = timesteps.device
        # Gather mean/variance for each example
        means = self.critic_rm_mean[timesteps].to(device)
        counts = self.critic_rm_count[timesteps].to(device)
        vars_ = torch.zeros_like(means, dtype=torch.float)
        mask_nonzero = counts > 1
        vars_[mask_nonzero] = (self.critic_rm_M2[timesteps][mask_nonzero] / (counts[mask_nonzero] - 1)).to(device)

        stds = vars_.clamp_min(0.0).sqrt()

        # Normal sampling can occasionally be negative – clamp at 0.
        extra = torch.normal(means, stds).clamp_min(0.0).round().long()
        return extra

    def _update_critic_stats(self, timesteps: torch.Tensor, re_mask_counts: torch.Tensor):
        """Update running mean/variance of critic re-mask counts (Welford).

        Parameters
        ----------
        timesteps : [B] (long)
            Each element is the diffusion step *t* for that sample.
        re_mask_counts : [B] (long)
            Number of voxels re-masked by the critic for the corresponding
            sample.
        """
        if not self.use_critic:
            return

        for t_val, cnt in zip(timesteps.tolist(), re_mask_counts.tolist()):
            t_idx = int(t_val)
            cnt_tensor = torch.tensor(cnt, device=self.critic_rm_count.device, dtype=torch.float)

            # Update count
            self.critic_rm_count[t_idx] += 1
            n = self.critic_rm_count[t_idx].item()

            # Welford update
            delta = cnt_tensor - self.critic_rm_mean[t_idx]
            self.critic_rm_mean[t_idx] += delta / n
            delta2 = cnt_tensor - self.critic_rm_mean[t_idx]
            self.critic_rm_M2[t_idx] += delta * delta2

    def _corrupt_ligand(self, ligand_onehot: torch.Tensor, pred_onehot: torch.Tensor, mask_pos: torch.Tensor) -> torch.Tensor:
        """Return a *corrupted* ligand tensor for critic training.

        A random fraction of *unmasked* voxels are replaced with the generator
        predictions ("false" values).  The number of corrupted voxels for each
        sample is drawn uniformly between 0 and ``max_corrupt_ratio *
        n_masked``.
        """
        if not self.use_critic:
            # No corruption requested – simply return the ground-truth ligand.
            return ligand_onehot.clone()

        B, _, D, H, W = ligand_onehot.shape
        device = ligand_onehot.device

        ligand_corrupted = ligand_onehot.clone()

        for b in range(B):
            n_masked = mask_pos[b].sum().item()
            max_corrupt = int(self.max_corrupt_ratio * n_masked)
            if max_corrupt == 0:
                continue
            n_corrupt = max_corrupt
            # n_corrupt = torch.randint(0, max_corrupt + 1, (1,), device=device).item()  # TODO consider removing this

            # Coordinates of voxels that are currently *unmasked*
            unmasked_coords = (~mask_pos[b]).nonzero(as_tuple=False)
            if unmasked_coords.size(0) == 0:
                continue

            # If we request more corrupt voxels than available, cap it.
            n_corrupt = min(n_corrupt, unmasked_coords.size(0))
            if n_corrupt == 0:
                continue

            perm = torch.randperm(unmasked_coords.size(0), device=device)
            sel = unmasked_coords[perm[:n_corrupt]]

            # Set selected voxels to generator predictions (detached!)
            for idx in sel:
                z, y, x = idx.tolist()
                ligand_corrupted[b, :, z, y, x] = pred_onehot[b, :, z, y, x]

        return ligand_corrupted