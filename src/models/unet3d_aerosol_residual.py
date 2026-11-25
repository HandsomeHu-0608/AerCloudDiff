"""A height-aware aerosol adapter that leaves the trained 2-D path intact.

Unlike the legacy concatenation model, this model keeps *all* backbone tensor
names and dimensions unchanged.  Its sole new path contributes a residual to
the existing fused general/WRF-cloud condition.  A zero-initialized final
projection makes migration from a 2-D checkpoint initially function preserving.
"""
from __future__ import annotations

from src.models.unet3d_grouped import GroupedConditionalUNet3D, _TORCH_IMPORT_ERROR, torch

if torch is None:  # pragma: no cover

    class GroupedAerosolResidualUNet3D:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"PyTorch is required: {_TORCH_IMPORT_ERROR}")

else:
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    from src.models.unet3d_grouped import _groups

    class HeightOrderedAerosolAdapter(nn.Module):
        """Encode ordered heights without horizontal or mean/max height pooling.

        The vertical convolutions operate on [B*T,S,Z,H,W].  Height Z is then
        retained as an *ordered* channel dimension for the final projection;
        moving a plume to another band is not made invariant by height pooling.
        Only ``output_projection`` is zero initialized; there is no second zero
        gate that would prevent both branches of a product from learning.
        """

        def __init__(self, species: int, levels: int, hidden_channels: int, output_channels: int):
            super().__init__()
            if min(species, levels, hidden_channels, output_channels) <= 0:
                raise ValueError("Aerosol adapter dimensions must be positive")
            self.species = int(species)
            self.levels = int(levels)
            self.hidden_channels = int(hidden_channels)
            self.output_channels = int(output_channels)
            self.profile = nn.Sequential(
                nn.Conv3d(self.species, self.hidden_channels, kernel_size=1),
                nn.GroupNorm(_groups(self.hidden_channels), self.hidden_channels),
                nn.SiLU(),
                nn.Conv3d(self.hidden_channels, self.hidden_channels, (3, 1, 1), padding=(1, 0, 0)),
                nn.GroupNorm(_groups(self.hidden_channels), self.hidden_channels),
                nn.SiLU(),
            )
            self.output_projection = nn.Conv2d(self.hidden_channels * self.levels, self.output_channels, 1)
            self.zero_output_projection()

        @torch.no_grad()
        def zero_output_projection(self) -> None:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

        def forward(self, aerosol):
            if aerosol.ndim != 6 or tuple(aerosol.shape[1:3]) != (self.species, self.levels):
                raise ValueError(
                    f"Expected aerosol [B,{self.species},{self.levels},T,H,W], got {tuple(aerosol.shape)}"
                )
            batch, _, _, time, height, width = aerosol.shape
            profile = aerosol.permute(0, 3, 1, 2, 4, 5).reshape(
                batch * time, self.species, self.levels, height, width
            )
            encoded = self.profile(profile)
            # The level index is never averaged/maximized away or resized.
            ordered = encoded.reshape(batch * time, self.hidden_channels * self.levels, height, width)
            residual = self.output_projection(ordered)
            return residual.reshape(batch, time, self.output_channels, height, width).permute(0, 2, 1, 3, 4)


    class GroupedAerosolResidualUNet3D(GroupedConditionalUNet3D):
        """The unmodified 2-D grouped backbone plus an additive 3-D adapter."""

        adapter_prefix = "aerosol_residual_adapter."

        def __init__(
            self,
            target_channels: int,
            condition_channels: int,
            wrf_cloud_channels: int,
            base_channels: int = 160,
            channel_mult: list[int] | None = None,
            condition_embed_channels: int = 80,
            wrf_cloud_embed_channels: int = 80,
            aerosol_3d_species: int = 9,
            aerosol_3d_levels: int = 6,
            aerosol_3d_embed_channels: int = 16,
            aerosol_3d_spatial_pool: int = 1,
            aerosol_3d_activation_checkpointing: bool = True,
            temporal_attention_heads: int = 8,
            timestep_embedding: str = "sincos",
            freeze_backbone: bool = True,
            residual_enabled: bool = True,
        ):
            if int(aerosol_3d_spatial_pool) != 1:
                raise ValueError("The residual adapter requires pool1; horizontal pooling is not supported")
            super().__init__(
                target_channels=target_channels,
                condition_channels=condition_channels,
                wrf_cloud_channels=wrf_cloud_channels,
                base_channels=base_channels,
                channel_mult=channel_mult,
                condition_embed_channels=condition_embed_channels,
                wrf_cloud_embed_channels=wrf_cloud_embed_channels,
                aerosol_3d_species=0,
                aerosol_3d_levels=0,
                temporal_attention_heads=temporal_attention_heads,
                timestep_embedding=timestep_embedding,
            )
            # The inherited aerosol_3d_stem remains None, ensuring the original
            # backbone tensors have exactly the 2-D checkpoint's dimensions.
            self.aerosol_3d_species = int(aerosol_3d_species)
            self.aerosol_3d_levels = int(aerosol_3d_levels)
            self.aerosol_3d_embed_channels = int(aerosol_3d_embed_channels)
            self.aerosol_3d_activation_checkpointing = bool(aerosol_3d_activation_checkpointing)
            self.residual_enabled = bool(residual_enabled)
            self.aerosol_residual_adapter = HeightOrderedAerosolAdapter(
                self.aerosol_3d_species,
                self.aerosol_3d_levels,
                self.aerosol_3d_embed_channels,
                condition_embed_channels + wrf_cloud_embed_channels,
            )
            self.set_backbone_trainable(not freeze_backbone)

        def adapter_parameters(self):
            return self.aerosol_residual_adapter.parameters()

        def backbone_parameters(self):
            return (parameter for name, parameter in self.named_parameters() if not name.startswith(self.adapter_prefix))

        def set_backbone_trainable(self, trainable: bool) -> None:
            """Call before constructing DDP/optimizer, not mid-DDP execution."""
            for parameter in self.backbone_parameters():
                parameter.requires_grad_(bool(trainable))
            self.backbone_frozen = not bool(trainable)

        def forward(self, noisy_target, condition, diffusion_timestep, wrf_cloud_condition=None, aerosol_3d_condition=None):
            if not self.residual_enabled:
                return super().forward(noisy_target, condition, diffusion_timestep, wrf_cloud_condition)
            if wrf_cloud_condition is None:
                raise ValueError("wrf_cloud_condition is required by the residual model")
            if condition.shape[0] != noisy_target.shape[0] or condition.shape[2:] != noisy_target.shape[2:]:
                raise ValueError("General condition must match target B/T/H/W")
            if wrf_cloud_condition.shape[0] != noisy_target.shape[0] or wrf_cloud_condition.shape[2:] != noisy_target.shape[2:]:
                raise ValueError("WRF-cloud condition must match target B/T/H/W")
            if aerosol_3d_condition is None:
                raise ValueError("aerosol_3d_condition is required when the residual adapter is enabled")
            expected = (
                noisy_target.shape[0], self.aerosol_3d_species, self.aerosol_3d_levels, *noisy_target.shape[2:]
            )
            if tuple(aerosol_3d_condition.shape) != expected:
                raise ValueError(f"Full-resolution aerosol shape {tuple(aerosol_3d_condition.shape)} != {expected}")
            general = self.general_stem(condition)
            wrf_cloud = self.wrf_cloud_stem(wrf_cloud_condition)
            if self.aerosol_3d_activation_checkpointing and self.training:
                residual = checkpoint(self.aerosol_residual_adapter, aerosol_3d_condition, use_reentrant=False)
            else:
                residual = self.aerosol_residual_adapter(aerosol_3d_condition)
            fused_condition = torch.cat([general, wrf_cloud], dim=1) + residual
            step_map = self._timestep_map(diffusion_timestep, noisy_target.shape)
            x = torch.cat([noisy_target, fused_condition, step_map], dim=1)
            # Keep the base forward math and all pretrained parameter names.
            e1 = self.enc1(x)
            condition2 = self.pool(fused_condition)
            e2 = self.enc2(self.pool(e1)) + self.cond2(condition2)
            condition3 = self.pool(condition2)
            e3 = self.enc3(self.pool(e2)) + self.cond3(condition3)
            middle = self.temporal_context(self.mid(e3))
            d2 = F.interpolate(middle, size=e2.shape[2:], mode="trilinear", align_corners=False)
            d2 = self.dec2(self.up2(torch.cat([d2, e2], dim=1)))
            d1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
            d1 = self.dec1(self.up1(torch.cat([d1, e1], dim=1)))
            return self.out(F.silu(self.out_norm(d1)))
