"""A 2D backbone with a spatiotemporal aerosol adapter."""
from __future__ import annotations

from src.models.unet3d_grouped import GroupedConditionalUNet3D, _TORCH_IMPORT_ERROR, torch

if torch is None:  # pragma: no cover

    class GroupedAerosolResidualST48UNet3D:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"PyTorch is required: {_TORCH_IMPORT_ERROR}")

else:
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    from src.models.unet3d_grouped import _groups

    class FactorizedSpatialTemporalBlock(nn.Module):
        """Dense horizontal and temporal convolutions on [B,C,T,H,W]."""

        def __init__(self, channels: int = 48):
            super().__init__()
            self.norm_spatial = nn.GroupNorm(_groups(channels), channels)
            self.spatial = nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1))
            self.norm_temporal = nn.GroupNorm(_groups(channels), channels)
            self.temporal = nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0))

        def forward(self, values):
            spatial = self.spatial(F.silu(self.norm_spatial(values)))
            return values + self.temporal(F.silu(self.norm_temporal(spatial)))


    class HeightOrderedSpatialTemporalAerosolAdapter(nn.Module):
        """Ordered native levels, bounded weather/step FiLM and one zero head.

        The vertical encoder treats each forecast hour independently. Only after
        ordered height flattening do the two ST blocks mix horizontal neighbours
        and forecast hours. GroupNorm also couples spatial statistics.
        """

        def __init__(
            self,
            wrf_cloud_embed_channels: int = 80,
            output_channels: int = 160,
            hidden_channels: int = 16,
        ):
            super().__init__()
            if min(wrf_cloud_embed_channels, output_channels, hidden_channels) <= 0:
                raise ValueError("Aerosol adapter dimensions must be positive")
            self.species = 9
            self.levels = 6
            self.hidden_channels = int(hidden_channels)
            self.st_channels = 48
            self.wrf_cloud_embed_channels = int(wrf_cloud_embed_channels)
            self.output_channels = int(output_channels)
            self.profile = nn.Sequential(
                nn.Conv3d(self.species, self.hidden_channels, 1),
                nn.GroupNorm(_groups(self.hidden_channels), self.hidden_channels),
                nn.SiLU(),
                nn.Conv3d(self.hidden_channels, self.hidden_channels, (3, 1, 1), padding=(1, 0, 0)),
                nn.GroupNorm(_groups(self.hidden_channels), self.hidden_channels),
                nn.SiLU(),
            )
            self.height_reduce = nn.Sequential(
                nn.Conv3d(self.hidden_channels * self.levels, self.st_channels, 1),
                nn.GroupNorm(_groups(self.st_channels), self.st_channels),
                nn.SiLU(),
            )
            self.meteo_step_film = nn.Conv3d(self.wrf_cloud_embed_channels + 5, 2 * self.st_channels, 1)
            self.st_blocks = nn.Sequential(
                FactorizedSpatialTemporalBlock(self.st_channels),
                FactorizedSpatialTemporalBlock(self.st_channels),
            )
            self.output_projection = nn.Conv3d(self.st_channels, self.output_channels, 1)
            self.reset_condition_modulation()
            self.zero_output_projection()

        @torch.no_grad()
        def reset_condition_modulation(self) -> None:
            """Initial FiLM is the identity, never a second zero multiplier."""
            nn.init.zeros_(self.meteo_step_film.weight)
            nn.init.zeros_(self.meteo_step_film.bias)

        @torch.no_grad()
        def zero_output_projection(self) -> None:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

        def forward(self, aerosol, wrf_cloud_embedding, diffusion_step_map):
            if aerosol.ndim != 6 or tuple(aerosol.shape[1:3]) != (self.species, self.levels):
                raise ValueError(f"Expected aerosol [B,9,6,T,H,W], got {tuple(aerosol.shape)}")
            batch, _, _, time, height, width = aerosol.shape
            expected_weather = (batch, self.wrf_cloud_embed_channels, time, height, width)
            if tuple(wrf_cloud_embedding.shape) != expected_weather:
                raise ValueError(f"WRF-cloud embedding shape must be {expected_weather}")
            if tuple(diffusion_step_map.shape) != (batch, 5, time, height, width):
                raise ValueError("ST48 FiLM requires a five-channel sincos diffusion-step map")
            vertical = aerosol.permute(0, 3, 1, 2, 4, 5).reshape(
                batch * time, self.species, self.levels, height, width
            )
            encoded = self.profile(vertical)
            ordered = encoded.reshape(batch, time, self.hidden_channels * self.levels, height, width)
            hidden = self.height_reduce(ordered.permute(0, 2, 1, 3, 4))
            gamma, beta = self.meteo_step_film(
                torch.cat([wrf_cloud_embedding, diffusion_step_map], dim=1)
            ).chunk(2, dim=1)
            hidden = hidden * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * torch.tanh(beta)
            return self.output_projection(self.st_blocks(hidden))


    class GroupedAerosolResidualST48UNet3D(GroupedConditionalUNet3D):
        """Original grouped backbone with one ST48 condition residual."""

        adapter_prefix = "aerosol_residual_adapter."
        backbone_migration_id = "strict_2d_backbone_zero_residual_st48_v2"

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
            if (int(aerosol_3d_species), int(aerosol_3d_levels)) != (9, 6):
                raise ValueError("The ST48 residual adapter requires native 9x6 aerosol inputs")
            if int(aerosol_3d_spatial_pool) != 1:
                raise ValueError("The ST48 residual adapter requires pool1")
            if timestep_embedding != "sincos":
                raise ValueError("The ST48 residual adapter requires sincos timestep_embedding")
            if not freeze_backbone:
                raise ValueError("3DAer requires a frozen backbone")
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
            self.aerosol_3d_species = 9
            self.aerosol_3d_levels = 6
            self.aerosol_3d_embed_channels = int(aerosol_3d_embed_channels)
            self.aerosol_3d_activation_checkpointing = bool(aerosol_3d_activation_checkpointing)
            self.residual_enabled = bool(residual_enabled)
            self.aerosol_residual_adapter = HeightOrderedSpatialTemporalAerosolAdapter(
                wrf_cloud_embed_channels=wrf_cloud_embed_channels,
                output_channels=condition_embed_channels + wrf_cloud_embed_channels,
                hidden_channels=self.aerosol_3d_embed_channels,
            )
            for parameter in self.backbone_parameters():
                parameter.requires_grad_(False)
            self.backbone_frozen = True

        def adapter_parameters(self):
            return self.aerosol_residual_adapter.parameters()

        def backbone_parameters(self):
            return (parameter for name, parameter in self.named_parameters() if not name.startswith(self.adapter_prefix))

        def forward(self, noisy_target, condition, diffusion_timestep, wrf_cloud_condition=None, aerosol_3d_condition=None):
            if not self.residual_enabled:
                return super().forward(noisy_target, condition, diffusion_timestep, wrf_cloud_condition)
            if wrf_cloud_condition is None:
                raise ValueError("wrf_cloud_condition is required by the ST48 residual model")
            if condition.shape[0] != noisy_target.shape[0] or condition.shape[2:] != noisy_target.shape[2:]:
                raise ValueError("General condition must match target B/T/H/W")
            if wrf_cloud_condition.shape[0] != noisy_target.shape[0] or wrf_cloud_condition.shape[2:] != noisy_target.shape[2:]:
                raise ValueError("WRF-cloud condition must match target B/T/H/W")
            if aerosol_3d_condition is None:
                raise ValueError("aerosol_3d_condition is required when the residual adapter is enabled")
            expected = (noisy_target.shape[0], 9, 6, *noisy_target.shape[2:])
            if tuple(aerosol_3d_condition.shape) != expected:
                raise ValueError(f"Full-resolution aerosol shape {tuple(aerosol_3d_condition.shape)} != {expected}")
            general = self.general_stem(condition)
            wrf_cloud = self.wrf_cloud_stem(wrf_cloud_condition)
            step_map = self._timestep_map(diffusion_timestep, noisy_target.shape)
            if self.aerosol_3d_activation_checkpointing and self.training:
                residual = checkpoint(
                    self.aerosol_residual_adapter, aerosol_3d_condition, wrf_cloud, step_map, use_reentrant=False
                )
            else:
                residual = self.aerosol_residual_adapter(aerosol_3d_condition, wrf_cloud, step_map)
            fused_condition = torch.cat([general, wrf_cloud], dim=1) + residual
            x = torch.cat([noisy_target, fused_condition, step_map], dim=1)
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
