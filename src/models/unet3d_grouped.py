from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as activation_checkpoint
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


if torch is None:

    class GroupedConditionalUNet3D:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"PyTorch is required for GroupedConditionalUNet3D: {_TORCH_IMPORT_ERROR}")

else:

    def _groups(channels: int) -> int:
        for groups in (16, 8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1


    class ResidualConvBlock3D(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
            self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
            self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            residual = self.skip(x)
            x = self.conv1(F.silu(self.norm1(x)))
            x = self.conv2(F.silu(self.norm2(x)))
            return x + residual


    class ConditionStem3D(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(_groups(out_channels), out_channels),
                nn.SiLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(_groups(out_channels), out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)


    class AerosolVerticalStem(nn.Module):
        """Encode [B, species, levels, T, H, W] without flattening levels.

        Vertical-only kernels preserve the ordering and local interactions of
        the WRF-Chem layers.  Mean and max pooling provide complementary
        column summaries after that profile encoding, while the original AOD
        and 2-D column-summary channels remain available in the general stem.
        """

        def __init__(self, species: int, embed_channels: int, spatial_pool: int = 1):
            super().__init__()
            if int(spatial_pool) < 1:
                raise ValueError(f"aerosol_3d_spatial_pool must be >= 1, got {spatial_pool}")
            self.spatial_pool = int(spatial_pool)
            self.profile = nn.Sequential(
                nn.Conv3d(species, embed_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                nn.GroupNorm(_groups(embed_channels), embed_channels),
                nn.SiLU(inplace=True),
                nn.Conv3d(embed_channels, embed_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                nn.GroupNorm(_groups(embed_channels), embed_channels),
                nn.SiLU(inplace=True),
            )
            self.project = nn.Sequential(
                nn.Conv2d(embed_channels * 2, embed_channels, kernel_size=1),
                nn.GroupNorm(_groups(embed_channels), embed_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, aerosol_3d_condition):
            if aerosol_3d_condition.ndim != 6:
                raise ValueError(
                    "aerosol_3d_condition must be [B,species,levels,T,H,W], "
                    f"got {tuple(aerosol_3d_condition.shape)}"
                )
            batch, species, levels, time, height, width = aerosol_3d_condition.shape
            profiles = aerosol_3d_condition.permute(0, 3, 1, 2, 4, 5).reshape(
                batch * time, species, levels, height, width
            )
            # A full-resolution [species, levels, H, W] profile is extremely
            # expensive: the vertical encoder otherwise materializes two
            # 32-channel volumes for every forecast hour. Pool only in the
            # horizontal plane before the vertical-only convolutions, then
            # restore the learned column embedding to the satellite grid.
            # The original 14 x 34 fields remain the encoder input; they are
            # not flattened into the 2-D condition nor replaced by AOD.
            if self.spatial_pool > 1:
                profiles = F.avg_pool3d(
                    profiles,
                    kernel_size=(1, self.spatial_pool, self.spatial_pool),
                    stride=(1, self.spatial_pool, self.spatial_pool),
                )
            encoded = self.profile(profiles)
            mean_profile = encoded.mean(dim=2)
            max_profile = encoded.max(dim=2).values
            projected = self.project(torch.cat([mean_profile, max_profile], dim=1))
            if projected.shape[-2:] != (height, width):
                projected = F.interpolate(projected, size=(height, width), mode="bilinear", align_corners=False)
            return projected.reshape(batch, time, projected.shape[1], height, width).permute(0, 2, 1, 3, 4)


    class TemporalContextBlock(nn.Module):
        """Cheap global temporal attention broadcast back to the spatial bottleneck."""

        def __init__(self, channels: int, heads: int):
            super().__init__()
            if channels % heads != 0:
                raise ValueError(f"temporal attention channels={channels} must be divisible by heads={heads}")
            self.norm1 = nn.LayerNorm(channels)
            self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
            self.norm2 = nn.LayerNorm(channels)
            self.mlp = nn.Sequential(nn.Linear(channels, channels * 2), nn.SiLU(), nn.Linear(channels * 2, channels))

        def forward(self, x):
            pooled = x.mean(dim=(-2, -1)).transpose(1, 2)
            normed = self.norm1(pooled)
            attended, _ = self.attn(normed, normed, normed, need_weights=False)
            pooled = pooled + attended
            pooled = pooled + self.mlp(self.norm2(pooled))
            context = pooled.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
            return x + context


    class GroupedConditionalUNet3D(nn.Module):
        """B13 U-Net with separate general and WRF-cloud conditioning branches."""

        def __init__(
            self,
            target_channels: int,
            condition_channels: int,
            wrf_cloud_channels: int,
            base_channels: int = 160,
            channel_mult: list[int] | None = None,
            condition_embed_channels: int = 80,
            wrf_cloud_embed_channels: int = 80,
            aerosol_3d_species: int = 0,
            aerosol_3d_levels: int = 0,
            aerosol_3d_embed_channels: int = 32,
            aerosol_3d_spatial_pool: int = 1,
            aerosol_3d_activation_checkpointing: bool = False,
            temporal_attention_heads: int = 8,
            timestep_embedding: str = "sincos",
        ):
            super().__init__()
            channel_mult = channel_mult or [1, 2, 4]
            if len(channel_mult) != 3:
                raise ValueError("GroupedConditionalUNet3D requires exactly three channel multipliers")
            if wrf_cloud_channels <= 0:
                raise ValueError("GroupedConditionalUNet3D requires wrf_cloud_channels > 0")
            if (aerosol_3d_species > 0) != (aerosol_3d_levels > 0):
                raise ValueError("aerosol_3d_species and aerosol_3d_levels must both be positive or both be zero")
            channels = [int(base_channels * m) for m in channel_mult]
            self.timestep_embedding = str(timestep_embedding)
            timestep_channels = 5 if self.timestep_embedding == "sincos" else 1

            self.general_stem = ConditionStem3D(condition_channels, condition_embed_channels)
            self.wrf_cloud_stem = ConditionStem3D(wrf_cloud_channels, wrf_cloud_embed_channels)
            self.aerosol_3d_species = int(aerosol_3d_species)
            self.aerosol_3d_levels = int(aerosol_3d_levels)
            self.aerosol_3d_embed_channels = int(aerosol_3d_embed_channels)
            self.aerosol_3d_spatial_pool = int(aerosol_3d_spatial_pool)
            self.aerosol_3d_activation_checkpointing = bool(
                aerosol_3d_activation_checkpointing
            )
            self.aerosol_3d_stem = (
                AerosolVerticalStem(
                    self.aerosol_3d_species,
                    self.aerosol_3d_embed_channels,
                    spatial_pool=self.aerosol_3d_spatial_pool,
                )
                if self.aerosol_3d_species > 0
                else None
            )
            fused_condition_channels = condition_embed_channels + wrf_cloud_embed_channels + (
                self.aerosol_3d_embed_channels if self.aerosol_3d_stem is not None else 0
            )
            input_channels = target_channels + timestep_channels + fused_condition_channels

            self.enc1 = ResidualConvBlock3D(input_channels, channels[0])
            self.enc2 = ResidualConvBlock3D(channels[0], channels[1])
            self.enc3 = ResidualConvBlock3D(channels[1], channels[2])
            self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
            self.cond2 = nn.Conv3d(fused_condition_channels, channels[1], kernel_size=1)
            self.cond3 = nn.Conv3d(fused_condition_channels, channels[2], kernel_size=1)
            self.mid = ResidualConvBlock3D(channels[2], channels[2])
            self.temporal_context = TemporalContextBlock(channels[2], temporal_attention_heads)
            self.up2 = nn.Conv3d(channels[2] + channels[1], channels[1], kernel_size=1)
            self.dec2 = ResidualConvBlock3D(channels[1], channels[1])
            self.up1 = nn.Conv3d(channels[1] + channels[0], channels[0], kernel_size=1)
            self.dec1 = ResidualConvBlock3D(channels[0], channels[0])
            self.out_norm = nn.GroupNorm(_groups(channels[0]), channels[0])
            self.out = nn.Conv3d(channels[0], target_channels, kernel_size=1)

        def _timestep_map(self, diffusion_timestep, shape):
            b, _, t, h, w = shape
            step = diffusion_timestep.float().view(b, 1, 1, 1, 1)
            denom = max(float(getattr(self, "diffusion_steps", 1000)), 1.0)
            step_norm = step / denom
            if self.timestep_embedding == "sincos":
                two_pi = 6.283185307179586
                values = torch.cat(
                    [
                        step_norm,
                        torch.sin(two_pi * step_norm),
                        torch.cos(two_pi * step_norm),
                        torch.sin(0.5 * two_pi * step_norm),
                        torch.cos(0.5 * two_pi * step_norm),
                    ],
                    dim=1,
                )
            else:
                values = step_norm
            return values.expand(b, values.shape[1], t, h, w)

        def forward(
            self,
            noisy_target,
            condition,
            diffusion_timestep,
            wrf_cloud_condition=None,
            aerosol_3d_condition=None,
        ):
            if wrf_cloud_condition is None:
                raise ValueError("wrf_cloud_condition is required by GroupedConditionalUNet3D")
            if condition.shape[2:] != noisy_target.shape[2:]:
                raise ValueError(f"Condition shape {condition.shape} must match target T/H/W {noisy_target.shape}")
            if wrf_cloud_condition.shape[2:] != noisy_target.shape[2:]:
                raise ValueError(
                    f"WRF cloud condition shape {wrf_cloud_condition.shape} must match target T/H/W {noisy_target.shape}"
                )
            if self.aerosol_3d_stem is None:
                if aerosol_3d_condition is not None:
                    raise ValueError("This checkpoint/model was built without a 3-D aerosol branch")
            else:
                if aerosol_3d_condition is None:
                    raise ValueError("aerosol_3d_condition is required by the configured 3-D aerosol branch")
                expected = (self.aerosol_3d_species, self.aerosol_3d_levels)
                if tuple(aerosol_3d_condition.shape[1:3]) != expected:
                    raise ValueError(
                        f"3-D aerosol dimensions {tuple(aerosol_3d_condition.shape[1:3])} != {expected}"
                    )
                if aerosol_3d_condition.shape[3] != noisy_target.shape[2]:
                    raise ValueError(
                        "3-D aerosol time axis must match target: "
                        f"{aerosol_3d_condition.shape[3]} vs {noisy_target.shape[2]}"
                    )
            general = self.general_stem(condition)
            wrf_cloud = self.wrf_cloud_stem(wrf_cloud_condition)
            parts = [general, wrf_cloud]
            if self.aerosol_3d_stem is not None:
                if self.aerosol_3d_activation_checkpointing and self.training:
                    aerosol_3d = activation_checkpoint(
                        self.aerosol_3d_stem,
                        aerosol_3d_condition,
                        use_reentrant=False,
                    )
                else:
                    aerosol_3d = self.aerosol_3d_stem(aerosol_3d_condition)
                if aerosol_3d.shape[2:] != noisy_target.shape[2:]:
                    aerosol_3d = F.interpolate(
                        aerosol_3d,
                        size=noisy_target.shape[2:],
                        mode="trilinear",
                        align_corners=False,
                    )
                parts.append(aerosol_3d)
            fused_condition = torch.cat(parts, dim=1)
            step_map = self._timestep_map(diffusion_timestep, noisy_target.shape)
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
