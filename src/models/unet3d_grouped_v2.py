from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


if torch is None:

    class GroupedConditionalUNet3DV2:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"PyTorch is required for GroupedConditionalUNet3DV2: {_TORCH_IMPORT_ERROR}")

else:

    def _groups(channels: int) -> int:
        for groups in (16, 8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1


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


    class DiffusionTimeEmbedding(nn.Module):
        """Sinusoidal diffusion-step embedding followed by a learned MLP."""

        def __init__(self, input_channels: int, embedding_channels: int, mode: str = "sincos"):
            super().__init__()
            self.input_channels = int(input_channels)
            self.mode = str(mode)
            mlp_input_channels = self.input_channels if self.mode == "sincos" else 1
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_channels, embedding_channels),
                nn.SiLU(),
                nn.Linear(embedding_channels, embedding_channels),
            )

        def _sincos(self, timestep):
            half = self.input_channels // 2
            if half == 0:
                return timestep.float().view(-1, 1)
            denominator = max(half - 1, 1)
            frequencies = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, device=timestep.device, dtype=torch.float32)
                / denominator
            )
            phases = timestep.float().view(-1, 1) * frequencies.view(1, -1)
            embedding = torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)
            if embedding.shape[1] < self.input_channels:
                embedding = F.pad(embedding, (0, self.input_channels - embedding.shape[1]))
            return embedding

        def forward(self, timestep, diffusion_steps: int = 1000):
            if self.mode == "sincos":
                values = self._sincos(timestep)
            else:
                denominator = max(float(diffusion_steps), 1.0)
                values = timestep.float().view(-1, 1) / denominator
            return self.mlp(values)


    class FiLMResidualConvBlock3D(nn.Module):
        """Residual 3-D convolution block modulated by the diffusion timestep."""

        def __init__(self, in_channels: int, out_channels: int, time_channels: int):
            super().__init__()
            self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
            self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
            self.time_affine = nn.Linear(time_channels, out_channels * 2)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv3d(in_channels, out_channels, kernel_size=1)
            )

        def forward(self, x, time_embedding):
            residual = self.skip(x)
            x = self.conv1(F.silu(self.norm1(x)))
            scale, shift = self.time_affine(time_embedding).chunk(2, dim=1)
            scale = scale[:, :, None, None, None]
            shift = shift[:, :, None, None, None]
            x = self.norm2(x) * (1.0 + scale) + shift
            x = self.conv2(F.silu(x))
            return x + residual


    class SpatialTemporalContextBlock(nn.Module):
        """Temporal self-attention independently at every spatial bottleneck cell.

        Unlike a spatially pooled temporal token, this preserves location-specific
        cloud evolution. Chunking controls the temporary attention workspace and
        does not change the result because spatial cells never attend to each other.
        """

        def __init__(self, channels: int, heads: int, chunk_size: int = 2048):
            super().__init__()
            if channels % heads != 0:
                raise ValueError(f"temporal attention channels={channels} must be divisible by heads={heads}")
            self.channels = int(channels)
            self.chunk_size = max(1, int(chunk_size))
            self.norm1 = nn.LayerNorm(channels)
            self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
            self.norm2 = nn.LayerNorm(channels)
            self.mlp = nn.Sequential(
                nn.Linear(channels, channels * 2),
                nn.SiLU(),
                nn.Linear(channels * 2, channels),
            )

        def _attend(self, tokens):
            normed = self.norm1(tokens)
            attended, _ = self.attn(normed, normed, normed, need_weights=False)
            tokens = tokens + attended
            return tokens + self.mlp(self.norm2(tokens))

        def forward(self, x):
            batch, channels, time, height, width = x.shape
            tokens = x.permute(0, 3, 4, 2, 1).reshape(batch * height * width, time, channels)
            if tokens.shape[0] <= self.chunk_size:
                tokens = self._attend(tokens)
            else:
                tokens = torch.cat(
                    [self._attend(chunk) for chunk in tokens.split(self.chunk_size, dim=0)],
                    dim=0,
                )
            return tokens.reshape(batch, height, width, time, channels).permute(0, 4, 3, 1, 2)


    class GroupedConditionalUNet3DV2(nn.Module):
        """Round3e grouped B13 U-Net with FiLM and spatially local temporal attention."""

        def __init__(
            self,
            target_channels: int,
            condition_channels: int,
            wrf_cloud_channels: int,
            base_channels: int = 176,
            channel_mult: list[int] | None = None,
            condition_embed_channels: int = 88,
            wrf_cloud_embed_channels: int = 88,
            time_embed_channels: int = 256,
            temporal_attention_heads: int = 8,
            temporal_attention_chunk_size: int = 2048,
            timestep_embedding: str = "sincos",
        ):
            super().__init__()
            channel_mult = channel_mult or [1, 2, 4]
            if len(channel_mult) != 3:
                raise ValueError("GroupedConditionalUNet3DV2 requires exactly three channel multipliers")
            if wrf_cloud_channels <= 0:
                raise ValueError("GroupedConditionalUNet3DV2 requires wrf_cloud_channels > 0")
            channels = [int(base_channels * multiplier) for multiplier in channel_mult]
            time_channels = int(time_embed_channels)
            if time_channels <= 0:
                raise ValueError("time_embed_channels must be positive")
            self.diffusion_steps = 1000

            self.general_stem = ConditionStem3D(condition_channels, condition_embed_channels)
            self.wrf_cloud_stem = ConditionStem3D(wrf_cloud_channels, wrf_cloud_embed_channels)
            fused_condition_channels = condition_embed_channels + wrf_cloud_embed_channels
            self.time_embedding = DiffusionTimeEmbedding(base_channels, time_channels, timestep_embedding)

            self.enc1 = FiLMResidualConvBlock3D(
                target_channels + fused_condition_channels, channels[0], time_channels
            )
            self.enc2 = FiLMResidualConvBlock3D(channels[0], channels[1], time_channels)
            self.enc3 = FiLMResidualConvBlock3D(channels[1], channels[2], time_channels)
            self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
            self.cond2 = nn.Conv3d(fused_condition_channels, channels[1], kernel_size=1)
            self.cond3 = nn.Conv3d(fused_condition_channels, channels[2], kernel_size=1)
            self.mid = FiLMResidualConvBlock3D(channels[2], channels[2], time_channels)
            self.temporal_context = SpatialTemporalContextBlock(
                channels[2], temporal_attention_heads, temporal_attention_chunk_size
            )
            self.up2 = nn.Conv3d(channels[2] + channels[1], channels[1], kernel_size=1)
            self.dec2 = FiLMResidualConvBlock3D(channels[1], channels[1], time_channels)
            self.up1 = nn.Conv3d(channels[1] + channels[0], channels[0], kernel_size=1)
            self.dec1 = FiLMResidualConvBlock3D(channels[0], channels[0], time_channels)
            self.out_norm = nn.GroupNorm(_groups(channels[0]), channels[0])
            self.out = nn.Conv3d(channels[0], target_channels, kernel_size=1)

        def forward(self, noisy_target, condition, diffusion_timestep, wrf_cloud_condition=None):
            if wrf_cloud_condition is None:
                raise ValueError("wrf_cloud_condition is required by GroupedConditionalUNet3DV2")
            if condition.shape[2:] != noisy_target.shape[2:]:
                raise ValueError(f"Condition shape {condition.shape} must match target T/H/W {noisy_target.shape}")
            if wrf_cloud_condition.shape[2:] != noisy_target.shape[2:]:
                raise ValueError(
                    f"WRF cloud condition shape {wrf_cloud_condition.shape} must match target T/H/W {noisy_target.shape}"
                )

            general = self.general_stem(condition)
            wrf_cloud = self.wrf_cloud_stem(wrf_cloud_condition)
            fused_condition = torch.cat([general, wrf_cloud], dim=1)
            time_embedding = self.time_embedding(diffusion_timestep, self.diffusion_steps)
            x = torch.cat([noisy_target, fused_condition], dim=1)

            e1 = self.enc1(x, time_embedding)
            condition2 = self.pool(fused_condition)
            e2 = self.enc2(self.pool(e1), time_embedding) + self.cond2(condition2)
            condition3 = self.pool(condition2)
            e3 = self.enc3(self.pool(e2), time_embedding) + self.cond3(condition3)
            middle = self.temporal_context(self.mid(e3, time_embedding))
            d2 = F.interpolate(middle, size=e2.shape[2:], mode="trilinear", align_corners=False)
            d2 = self.dec2(self.up2(torch.cat([d2, e2], dim=1)), time_embedding)
            d1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
            d1 = self.dec1(self.up1(torch.cat([d1, e1], dim=1)), time_embedding)
            return self.out(F.silu(self.out_norm(d1)))
