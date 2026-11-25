from __future__ import annotations

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

    class ConditionalUNet3D:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"PyTorch is required for ConditionalUNet3D: {_TORCH_IMPORT_ERROR}")

else:

    def _groups(channels: int) -> int:
        for groups in [8, 4, 2, 1]:
            if channels % groups == 0:
                return groups
        return 1


    class ConvBlock3D(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(_groups(out_channels), out_channels),
                nn.SiLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(_groups(out_channels), out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)


    class ConditionalUNet3D(nn.Module):
        """Lightweight spatial 3D U-Net for [B,C,T,H,W] tensors."""

        def __init__(
            self,
            target_channels: int,
            condition_channels: int,
            base_channels: int = 16,
            channel_mult: list[int] | None = None,
            timestep_embedding: str = "scalar",
        ):
            super().__init__()
            channel_mult = channel_mult or [1, 2, 4]
            channels = [base_channels * m for m in channel_mult]
            self.timestep_embedding = str(timestep_embedding)
            timestep_channels = 5 if self.timestep_embedding == "sincos" else 1
            in_channels = target_channels + condition_channels + timestep_channels

            self.enc1 = ConvBlock3D(in_channels, channels[0])
            self.enc2 = ConvBlock3D(channels[0], channels[1])
            self.enc3 = ConvBlock3D(channels[1], channels[2])
            self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
            self.mid = ConvBlock3D(channels[2], channels[2])
            self.up2 = nn.Conv3d(channels[2] + channels[1], channels[1], kernel_size=1)
            self.dec2 = ConvBlock3D(channels[1], channels[1])
            self.up1 = nn.Conv3d(channels[1] + channels[0], channels[0], kernel_size=1)
            self.dec1 = ConvBlock3D(channels[0], channels[0])
            self.out = nn.Conv3d(channels[0], target_channels, kernel_size=1)

        def forward(self, noisy_target, condition, diffusion_timestep):
            if condition.shape[2:] != noisy_target.shape[2:]:
                raise ValueError(f"Condition shape {condition.shape} must match target T/H/W {noisy_target.shape}")
            b, _, t, h, w = noisy_target.shape
            step = diffusion_timestep.float().view(b, 1, 1, 1, 1)
            denom = max(float(getattr(self, "diffusion_steps", 1000)), 1.0)
            step_norm = step / denom
            if self.timestep_embedding == "sincos":
                two_pi = 6.283185307179586
                step_map = torch.cat(
                    [
                        step_norm,
                        torch.sin(two_pi * step_norm),
                        torch.cos(two_pi * step_norm),
                        torch.sin(0.5 * two_pi * step_norm),
                        torch.cos(0.5 * two_pi * step_norm),
                    ],
                    dim=1,
                ).expand(b, 5, t, h, w)
            else:
                step_map = step_norm.expand(b, 1, t, h, w)
            x = torch.cat([noisy_target, condition, step_map], dim=1)

            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            m = self.mid(e3)
            d2 = F.interpolate(m, size=e2.shape[2:], mode="trilinear", align_corners=False)
            d2 = self.dec2(self.up2(torch.cat([d2, e2], dim=1)))
            d1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
            d1 = self.dec1(self.up1(torch.cat([d1, e1], dim=1)))
            return self.out(d1)
