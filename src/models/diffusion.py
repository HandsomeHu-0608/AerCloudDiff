from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


VALID_SAMPLERS = {"ddpm", "ddim"}
VALID_CLAMP_MODES = {"none", "final", "per_step", "dynamic_threshold"}


@dataclass(frozen=True)
class SamplingConfig:
    """Resolved diffusion sampling controls shared by train/eval/plots/cache.

    Old project configs have no top-level ``sampling`` section. They resolve to
    the former behavior: DDIM clamps reconstructed x0 at every step, while DDPM
    only clamps the final sample.
    """

    sampler: str = "ddim"
    steps: int = 100
    eta: float = 0.0
    guidance_scale: float = 0.0
    clamp_mode: str = "per_step"
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    dynamic_threshold_percentile: float = 0.995
    use_ema: bool = True
    seed: int = 42
    ensemble_members: int = 1
    seed_mode: str = "legacy_global"
    offset_noise_enabled: bool = False
    offset_noise_scale: float = 0.0
    offset_noise_mode: str = "per_channel_time"
    legacy_config: bool = False

    def __post_init__(self) -> None:
        sampler = self.sampler.lower()
        clamp_mode = self.clamp_mode.lower()
        if sampler not in VALID_SAMPLERS:
            raise ValueError(f"sampling.sampler must be one of {sorted(VALID_SAMPLERS)}, got {self.sampler!r}")
        if clamp_mode not in VALID_CLAMP_MODES:
            raise ValueError(
                f"sampling.clamp_mode must be one of {sorted(VALID_CLAMP_MODES)}, got {self.clamp_mode!r}"
            )
        if self.steps <= 0:
            raise ValueError("sampling.steps must be positive")
        if self.clamp_max <= self.clamp_min:
            raise ValueError("sampling.clamp_max must exceed sampling.clamp_min")
        if not 0.0 < self.dynamic_threshold_percentile <= 1.0:
            raise ValueError("sampling.dynamic_threshold_percentile must be in (0, 1]")
        if self.ensemble_members <= 0:
            raise ValueError("sampling.ensemble_members must be positive")
        if self.seed_mode not in {"legacy_global", "sample_keyed"}:
            raise ValueError("sampling.seed_mode must be legacy_global or sample_keyed")
        if self.offset_noise_mode != "per_channel_time":
            raise ValueError("Only offset_noise_mode=per_channel_time is supported")

    @classmethod
    def from_config(cls, config: dict[str, Any], **overrides: Any) -> "SamplingConfig":
        model_cfg = config.get("model", {})
        explicit = config.get("sampling")
        diffusion = config.get("diffusion", {})
        legacy = explicit is None
        values = dict(explicit or {})
        sampler = str(values.get("sampler", model_cfg.get("sampler", "ddpm"))).lower()
        legacy_clamp = "per_step" if sampler == "ddim" else "final"
        resolved = {
            "sampler": sampler,
            "steps": int(values.get("steps", model_cfg.get("sampling_steps", model_cfg.get("diffusion_steps", 1000)))),
            "eta": float(values.get("eta", model_cfg.get("ddim_eta", 0.0))),
            "guidance_scale": float(values.get("guidance_scale", 0.0)),
            "clamp_mode": str(values.get("clamp_mode", legacy_clamp)).lower(),
            "clamp_min": float(values.get("clamp_min", 0.0)),
            "clamp_max": float(values.get("clamp_max", 1.0)),
            "dynamic_threshold_percentile": float(values.get("dynamic_threshold_percentile", 0.995)),
            "use_ema": bool(values.get("use_ema", True)),
            "seed": int(values.get("seed", config.get("training", {}).get("seed", 42))),
            "ensemble_members": int(values.get("ensemble_members", 1)),
            "seed_mode": str(values.get("seed_mode", "legacy_global")),
            "offset_noise_enabled": bool(diffusion.get("offset_noise_enabled", False)),
            "offset_noise_scale": float(diffusion.get("offset_noise_scale", 0.0)),
            "offset_noise_mode": str(diffusion.get("offset_noise_mode", "per_channel_time")),
            "legacy_config": legacy,
        }
        resolved.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**resolved)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def guided_noise_prediction(pred_cond, pred_uncond, guidance_scale: float):
    """SATCast-style delta guidance where scale=0 is the conditional baseline."""

    scale = float(guidance_scale)
    if scale == 0.0:
        return pred_cond
    if pred_uncond is None:
        raise ValueError("Non-zero guidance_scale requires an unconditional prediction")
    return pred_cond + scale * (pred_cond - pred_uncond)


if torch is None:

    class DDPMScheduler:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"PyTorch is required for DDPMScheduler: {_TORCH_IMPORT_ERROR}")

else:

    class DDPMScheduler:
        def __init__(
            self,
            steps: int = 1000,
            beta_start: float = 1.0e-4,
            beta_end: float = 2.0e-2,
            device: str | None = None,
        ):
            self.steps = int(steps)
            self.beta_start = float(beta_start)
            self.beta_end = float(beta_end)
            self.device = device or "cpu"
            self.betas = torch.linspace(beta_start, beta_end, self.steps, device=self.device)
            self.alphas = 1.0 - self.betas
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        def to(self, device):
            self.device = device
            self.betas = self.betas.to(device)
            self.alphas = self.alphas.to(device)
            self.alpha_bars = self.alpha_bars.to(device)
            return self

        def q_sample(self, x0, t, noise=None):
            if noise is None:
                noise = torch.randn_like(x0)
            alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1, 1)
            return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise, noise

        def predict_x0(self, xt, t, pred_noise):
            alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1, 1)
            return (xt - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)

        def noise_loss(self, pred_noise, true_noise, loss_mask=None):
            loss = (pred_noise - true_noise) ** 2
            if loss_mask is not None:
                loss = loss * loss_mask
                return loss.sum() / torch.clamp(loss_mask.sum(), min=1.0)
            return loss.mean()

        def _sampling_schedule(self, num_steps: int | None = None) -> list[int]:
            if num_steps is None or num_steps >= self.steps:
                return list(reversed(range(self.steps)))
            steps = torch.linspace(self.steps - 1, 0, int(num_steps)).round().long().tolist()
            steps = sorted(set(int(s) for s in steps), reverse=True)
            if steps[-1] != 0:
                steps.append(0)
            return steps

        def sampling_schedule(self, num_steps: int | None = None) -> list[int]:
            return self._sampling_schedule(num_steps)

        @staticmethod
        def _randn(shape, *, reference, generator=None):
            kwargs = {"device": reference.device, "dtype": reference.dtype}
            if generator is not None:
                kwargs["generator"] = generator
            return torch.randn(shape, **kwargs)

        @staticmethod
        def _clamp(x, sampling: SamplingConfig):
            return torch.clamp(x, sampling.clamp_min, sampling.clamp_max)

        @staticmethod
        def _dynamic_threshold(x, sampling: SamplingConfig):
            low, high = float(sampling.clamp_min), float(sampling.clamp_max)
            centered = 2.0 * (x - low) / (high - low) - 1.0
            flat = centered.abs().reshape(centered.shape[0], -1).float()
            scale = torch.quantile(flat, float(sampling.dynamic_threshold_percentile), dim=1)
            scale = torch.maximum(scale, torch.ones_like(scale)).to(dtype=x.dtype)
            scale = scale.view(-1, *([1] * (x.ndim - 1)))
            centered = torch.clamp(centered, -scale, scale) / scale
            return (0.5 * (centered + 1.0)) * (high - low) + low

        def _step_x0_constraint(self, pred_x0, sampling: SamplingConfig):
            if sampling.clamp_mode == "per_step":
                return self._clamp(pred_x0, sampling)
            if sampling.clamp_mode == "dynamic_threshold":
                return self._dynamic_threshold(pred_x0, sampling)
            return pred_x0

        @staticmethod
        def _model_prediction(
            model,
            xt,
            condition,
            timestep,
            sampling: SamplingConfig,
            model_kwargs: dict | None,
            unconditional_condition=None,
            unconditional_model_kwargs: dict | None = None,
        ):
            pred_cond = model(xt, condition, timestep, **(model_kwargs or {}))
            if float(sampling.guidance_scale) == 0.0:
                return pred_cond
            if unconditional_condition is None:
                raise ValueError("guidance_scale > 0 requires unconditional_condition")
            pred_uncond = model(
                xt,
                unconditional_condition,
                timestep,
                **(unconditional_model_kwargs or {}),
            )
            return guided_noise_prediction(pred_cond, pred_uncond, sampling.guidance_scale)

        @torch.no_grad()
        def sample(
            self,
            model,
            condition,
            target_channels: int,
            sampling: SamplingConfig,
            model_kwargs: dict | None = None,
            unconditional_condition=None,
            unconditional_model_kwargs: dict | None = None,
            initial_noise=None,
            generator=None,
        ):
            model.eval()
            shape = (
                condition.shape[0],
                int(target_channels),
                condition.shape[2],
                condition.shape[3],
                condition.shape[4],
            )
            xt = initial_noise
            if xt is None:
                xt = self._randn(shape, reference=condition, generator=generator)
                if sampling.offset_noise_enabled and sampling.offset_noise_scale != 0.0:
                    offset_shape = (shape[0], shape[1], shape[2], 1, 1)
                    xt = xt + sampling.offset_noise_scale * self._randn(
                        offset_shape, reference=condition, generator=generator
                    )
            if sampling.sampler == "ddim":
                xt = self._ddim_loop(
                    model,
                    xt,
                    condition,
                    sampling,
                    model_kwargs,
                    unconditional_condition,
                    unconditional_model_kwargs,
                    generator,
                )
            else:
                xt = self._ddpm_loop(
                    model,
                    xt,
                    condition,
                    sampling,
                    model_kwargs,
                    unconditional_condition,
                    unconditional_model_kwargs,
                    generator,
                )
            if sampling.clamp_mode in {"final", "per_step", "dynamic_threshold"}:
                xt = self._clamp(xt, sampling)
            return xt

        def _ddim_loop(
            self,
            model,
            xt,
            condition,
            sampling,
            model_kwargs,
            unconditional_condition,
            unconditional_model_kwargs,
            generator,
        ):
            steps = self._sampling_schedule(sampling.steps)
            for index, step in enumerate(steps):
                next_step = steps[index + 1] if index + 1 < len(steps) else -1
                t = torch.full((xt.shape[0],), step, dtype=torch.long, device=xt.device)
                pred_noise = self._model_prediction(
                    model,
                    xt,
                    condition,
                    t,
                    sampling,
                    model_kwargs,
                    unconditional_condition,
                    unconditional_model_kwargs,
                )
                alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1, 1)
                pred_x0 = self._step_x0_constraint(self.predict_x0(xt, t, pred_noise), sampling)
                if next_step < 0:
                    xt = pred_x0
                    continue
                alpha_bar_prev = self.alpha_bars[
                    torch.full((xt.shape[0],), next_step, dtype=torch.long, device=xt.device)
                ].view(-1, 1, 1, 1, 1)
                sigma = sampling.eta * torch.sqrt(
                    torch.clamp((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t), min=0.0)
                    * torch.clamp(1.0 - alpha_bar_t / alpha_bar_prev, min=0.0)
                )
                direction = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0))
                xt = torch.sqrt(alpha_bar_prev) * pred_x0 + direction * pred_noise
                if sampling.eta > 0.0:
                    xt = xt + sigma * self._randn(xt.shape, reference=xt, generator=generator)
            return xt

        def _ddpm_loop(
            self,
            model,
            xt,
            condition,
            sampling,
            model_kwargs,
            unconditional_condition,
            unconditional_model_kwargs,
            generator,
        ):
            steps = self._sampling_schedule(sampling.steps)
            for step in steps:
                t = torch.full((xt.shape[0],), step, dtype=torch.long, device=xt.device)
                pred_noise = self._model_prediction(
                    model,
                    xt,
                    condition,
                    t,
                    sampling,
                    model_kwargs,
                    unconditional_condition,
                    unconditional_model_kwargs,
                )
                if sampling.clamp_mode in {"per_step", "dynamic_threshold"}:
                    pred_x0 = self._step_x0_constraint(self.predict_x0(xt, t, pred_noise), sampling)
                    alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1, 1)
                    pred_noise = (xt - torch.sqrt(alpha_bar) * pred_x0) / torch.sqrt(1.0 - alpha_bar)
                beta = self.betas[t].view(-1, 1, 1, 1, 1)
                alpha = self.alphas[t].view(-1, 1, 1, 1, 1)
                alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1, 1)
                mean = (xt - beta / torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha)
                if step > 0:
                    xt = mean + torch.sqrt(beta) * self._randn(xt.shape, reference=xt, generator=generator)
                else:
                    xt = mean
            return xt

        @torch.no_grad()
        def p_sample_loop(
            self,
            model,
            condition,
            target_channels: int,
            num_steps: int | None = None,
            sampler: str = "ddpm",
            eta: float = 0.0,
            clip_denoised: bool = True,
            model_kwargs: dict | None = None,
        ):
            """Backward-compatible sampler used by pre-Round3b code."""

            sampler = sampler.lower()
            clamp_mode = ("per_step" if sampler == "ddim" else "final") if clip_denoised else "none"
            sampling = SamplingConfig(
                sampler=sampler,
                steps=int(num_steps or self.steps),
                eta=float(eta),
                clamp_mode=clamp_mode,
                legacy_config=True,
            )
            return self.sample(
                model,
                condition,
                target_channels=target_channels,
                sampling=sampling,
                model_kwargs=model_kwargs,
            )

        @torch.no_grad()
        def ddim_sample_loop(
            self,
            model,
            condition,
            target_channels: int,
            num_steps: int | None = None,
            eta: float = 0.0,
            clip_denoised: bool = True,
            initial_noise=None,
            model_kwargs: dict | None = None,
        ):
            sampling = SamplingConfig(
                sampler="ddim",
                steps=int(num_steps or self.steps),
                eta=float(eta),
                clamp_mode="per_step" if clip_denoised else "none",
                legacy_config=True,
            )
            return self.sample(
                model,
                condition,
                target_channels=target_channels,
                sampling=sampling,
                initial_noise=initial_noise,
                model_kwargs=model_kwargs,
            )
