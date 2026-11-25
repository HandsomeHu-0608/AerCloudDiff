from __future__ import annotations

from contextlib import contextmanager

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


class ExponentialMovingAverage:
    """Parameter EMA for more stable diffusion sampling checkpoints."""

    def __init__(self, model, decay: float = 0.999):
        if torch is None:
            raise ImportError(f"PyTorch is required for EMA: {_TORCH_IMPORT_ERROR}")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup = {}

    @torch.no_grad()
    def update(self, model) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
            else:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        self.num_updates += 1

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {name: tensor.detach().cpu() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state: dict, device=None) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.num_updates = int(state.get("num_updates", 0))
        shadow = state.get("shadow", state)
        self.shadow = {
            name: tensor.detach().clone().to(device) if device is not None else tensor.detach().clone()
            for name, tensor in shadow.items()
        }

    @torch.no_grad()
    def copy_to(self, model) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name].to(param.device, dtype=param.dtype))

    @torch.no_grad()
    def store(self, model) -> None:
        self.backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def restore(self, model) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.copy_(self.backup[name].to(param.device, dtype=param.dtype))
        self.backup = {}

    @contextmanager
    def average_parameters(self, model):
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)
