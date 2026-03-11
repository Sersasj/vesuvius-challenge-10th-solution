"""Exponential Moving Average (EMA) for model weights."""
import torch
import torch.nn as nn
from copy import deepcopy


class EMA(nn.Module):
    """Exponential Moving Average of model weights.

    References:
        - https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/discussion/429060
        - https://github.com/Lightning-AI/pytorch-lightning/issues/10914

    Args:
        model: The model to track.
        decay: EMA decay factor (0.999 = slow update, 0.9 = fast update).
        warmup: Number of updates before starting EMA (uses direct copy during warmup).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: int = 0):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0

    def _update(self, model: nn.Module, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(),
                model.state_dict().values()
            ):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model: nn.Module):
        """Update EMA weights."""
        if self.num_updates < self.warmup:
            # During warmup, just copy weights directly
            self._update(model, update_fn=lambda e, m: m)
        else:
            # After warmup, apply EMA
            self._update(model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m)
        self.num_updates += 1

    def set(self, model: nn.Module):
        """Directly copy model weights to EMA (no decay)."""
        self._update(model, update_fn=lambda e, m: m)

    def forward(self, *args, **kwargs):
        """Forward pass using EMA weights."""
        return self.module(*args, **kwargs)
