from __future__ import annotations

import torch.nn as nn

from ml.models.fno import FNO1d
from ml.models.spectral import SpectralNet

MODELS: dict[str, type[nn.Module]] = {
    "spectral": SpectralNet,
    "fno": FNO1d,
}


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a registered model by name."""
    try:
        model_cls = MODELS[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown model '{name}'. Available: {available}") from exc
    return model_cls(**kwargs)
