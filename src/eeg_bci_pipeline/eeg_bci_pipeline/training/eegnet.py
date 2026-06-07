"""EEGNet architecture for EEG classification (Lawhern et al. 2018)."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as _err:  # pragma: no cover
    raise RuntimeError("EEGNet requires PyTorch. Install: pip install torch") from _err


# block1 and block2 each average-pool the time axis (by 4 then 8), so the
# temporal dimension is reduced by 4 * 8 before the classifier. n_samples must
# be at least this large or the final feature map collapses to zero width.
_TEMPORAL_REDUCTION = 32


class EEGNet(nn.Module):
    def __init__(
        self,
        *,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        f1: int = 8,
        d: int = 2,
        kernel_length: int = 125,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        if n_samples < _TEMPORAL_REDUCTION:
            raise ValueError(f"n_samples must be at least {_TEMPORAL_REDUCTION}; got {n_samples}")
        f2 = f1 * d

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(f2, f2, (1, 16), groups=f2, padding="same", bias=False),
            nn.Conv2d(f2, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        self.classifier = nn.Linear(f2 * (n_samples // _TEMPORAL_REDUCTION), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x)
