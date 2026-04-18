from typing import List

import numpy as np
import torch
import torch.nn as nn

from constants import ENTRY, EXIT, ROAD, SAVE


CELL_TYPES: List[str] = [ROAD, ENTRY, EXIT, SAVE]


def encode_grid(grid: np.ndarray, m: int, n: int) -> torch.Tensor:
    """One-hot 4-channel encoding: (m, n) -> (1, 4, m, n) float32."""
    encoded = np.zeros((4, m, n), dtype=np.float32)
    for ch, ct in enumerate(CELL_TYPES):
        encoded[ch] = (grid == ct).astype(np.float32)
    return torch.from_numpy(encoded).unsqueeze(0)


class FitnessSurrogate(nn.Module):
    """
    CNN surrogate: (B, 4, m, n) -> (B,) scalar fitness.

    4-channel input (ROAD, ENTRY, EXIT, SAVE).
    Dilated convolutions expand the receptive field to 17x17 - covers 15x15.

    Architecture - per-cell scoring + sum:
        Conv(4->32,  3x3, d=1) + BN + ReLU + Drop    RF:  3
        Conv(32->32, 3x3, d=1) + BN + ReLU + Drop    RF:  5
        Conv(32->16, 3x3, d=2) + BN + ReLU + Drop    RF:  9
        Conv(16->8,  3x3, d=4) + BN + ReLU           RF: 17
        Per-cell head: Conv(8->16, 1x1) + ReLU + Conv(16->1, 1x1)
        SUM over spatial dims -> scalar fitness (B,)

    No global pooling: each cell's gradient is its direct contribution.
    ~15 000 parameters.
    """

    def __init__(self, m: int, n: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.m, self.n = m, n
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 16, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(16, 8, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.cell_head = nn.Sequential(
            nn.Conv2d(8, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        cell_scores = self.cell_head(feats)
        return cell_scores.sum(dim=[1, 2, 3])
