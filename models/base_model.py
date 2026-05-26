"""Abstract base class for all psychiatric screening models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class BaseScreeningModel(nn.Module, ABC):
    """Abstract base for all modality-specific and fusion screening models."""

    def __init__(self, num_classes: int = 4, device: str = "cpu") -> None:
        super().__init__()
        self.num_classes = num_classes
        self.device = device

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass returning raw logits."""
        ...

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Run inference and return probabilities and predicted class.

        Returns:
            Dict with keys: probabilities, predicted_class, confidence.
        """
        ...

    def predict_batch(self, batch: Any) -> list[dict[str, Any]]:
        """Run batched inference."""
        ...

    def save(self, path: str | Path) -> None:
        """Persist model weights and metadata to disk."""
        ...

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "BaseScreeningModel":
        """Load a saved model from disk."""
        ...

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self) -> None:
        """Freeze backbone weights, leaving only the classification head trainable."""
        ...

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters for full fine-tuning."""
        ...
