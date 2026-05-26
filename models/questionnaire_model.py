"""Tabular MLP classifier for questionnaire-derived risk features."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.base_model import BaseScreeningModel


class QuestionnaireRiskModel(BaseScreeningModel):
    """MLP-based classifier trained on structured questionnaire feature vectors."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: list[int] | None = None,
        num_classes: int = 4,
        dropout: float = 0.3,
        device: str = "cpu",
    ) -> None:
        super().__init__(num_classes=num_classes, device=device)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.dropout_rate = dropout

        self.network: nn.Sequential
        self._build()

    def _build(self) -> None:
        """Stack fully-connected layers with BatchNorm and Dropout."""
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on a questionnaire feature vector batch.

        Args:
            x: Feature tensor of shape (B, input_dim).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        ...

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate-layer embedding for fusion."""
        ...
