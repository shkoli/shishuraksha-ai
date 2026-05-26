"""End-to-end multi-modal fusion model for psychiatric risk classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.base_model import BaseScreeningModel
from modules.fusion.attention_fusion import AttentionFusion, FusionInput, FusionOutput


class FusionRiskModel(BaseScreeningModel):
    """Wraps AttentionFusion into the BaseScreeningModel interface for training/inference."""

    def __init__(
        self,
        modality_dims: dict[str, int] | None = None,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_classes: int = 4,
        dropout: float = 0.3,
        device: str = "cpu",
    ) -> None:
        super().__init__(num_classes=num_classes, device=device)
        self.modality_dims = modality_dims or {
            "questionnaire": 64,
            "text": 768,
            "drawing": 1536,
            "facial": 128,
        }
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout

        self.fusion: AttentionFusion
        self._build()

    def _build(self) -> None:
        """Instantiate the AttentionFusion module."""
        ...

    def forward(self, fusion_input: FusionInput) -> torch.Tensor:
        """Forward pass returning risk logits.

        Args:
            fusion_input: FusionInput with available modality tensors.

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        ...

    def forward_with_attention(self, fusion_input: FusionInput) -> FusionOutput:
        """Forward pass returning full FusionOutput including attention weights."""
        ...

    def compute_loss(
        self,
        fusion_input: FusionInput,
        labels: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute weighted cross-entropy loss for training."""
        ...
