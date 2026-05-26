"""BanglaBERT-based text risk classification model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.base_model import BaseScreeningModel


class TextRiskModel(BaseScreeningModel):
    """Fine-tuned BanglaBERT sequence classifier for psychiatric risk from Bengali text."""

    def __init__(
        self,
        pretrained_model: str = "sagorsarker/bangla-bert-base",
        num_classes: int = 4,
        dropout: float = 0.3,
        device: str = "cpu",
    ) -> None:
        super().__init__(num_classes=num_classes, device=device)
        self.pretrained_model = pretrained_model
        self.dropout_rate = dropout

        self.encoder: nn.Module
        self.dropout: nn.Dropout
        self.classifier: nn.Linear
        self._build()

    def _build(self) -> None:
        """Load BanglaBERT encoder and attach classification head."""
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass using CLS token for classification.

        Args:
            input_ids: Token IDs tensor of shape (B, L).
            attention_mask: Attention mask tensor of shape (B, L).
            token_type_ids: Optional segment IDs.

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        ...

    def get_cls_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the CLS token embedding without the classification head."""
        ...

    def get_token_embeddings(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return all token embeddings for SHAP/LIME word-level attribution."""
        ...
