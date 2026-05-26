"""CNN-based drawing risk classification model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.base_model import BaseScreeningModel


class DrawingRiskModel(BaseScreeningModel):
    """EfficientNet-based risk classifier for child psychiatric drawings."""

    def __init__(
        self,
        backbone: str = "efficientnet_b3",
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.3,
        device: str = "cpu",
    ) -> None:
        super().__init__(num_classes=num_classes, device=device)
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.dropout_rate = dropout

        self.features: nn.Module
        self.pool: nn.Module
        self.classifier: nn.Module
        self._build()

    def _build(self) -> None:
        """Initialise EfficientNet backbone and attach classification head."""
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning risk logits.

        Args:
            x: Image batch tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        ...

    def get_feature_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (feature_maps, logits) for Grad-CAM visualisation."""
        ...

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled feature embedding before the classifier head."""
        ...
