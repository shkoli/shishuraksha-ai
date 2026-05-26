"""Grad-CAM visualisation for the drawing risk classification model.

Highlights image regions that most influenced the model's risk prediction,
enabling clinician review of drawing-based psychiatric assessments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass
class GradCAMResult:
    """Grad-CAM output for one drawing."""

    image_path: Path | None
    heatmap: np.ndarray
    overlaid_image: np.ndarray
    predicted_class: int
    predicted_proba: float
    target_layer: str


class GradCAM:
    """Gradient-weighted Class Activation Mapping for the DrawingRiskModel."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: str = "features.8",
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self._hooks: list[Any] = []
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None

    def _register_hooks(self) -> None:
        """Attach forward and backward hooks to the target convolutional layer."""
        ...

    def _remove_hooks(self) -> None:
        """Detach all registered hooks after explanation is complete."""
        ...

    def generate(self, image_tensor: torch.Tensor, target_class: int | None = None) -> GradCAMResult:
        """Compute Grad-CAM heatmap for one image.

        Args:
            image_tensor: Preprocessed image tensor of shape (1, C, H, W).
            target_class: Class index to explain; defaults to predicted class.

        Returns:
            GradCAMResult with heatmap and overlay.
        """
        ...

    def _compute_heatmap(self) -> np.ndarray:
        """Weight activation maps by gradient magnitudes and apply ReLU."""
        ...

    def _overlay_heatmap(self, original_image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """Blend the Grad-CAM heatmap with the original drawing image."""
        ...

    def batch_generate(
        self,
        image_tensors: list[torch.Tensor],
        target_classes: list[int] | None = None,
    ) -> list[GradCAMResult]:
        """Generate Grad-CAM explanations for a batch of images."""
        ...

    def save(self, result: GradCAMResult, output_path: str | Path) -> None:
        """Save the heatmap overlay to disk."""
        ...
