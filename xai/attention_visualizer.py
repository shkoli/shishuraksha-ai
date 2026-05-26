"""Attention weight visualisation for the multi-modal fusion model.

Renders per-modality and cross-modal attention weights to support
clinician understanding of which data sources drove a risk prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AttentionVisualization:
    """Visualisation data for one fusion model prediction."""

    modality_weights: dict[str, float]
    cross_modal_matrix: np.ndarray | None
    token_attention: dict[str, list[float]] | None
    figure: Any | None = None


class AttentionVisualizer:
    """Visualises attention weights from the AttentionFusion model."""

    MODALITY_LABELS = {
        "questionnaire": "Questionnaire",
        "text": "Narrative Text",
        "drawing": "Drawing",
        "facial": "Facial Expression",
    }

    def __init__(self) -> None:
        self._style_config: dict[str, Any] = {}

    def visualize_modality_weights(
        self,
        modality_weights: dict[str, float],
        predicted_risk: str,
    ) -> AttentionVisualization:
        """Render a bar chart of per-modality attention weights.

        Args:
            modality_weights: Mapping of modality name to attention scalar.
            predicted_risk: The predicted risk level label.

        Returns:
            AttentionVisualization with rendered figure.
        """
        ...

    def visualize_cross_modal(self, attention_matrix: np.ndarray) -> AttentionVisualization:
        """Render a heatmap of the cross-modal attention weight matrix."""
        ...

    def visualize_token_attention(
        self,
        tokens: list[str],
        attention_weights: list[float],
        language: str = "bn",
    ) -> AttentionVisualization:
        """Highlight high-attention tokens in the Bengali narrative text."""
        ...

    def save(self, viz: AttentionVisualization, output_path: str | Path) -> None:
        """Save the visualisation figure to disk as PNG or SVG."""
        ...

    def to_html(self, viz: AttentionVisualization) -> str:
        """Render the visualisation as an embeddable HTML snippet."""
        ...
