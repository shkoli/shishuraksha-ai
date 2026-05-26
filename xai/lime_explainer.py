"""LIME-based local explanations for text and tabular screening models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LIMEExplanation:
    """LIME explanation output for one prediction."""

    feature_importances: list[tuple[str, float]]
    predicted_class: int
    predicted_proba: float
    local_model_score: float
    intercept: float


class LIMEExplainer:
    """Generates LIME local explanations for text and tabular psychiatric risk models."""

    def __init__(
        self,
        model: Any,
        mode: str = "tabular",
        num_features: int = 10,
        num_samples: int = 500,
    ) -> None:
        self.model = model
        self.mode = mode
        self.num_features = num_features
        self.num_samples = num_samples
        self._explainer: Any = None

    def fit(self, training_data: np.ndarray, feature_names: list[str], class_names: list[str]) -> None:
        """Initialise the LIME explainer with training data statistics.

        Args:
            training_data: Representative training samples for perturbation.
            feature_names: Ordered feature name list.
            class_names: List of class label strings.
        """
        ...

    def explain_tabular(self, x: np.ndarray) -> LIMEExplanation:
        """Generate a LIME explanation for a tabular feature vector.

        Args:
            x: Feature vector of shape (n_features,).

        Returns:
            LIMEExplanation with ranked feature importances.
        """
        ...

    def explain_text(self, text: str, class_idx: int = 1) -> LIMEExplanation:
        """Generate a LIME explanation for a text input, highlighting influential tokens."""
        ...

    def plot_explanation(self, explanation: LIMEExplanation, output_path: str | None = None) -> Any:
        """Render the LIME explanation as a bar chart."""
        ...
