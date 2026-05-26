"""Clinical validation framework for psychiatric screening model evaluation.

Implements criterion validity, inter-rater reliability, and clinician
agreement assessments aligned with psychiatric assessment standards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ClinicalValidationResult:
    """Results of clinical validation against clinician gold-standard labels."""

    cohen_kappa: float
    gwet_ac1: float
    icc: float
    sensitivity_vs_clinician: float
    specificity_vs_clinician: float
    positive_agreement: float
    negative_agreement: float
    mse_continuous: float | None = None
    notes: list[str] = field(default_factory=list)


class ClinicalValidator:
    """Validates model predictions against psychiatrist reference standard."""

    def __init__(self, n_raters: int = 2) -> None:
        self.n_raters = n_raters

    def validate_against_clinician(
        self,
        model_predictions: np.ndarray,
        clinician_labels: np.ndarray,
        class_names: list[str] | None = None,
    ) -> ClinicalValidationResult:
        """Compute agreement statistics between model and clinician ratings.

        Args:
            model_predictions: Predicted risk levels from the model.
            clinician_labels: Reference standard ratings by psychiatrists.
            class_names: Optional class label strings.

        Returns:
            ClinicalValidationResult with all agreement metrics.
        """
        ...

    def cohen_kappa(self, rater1: np.ndarray, rater2: np.ndarray) -> float:
        """Compute Cohen's kappa for inter-rater reliability."""
        ...

    def gwet_ac1(self, rater1: np.ndarray, rater2: np.ndarray) -> float:
        """Compute Gwet's AC1 as a more stable alternative to Cohen's kappa."""
        ...

    def intraclass_correlation(self, ratings_matrix: np.ndarray, icc_type: str = "ICC3k") -> float:
        """Compute intraclass correlation coefficient from a ratings matrix."""
        ...

    def criterion_validity(
        self,
        predictions: np.ndarray,
        gold_standard: np.ndarray,
    ) -> dict[str, float]:
        """Assess criterion validity against a clinical gold standard measure."""
        ...

    def run_validation_suite(self, results_df: pd.DataFrame) -> ClinicalValidationResult:
        """Run all validation analyses from a structured results DataFrame."""
        ...
