"""Algorithmic fairness and bias audit for the psychiatric screening system.

Evaluates performance disparities across demographic subgroups (age, gender,
division, socioeconomic status) to ensure equitable screening in Bangladesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BiasAuditResult:
    """Bias audit output across a protected attribute."""

    attribute: str
    subgroup_metrics: dict[str, dict[str, float]]
    disparate_impact_ratios: dict[str, float]
    demographic_parity_diff: float
    equalized_odds_diff: float
    max_metric_gap: dict[str, float]
    flagged_disparities: list[str] = field(default_factory=list)


class BiasAuditor:
    """Audits model fairness across demographic subgroups defined in config."""

    PROTECTED_ATTRIBUTES = ["age_group", "gender", "division", "ses"]

    FAIRNESS_THRESHOLDS = {
        "disparate_impact_ratio": 0.80,
        "demographic_parity_diff": 0.10,
        "equalized_odds_diff": 0.10,
    }

    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        self.thresholds = thresholds or self.FAIRNESS_THRESHOLDS

    def audit(
        self,
        predictions_df: pd.DataFrame,
        y_pred_col: str = "predicted",
        y_true_col: str = "label",
        y_proba_col: str = "probability",
    ) -> dict[str, BiasAuditResult]:
        """Run bias audit across all configured protected attributes.

        Args:
            predictions_df: DataFrame with predictions, labels, and demographic columns.
            y_pred_col: Column name for binary/class predictions.
            y_true_col: Column name for ground-truth labels.
            y_proba_col: Column name for predicted probabilities.

        Returns:
            Dict mapping attribute name to BiasAuditResult.
        """
        ...

    def audit_attribute(
        self,
        predictions_df: pd.DataFrame,
        attribute: str,
        y_pred_col: str,
        y_true_col: str,
    ) -> BiasAuditResult:
        """Audit fairness for a single protected attribute."""
        ...

    def _compute_disparate_impact(
        self,
        subgroup_metrics: dict[str, dict[str, float]],
        metric: str = "positive_rate",
    ) -> dict[str, float]:
        """Compute disparate impact ratio relative to the majority subgroup."""
        ...

    def _compute_demographic_parity_diff(
        self,
        subgroup_metrics: dict[str, dict[str, float]],
    ) -> float:
        """Compute max pairwise difference in positive prediction rates."""
        ...

    def _compute_equalized_odds_diff(
        self,
        subgroup_metrics: dict[str, dict[str, float]],
    ) -> float:
        """Compute max gap in TPR and FPR across subgroups."""
        ...

    def generate_report(self, audit_results: dict[str, BiasAuditResult]) -> pd.DataFrame:
        """Summarise all bias audit results into a flat reporting DataFrame."""
        ...
