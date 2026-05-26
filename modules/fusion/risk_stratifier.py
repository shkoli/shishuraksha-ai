"""Clinical risk stratification from fusion output.

Maps the composite risk score to four clinical tiers and generates
Bangladesh-specific referral actions.

Risk tiers
----------
LOW       [0.00, 0.25)  – routine monitoring
MODERATE  [0.25, 0.50)  – school counsellor referral
HIGH      [0.50, 0.75)  – DSS 1098 within 24 h
CRITICAL  [0.75, 1.00]  – OCC 16767 immediate

Critical override
-----------------
If any single modality scores above SINGLE_MODALITY_OVERRIDE_THRESHOLD (0.85),
the risk level is raised to HIGH at minimum, regardless of the composite score.
This prevents a very strong single signal being diluted by low-risk modalities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

from modules.fusion.attention_fusion import FusionOutput

logger = logging.getLogger(__name__)

# ── Risk thresholds (lower bound inclusive) ───────────────────────────────────
_THRESHOLDS: list[tuple[float, str]] = [
    (0.75, "CRITICAL"),
    (0.50, "HIGH"),
    (0.25, "MODERATE"),
    (0.00, "LOW"),
]

SINGLE_MODALITY_OVERRIDE_THRESHOLD: float = 0.85


# ── Bangladesh referral resources ────────────────────────────────────────────
_REFERRALS: dict[str, dict[str, str]] = {
    "CRITICAL": {
        "en": "OCC Hotline: 16767 (Immediate — call now)",
        "bn": "OCC হটলাইন: ১৬৭৬৭ (তাৎক্ষণিক — এখনই কল করুন)",
        "action": "immediate_psychiatric_evaluation",
        "urgency": "immediate",
    },
    "HIGH": {
        "en": "DSS Hotline: 1098 (Within 24 hours)",
        "bn": "DSS হটলাইন: ১০৯৮ (২৪ ঘণ্টার মধ্যে)",
        "action": "urgent_psychiatric_referral",
        "urgency": "within_24h",
    },
    "MODERATE": {
        "en": "School Counsellor / Community Mental Health (within 2 weeks)",
        "bn": "স্কুল কাউন্সেলর / কমিউনিটি মানসিক স্বাস্থ্য (২ সপ্তাহের মধ্যে)",
        "action": "counselling_referral",
        "urgency": "within_2_weeks",
    },
    "LOW": {
        "en": "Routine monitoring (6-week follow-up)",
        "bn": "রুটিন পর্যবেক্ষণ (৬ সপ্তাহ পর ফলো-আপ)",
        "action": "routine_monitoring",
        "urgency": "routine",
    },
}

# Suggested clinical actions per tier
_ACTIONS: dict[str, list[str]] = {
    "CRITICAL": [
        "immediate_psychiatric_evaluation",
        "guardian_notification",
        "safety_protocol",
        "mandatory_report_children_act_2013",
        "occ_hotline_16767",
    ],
    "HIGH": [
        "urgent_psychiatric_referral",
        "safety_planning",
        "parent_consultation",
        "dss_hotline_1098",
        "1_week_followup",
    ],
    "MODERATE": [
        "school_counsellor_referral",
        "parent_psychoeducation",
        "nmhh_16789",
        "6_week_followup",
    ],
    "LOW": [
        "routine_monitoring",
        "psychoeducation",
        "3_month_followup",
    ],
}

# Bengali level labels
_LEVEL_BN: dict[str, str] = {
    "CRITICAL": "জরুরি",
    "HIGH":     "উচ্চ ঝুঁকি",
    "MODERATE": "মধ্যম ঝুঁকি",
    "LOW":      "নিম্ন ঝুঁকি",
}

# Display labels for human-readable strings
_DISPLAY_LABELS: dict[str, str] = {
    "CRITICAL": "Critical",
    "HIGH":     "High Risk",
    "MODERATE": "Moderate Risk",
    "LOW":      "Low Risk",
}

# Modality agreement threshold (max spread considered agreeing)
_AGREEMENT_THRESHOLD: float = 0.15


# ═══════════════════════════════════════════════════════════════════════════════
# RiskStratifier
# ═══════════════════════════════════════════════════════════════════════════════

class RiskStratifier:
    """Maps FusionOutput → structured risk dict with Bangladesh-specific referral action."""

    def __init__(
        self,
        thresholds:         list[tuple[float, str]] | None = None,
        override_threshold: float = SINGLE_MODALITY_OVERRIDE_THRESHOLD,
    ) -> None:
        self.thresholds         = thresholds or _THRESHOLDS
        self.override_threshold = override_threshold

    # ------------------------------------------------------------------
    def _score_to_level(self, score: float) -> str:
        for lower_bound, level in self.thresholds:
            if score >= lower_bound:
                return level
        return "LOW"

    def _apply_override(
        self,
        level: str,
        per_modality_scores: dict[str, float],
    ) -> tuple[str, list[str]]:
        """Raise level to HIGH minimum if any modality score > override_threshold."""
        override_notes: list[str] = []
        triggered = [
            f"{k}={v:.3f}"
            for k, v in per_modality_scores.items()
            if v > self.override_threshold
        ]
        if triggered:
            note = f"Critical override triggered by: {', '.join(triggered)}"
            override_notes.append(note)
            logger.warning(note)
            tier_rank = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
            if tier_rank.get(level, 0) < tier_rank["HIGH"]:
                level = "HIGH"
        return level, override_notes

    def _confidence_from_output(self, out: FusionOutput) -> float:
        return float(np.clip(out.confidence, 0.0, 1.0))

    def _modality_confidence(
        self,
        per_modality_scores: dict[str, float],
    ) -> tuple[float, str, bool]:
        """Compute confidence from inter-modality agreement via std.

        Formula: confidence = 1 - std(per_modality_scores.values())
        """
        scores = list(per_modality_scores.values())
        std_val = float(np.std(scores)) if len(scores) > 1 else 0.0
        confidence_score = float(np.clip(1.0 - std_val, 0.0, 1.0))

        if confidence_score > 0.80:
            confidence_label = "High"
        elif confidence_score >= 0.60:
            confidence_label = "Medium"
        else:
            confidence_label = "Low"

        modality_agreement = (
            (max(scores) - min(scores)) <= _AGREEMENT_THRESHOLD
            if len(scores) > 1 else True
        )
        return confidence_score, confidence_label, modality_agreement

    # ------------------------------------------------------------------
    def stratify(self, fusion_output: FusionOutput) -> dict[str, Any]:
        score = float(fusion_output.risk_score)
        level = self._score_to_level(score)

        # Critical override: single modality dominance
        level, override_notes = self._apply_override(
            level, fusion_output.per_modality_scores
        )

        ref    = _REFERRALS[level]
        conf   = self._confidence_from_output(fusion_output)

        confidence_score, confidence_label, modality_agreement = \
            self._modality_confidence(fusion_output.per_modality_scores)
        display_string = (
            f"{_DISPLAY_LABELS[level]} "
            f"({int(round(confidence_score * 100))}% confidence)"
        )

        logger.info(
            "Risk stratified: level=%s score=%.3f conf=%.3f modality_conf=%.3f [%s]",
            level, score, conf, confidence_score, confidence_label,
        )

        return {
            "risk_level":          level,
            "risk_level_bn":       _LEVEL_BN[level],
            "risk_score":          score,
            "confidence":          round(conf, 4),
            "confidence_score":    round(confidence_score, 4),
            "confidence_label":    confidence_label,
            "display_string":      display_string,
            "modality_agreement":  modality_agreement,
            "referral":            ref["en"],
            "referral_bn":         ref["bn"],
            "urgency":             ref["urgency"],
            "recommended_actions": _ACTIONS[level],
            "per_modality_scores": dict(fusion_output.per_modality_scores),
            "modality_weights":    dict(fusion_output.modality_weights),
            "override_notes":      override_notes,
        }

    def batch_stratify(
        self,
        outputs: list[FusionOutput],
    ) -> list[dict[str, Any]]:
        return [self.stratify(o) for o in outputs]
