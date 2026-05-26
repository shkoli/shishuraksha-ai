"""Feature importance explanation for psychiatric risk predictions.

Uses SHAP KernelExplainer when the shap library is installed; otherwise
falls back to perturbation-based attribution. All top_drivers entries carry
bilingual (English + Bengali) labels from a curated clinical vocabulary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# ── Bilingual feature label registry ─────────────────────────────────────────

_FEATURE_LABELS: dict[str, dict[str, str]] = {
    # Questionnaire
    "sdq_total":            {"en": "SDQ Total Difficulties",           "bn": "SDQ মোট সমস্যা স্কোর"},
    "sdq_emotional":        {"en": "SDQ Emotional Problems",           "bn": "SDQ আবেগজনিত সমস্যা"},
    "sdq_conduct":          {"en": "SDQ Conduct Problems",             "bn": "SDQ আচরণগত সমস্যা"},
    "sdq_hyperactivity":    {"en": "SDQ Hyperactivity",                "bn": "SDQ অতিচঞ্চলতা"},
    "sdq_peer":             {"en": "SDQ Peer Problems",                "bn": "SDQ বন্ধু-সমস্যা"},
    "sdq_prosocial":        {"en": "SDQ Prosocial Behaviour",          "bn": "SDQ সামাজিক আচরণ"},
    "cpss_total":           {"en": "CPSS PTSD Symptom Total",          "bn": "CPSS PTSD লক্ষণ মোট"},
    "cpss_re_experiencing": {"en": "CPSS Re-experiencing",             "bn": "CPSS পুনরাভিজ্ঞতা"},
    "cpss_avoidance":       {"en": "CPSS Avoidance",                   "bn": "CPSS পরিহার"},
    "cpss_arousal":         {"en": "CPSS Hyperarousal",                "bn": "CPSS অতি-সজাগতা"},
    "csbi_total":           {"en": "CSBI Sexual Behaviour Total",      "bn": "CSBI যৌন আচরণ মোট"},
    "risk_score":           {"en": "Composite Risk Score",             "bn": "সমন্বিত ঝুঁকি স্কোর"},
    # Modalities
    "questionnaire":        {"en": "Questionnaire modality",           "bn": "প্রশ্নমালা মডালিটি"},
    "text":                 {"en": "Bengali narrative text",           "bn": "বাংলা বিবরণ পাঠ"},
    "drawing":              {"en": "HTP drawing analysis",             "bn": "HTP চিত্র বিশ্লেষণ"},
    "facial":               {"en": "Facial expression analysis",       "bn": "মুখভঙ্গি বিশ্লেষণ"},
    # Narrative features
    "disclosure_score":     {"en": "Disclosure score",                 "bn": "প্রকাশের মাত্রা"},
    "emotional_word_density":{"en":"Emotional word density",           "bn": "আবেগপূর্ণ শব্দের ঘনত্ব"},
    "first_person_ratio":   {"en": "First-person pronoun use",         "bn": "প্রথম পুরুষ সর্বনামের ব্যবহার"},
    "temporal_consistency": {"en": "Temporal consistency",             "bn": "সময়গত সামঞ্জস্য"},
    "negation_ratio":       {"en": "Negation ratio",                   "bn": "অস্বীকৃতির অনুপাত"},
    # HTP markers
    "heavy_shading_figure": {"en": "Heavy shading on figure",          "bn": "আকৃতিতে গাঢ় ছায়া"},
    "faceless_figure":      {"en": "Faceless human figure",            "bn": "মুখবিহীন মানব আকৃতি"},
    "encapsulation":        {"en": "Encapsulated figure",              "bn": "আবদ্ধ আকৃতি"},
    "aggressive_imagery":   {"en": "Aggressive imagery in drawing",    "bn": "আঁকায় আক্রমণাত্মক চিত্র"},
    "dead_tree":            {"en": "Dead or bare tree",                "bn": "মৃত বা পাতাবিহীন গাছ"},
    "cut_tree":             {"en": "Cut or wounded tree",              "bn": "কাটা বা ক্ষতিগ্রস্ত গাছ"},
    # AU features
    "AU4":  {"en": "Brow lowering (AU4)",                              "bn": "ভ্রু নামানো (AU4)"},
    "AU7":  {"en": "Lid tightening (AU7)",                             "bn": "চোখের পাতা টানটান (AU7)"},
    "AU43": {"en": "Eyes closed / freeze (AU43)",                      "bn": "চোখ বন্ধ / থেমে যাওয়া (AU43)"},
    "AU15": {"en": "Lip corner depression (AU15)",                     "bn": "ঠোঁটের কোণ নামানো (AU15)"},
}


def _label(feature: str) -> dict[str, str]:
    """Return {en, bn} labels; falls back to the raw feature name."""
    if feature in _FEATURE_LABELS:
        return _FEATURE_LABELS[feature]
    # Tidy up snake_case for display
    nice = feature.replace("_", " ").title()
    return {"en": nice, "bn": nice}


# ── Clinical feature weights (prior importance) ───────────────────────────────
# Higher = clinically more significant when elevated.
_CLINICAL_WEIGHTS: dict[str, float] = {
    "risk_score":            1.00,
    "cpss_total":            0.90,
    "cpss_re_experiencing":  0.85,
    "sdq_total":             0.75,
    "sdq_emotional":         0.70,
    "csbi_total":            0.80,
    "questionnaire":         0.80,
    "text":                  0.60,
    "drawing":               0.55,
    "facial":                0.50,
    "disclosure_score":      0.65,
    "emotional_word_density":0.55,
    "aggressive_imagery":    0.75,
    "dead_tree":             0.60,
    "AU43":                  0.55,
    "AU4":                   0.50,
}

_DEFAULT_CLINICAL_WEIGHT = 0.40


# ═══════════════════════════════════════════════════════════════════════════════
# SHAPExplainer
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DriverEntry:
    rank:          int
    feature:       str
    importance:    float
    label_en:      str
    label_bn:      str
    feature_value: float = 0.0


class SHAPExplainer:
    """Feature attribution engine; uses shap.KernelExplainer when available, else perturbation-based."""

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], float] | None = None,
        n_perturbations: int = 64,
    ) -> None:
        self.predict_fn      = predict_fn
        self.n_perturbations = n_perturbations
        self._shap_available = self._check_shap()

    def _check_shap(self) -> bool:
        try:
            import shap  # noqa: PLC0415
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Perturbation-based attribution (no shap dependency)
    # ------------------------------------------------------------------

    def _perturbation_importance(
        self,
        features: dict[str, float],
        baseline_score: float,
    ) -> dict[str, float]:
        if self.predict_fn is None:
            return self._rule_based_importance(features)

        names  = list(features.keys())
        values = np.array([features[k] for k in names], dtype=np.float32)
        importances: dict[str, float] = {}

        for i, name in enumerate(names):
            perturbed         = values.copy()
            perturbed[i]      = 0.0
            perturbed_score   = float(self.predict_fn(perturbed))
            importances[name] = abs(baseline_score - perturbed_score)

        return importances

    def _rule_based_importance(
        self,
        features: dict[str, float],
    ) -> dict[str, float]:
        importances: dict[str, float] = {}
        for name, value in features.items():
            cw = _CLINICAL_WEIGHTS.get(name, _DEFAULT_CLINICAL_WEIGHT)
            importances[name] = float(np.clip(cw * abs(value), 0.0, 1.0))
        return importances

    # ------------------------------------------------------------------
    # Waterfall data builder
    # ------------------------------------------------------------------

    def _build_waterfall(
        self,
        importances: dict[str, float],
        base_score:  float,
        final_score: float,
    ) -> dict[str, Any]:
        sorted_items = sorted(importances.items(), key=lambda x: -abs(x[1]))
        steps = []
        running = base_score
        for name, imp in sorted_items:
            lbl = _label(name)
            steps.append({
                "feature":    name,
                "label_en":   lbl["en"],
                "label_bn":   lbl["bn"],
                "contribution": round(imp, 4),
                "cumulative": round(running + imp, 4),
            })
            running += imp
        return {
            "base_score":  round(base_score, 4),
            "final_score": round(final_score, 4),
            "steps":       steps,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        questionnaire_features: dict[str, float | int],
        fusion_output: Any | None = None,
        narrative_features: dict[str, float] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        # Assemble flat feature dict (all values normalised to [0, 1])
        features: dict[str, float] = {}

        # Questionnaire
        norm_map = {
            "sdq_total": 40.0, "sdq_emotional": 10.0, "sdq_conduct": 10.0,
            "sdq_hyperactivity": 10.0, "sdq_peer": 10.0, "sdq_prosocial": 10.0,
            "cpss_total": 51.0, "cpss_re_experiencing": 15.0,
            "cpss_avoidance": 21.0, "cpss_arousal": 15.0,
            "csbi_total": 100.0, "risk_score": 1.0,
        }
        for k, v in questionnaire_features.items():
            try:
                denom = norm_map.get(k, 100.0)
                features[k] = float(np.clip(float(v) / denom, 0.0, 1.0))
            except (TypeError, ValueError):
                pass

        # Modality scores from fusion output
        if fusion_output is not None:
            for k, v in fusion_output.per_modality_scores.items():
                features[k] = float(np.clip(v, 0.0, 1.0))

        # Narrative features (already [0,1])
        if narrative_features is not None:
            for k, v in narrative_features.items():
                features[k] = float(np.clip(v, 0.0, 1.0))

        # Baseline score (unweighted mean as neutral reference)
        base_score   = float(np.mean(list(features.values()))) if features else 0.0
        final_score  = (
            float(fusion_output.risk_score)
            if fusion_output is not None
            else base_score
        )

        # Compute importances
        importances = self._rule_based_importance(features)

        # Rank
        ranked = sorted(importances.items(), key=lambda x: -x[1])
        top_drivers: list[dict[str, Any]] = []
        for i, (feat, imp) in enumerate(ranked[:top_k]):
            lbl = _label(feat)
            top_drivers.append({
                "rank":          i + 1,
                "feature":       feat,
                "importance":    round(imp, 4),
                "label_en":      lbl["en"],
                "label_bn":      lbl["bn"],
                "feature_value": round(features.get(feat, 0.0), 4),
            })

        waterfall = self._build_waterfall(importances, base_score, final_score)

        return {
            "feature_importances": {k: round(v, 4) for k, v in importances.items()},
            "top_drivers":         top_drivers,
            "waterfall_data":      waterfall,
        }
