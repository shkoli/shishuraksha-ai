"""Structured bilingual clinical report generation — JSON output for clinicians."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Mandatory clinical disclaimer ─────────────────────────────────────────────
DISCLAIMER_BN = (
    "এটি একটি স্ক্রিনিং টুল — রোগ নির্ণয় নয়। "
    "চূড়ান্ত সিদ্ধান্তের জন্য প্রশিক্ষিত ক্লিনিশিয়ানের পরামর্শ নিন।"
)
DISCLAIMER_EN = (
    "This is a screening tool — not a diagnosis. "
    "Consult a trained clinician for final clinical decisions."
)

# ── Modality display names ────────────────────────────────────────────────────
_MODALITY_DISPLAY: dict[str, dict[str, str]] = {
    "questionnaire": {"en": "Questionnaire (SDQ/CPSS/CSBI)", "bn": "প্রশ্নমালা (SDQ/CPSS/CSBI)"},
    "text":          {"en": "Bengali Narrative Text",          "bn": "বাংলা বিবরণ পাঠ"},
    "drawing":       {"en": "HTP Drawing Analysis",            "bn": "HTP চিত্র বিশ্লেষণ"},
    "facial":        {"en": "Facial Expression Analysis",      "bn": "মুখভঙ্গি বিশ্লেষণ"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# ClinicalReportGenerator
# ═══════════════════════════════════════════════════════════════════════════════

class ClinicalReportGenerator:
    """Assembles modality results, risk stratification, and SHAP explanations into bilingual JSON reports."""

    def __init__(
        self,
        output_dir: str | Path = "outputs/reports",
        language:   str        = "bn",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.language   = language

    # ------------------------------------------------------------------
    def _modality_summary(
        self,
        per_modality_scores: dict[str, float],
        modality_weights:    dict[str, float],
    ) -> list[dict[str, Any]]:
        rows = []
        for key, score in per_modality_scores.items():
            disp = _MODALITY_DISPLAY.get(key, {"en": key, "bn": key})
            rows.append({
                "modality":      key,
                "label_en":      disp["en"],
                "label_bn":      disp["bn"],
                "risk_score":    round(score, 4),
                "weight":        round(modality_weights.get(key, 0.0), 4),
                "contribution":  round(score * modality_weights.get(key, 0.0), 4),
            })
        rows.sort(key=lambda r: -r["contribution"])
        return rows

    def _timestamp(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def _date_str(self) -> str:
        return date.today().isoformat()

    # ------------------------------------------------------------------
    def generate(
        self,
        fusion_output:          Any,
        risk_output:            dict[str, Any],
        shap_output:            dict[str, Any] | None = None,
        questionnaire_result:   Any | None = None,
        narrative_result:       Any | None = None,
        drawing_result:         Any | None = None,
        facial_result:          Any | None = None,
        case_id:                str | None = None,
        child_age:              int | None = None,
        child_gender:           str | None = None,
        clinician_notes:        str        = "",
    ) -> dict[str, Any]:
        cid = case_id or f"CASE-{uuid.uuid4().hex[:8].upper()}"

        # Top risk drivers from SHAP (or empty)
        top_drivers = (shap_output or {}).get("top_drivers", [])

        # Per-modality summary
        mod_summary = self._modality_summary(
            risk_output.get("per_modality_scores", {}),
            risk_output.get("modality_weights", {}),
        )

        report: dict[str, Any] = {
            # ── Identifiers ──────────────────────────────────────────
            "case_id":        cid,
            "generated_at":   self._timestamp(),
            "screening_date": self._date_str(),
            # ── Demographics (pseudonymised) ─────────────────────────
            "child_age":      child_age,
            "child_gender":   child_gender,
            # ── Risk summary ─────────────────────────────────────────
            "risk_level":     risk_output.get("risk_level",    "UNKNOWN"),
            "risk_level_bn":  risk_output.get("risk_level_bn", "অজানা"),
            "risk_score":     round(float(risk_output.get("risk_score", 0.0)), 4),
            "confidence":     round(float(risk_output.get("confidence", 0.0)), 4),
            # ── Referral ─────────────────────────────────────────────
            "referral_action":    risk_output.get("action",      "routine_monitoring"),
            "referral_en":        risk_output.get("referral",    ""),
            "referral_bn":        risk_output.get("referral_bn", ""),
            "urgency":            risk_output.get("urgency",     "routine"),
            "recommended_actions":risk_output.get("recommended_actions", []),
            # ── XAI ──────────────────────────────────────────────────
            "top_risk_drivers":   top_drivers,           # list of {rank, feature, importance, label_en, label_bn}
            "waterfall_data":     (shap_output or {}).get("waterfall_data", {}),
            "feature_importances":(shap_output or {}).get("feature_importances", {}),
            # ── Modality breakdown ───────────────────────────────────
            "per_modality_summary": mod_summary,
            # ── Override / clinical flags ─────────────────────────────
            "override_notes": risk_output.get("override_notes", []),
            # ── Clinician ────────────────────────────────────────────
            "clinician_notes": clinician_notes,
            # ── Legal / ethical disclaimer ───────────────────────────
            "disclaimer_bn": DISCLAIMER_BN,
            "disclaimer_en": DISCLAIMER_EN,
        }

        # Attach per-modality detail if results provided
        for label, result in [
            ("questionnaire_detail", questionnaire_result),
            ("narrative_detail",     narrative_result),
            ("drawing_detail",       drawing_result),
            ("facial_detail",        facial_result),
        ]:
            if result is None:
                report[label] = None
            elif hasattr(result, "to_dict"):
                report[label] = result.to_dict()
            elif hasattr(result, "__dict__"):
                report[label] = {
                    k: v for k, v in result.__dict__.items()
                    if not k.startswith("_")
                    and not callable(v)
                    and not hasattr(v, "__len__") or isinstance(v, (str, list, dict, int, float, bool, type(None)))
                }
            else:
                report[label] = str(result)

        return report

    # ------------------------------------------------------------------
    def save_json(
        self,
        report: dict[str, Any],
        filename: str | None = None,
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = filename or f"report_{report.get('case_id', 'unknown')}_{report.get('screening_date', 'nodate')}"
        if not stem.endswith(".json"):
            stem += ".json"
        path = self.output_dir / stem

        # Custom encoder to handle numpy types
        class _NpEncoder(json.JSONEncoder):
            def default(self, obj: Any) -> Any:
                if hasattr(obj, "tolist"):      # numpy arrays
                    return obj.tolist()
                if hasattr(obj, "item"):        # numpy scalars
                    return obj.item()
                return super().default(obj)

        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, cls=_NpEncoder)

        logger.info("Clinical report saved: %s", path)
        return path

    # ------------------------------------------------------------------
    def generate_and_save(self, **kwargs: Any) -> tuple[dict[str, Any], Path]:
        """generate() + save_json() in one call; returns (report_dict, json_path)."""
        report = self.generate(**kwargs)
        path   = self.save_json(report)
        return report, path


# Backward-compatibility alias (old stub name)
XAIReportGenerator = ClinicalReportGenerator
