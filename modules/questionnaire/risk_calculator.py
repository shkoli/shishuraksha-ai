"""Composite risk calculator for multi-instrument psychiatric screening.

Combines SDQ, CPSS, and CSBI_BD scores into a single composite risk score,
classifies the child into a risk level, and generates referral recommendations
aligned with Bangladesh's child protection and mental health system.

Weights (sum = 1.0):
  SDQ  = 0.35   — broad psychopathology screen
  CPSS = 0.40   — trauma/PTSD severity (primary indicator of abuse impact)
  CSBI = 0.25   — sexualized behaviour (specific abuse indicator)

Risk levels:
  low      : composite 0–24   → routine monitoring
  moderate : composite 25–49  → school counsellor / community mental health
  high     : composite 50–74  → DSS (National Child Helpline 1098)
  critical : composite 75–100 → OCC (One-Stop Crisis Centre hotline 16767)
             + mandatory override on specific critical flags
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

from modules.questionnaire.scorer import ScoreResult, InstrumentScorer
from modules.questionnaire.instrument import InstrumentType


class RiskLevel(IntEnum):
    LOW      = 0
    MODERATE = 1
    HIGH     = 2
    CRITICAL = 3

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def label_bn(self) -> str:
        _bn = {
            "LOW":      "নিম্ন ঝুঁকি",
            "MODERATE": "মাঝারি ঝুঁকি",
            "HIGH":     "উচ্চ ঝুঁকি",
            "CRITICAL": "জরুরি ঝুঁকি",
        }
        return _bn[self.name]


@dataclass(frozen=True)
class ReferralResource:
    name:      str
    name_bn:   str
    hotline:   str
    level:     str
    urgency:   str
    notes_en:  str = ""
    notes_bn:  str = ""


_REFERRALS: dict[str, ReferralResource] = {
    "occ": ReferralResource(
        name     = "One-Stop Crisis Centre (OCC)",
        name_bn  = "ওয়ান-স্টপ ক্রাইসিস সেন্টার",
        hotline  = "16767",
        level    = "national",
        urgency  = "immediate",
        notes_en = ("Available 24/7 at district-level hospitals. "
                    "Provides medical, legal, and psychological support for abuse victims."),
        notes_bn = ("জেলা হাসপাতালে ২৪/৭ উপলব্ধ। "
                    "নির্যাতনের শিকারদের চিকিৎসা, আইনি ও মনোসামাজিক সহায়তা প্রদান করে।"),
    ),
    "dss": ReferralResource(
        name     = "Department of Social Services — National Child Helpline",
        name_bn  = "সমাজসেবা অধিদপ্তর — জাতীয় শিশু হেল্পলাইন",
        hotline  = "1098",
        level    = "national",
        urgency  = "within_24h",
        notes_en = "Free 24/7 child protection helpline. Connects to local child welfare officers.",
        notes_bn = ("বিনামূল্যে ২৪/৭ শিশু সুরক্ষা হেল্পলাইন। "
                    "স্থানীয় শিশু কল্যাণ কর্মকর্তার সাথে সংযুক্ত করে।"),
    ),
    "nmhh": ReferralResource(
        name     = "National Mental Health Helpline",
        name_bn  = "জাতীয় মানসিক স্বাস্থ্য হেল্পলাইন",
        hotline  = "16789",
        level    = "national",
        urgency  = "within_24h",
        notes_en = "Psychiatric consultation and crisis support.",
        notes_bn = "মনোচিকিৎসা পরামর্শ ও সংকট সহায়তা।",
    ),
    "school_counsellor": ReferralResource(
        name     = "School Counsellor / Upazila Mental Health Officer",
        name_bn  = "স্কুল কাউন্সেলর / উপজেলা মানসিক স্বাস্থ্য কর্মকর্তা",
        hotline  = "N/A",
        level    = "district",
        urgency  = "routine",
        notes_en = "Schedule within 1–2 weeks. Coordinate with class teacher.",
        notes_bn = "১–২ সপ্তাহের মধ্যে নির্ধারণ করুন। শ্রেণী শিক্ষকের সাথে সমন্বয় করুন।",
    ),
    "community_mh": ReferralResource(
        name     = "Community Mental Health Unit / Upazila Health Complex",
        name_bn  = "কমিউনিটি মানসিক স্বাস্থ্য ইউনিট / উপজেলা স্বাস্থ্য কমপ্লেক্স",
        hotline  = "N/A",
        level    = "district",
        urgency  = "routine",
        notes_en = "Refer for psychiatric assessment within 2 weeks.",
        notes_bn = "২ সপ্তাহের মধ্যে মনোচিকিৎসা মূল্যায়নের জন্য রেফার করুন।",
    ),
}


@dataclass
class ClinicalFlag:
    code:        str
    description: str
    source:      str
    severity:    str


@dataclass
class RiskOutput:
    """Complete risk assessment output for one child screening session."""
    risk_level:              RiskLevel
    risk_score:              float
    confidence:              float
    sdq_score:               float | None
    cpss_score:              float | None
    csbi_score:              float | None
    instruments_available:   list[str]
    clinical_flags:          list[ClinicalFlag]
    referral_recommendation: ReferralResource
    additional_referrals:    list[ReferralResource]
    recommended_actions:     list[str]
    mandatory_report:        bool
    urgency_flag:            bool
    age:                     int
    gender:                  str

    @property
    def risk_label(self) -> str:
        return self.risk_level.label

    @property
    def risk_label_bn(self) -> str:
        return self.risk_level.label_bn

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  PSYCHIATRIC RISK ASSESSMENT SUMMARY",
            "=" * 60,
            f"  Age: {self.age}  |  Gender: {self.gender}",
            f"  Risk Level  : {self.risk_label.upper()} ({self.risk_label_bn})",
            f"  Risk Score  : {self.risk_score:.1f}/100",
            f"  Confidence  : {self.confidence:.0%}",
            "",
            "  Modality scores (normalised 0-100):",
        ]
        for inst, score in [("SDQ", self.sdq_score),
                             ("CPSS", self.cpss_score),
                             ("CSBI", self.csbi_score)]:
            if score is not None:
                lines.append(f"    {inst:<6}: {score:5.1f}")
            else:
                lines.append(f"    {inst:<6}: [not administered]")

        if self.clinical_flags:
            lines += ["", "  Clinical Flags:"]
            for f in self.clinical_flags:
                marker = "!!" if f.severity == "critical" else " *"
                lines.append(f"    {marker} [{f.source}] {f.description}")

        lines += [
            "",
            f"  Primary Referral : {self.referral_recommendation.name}",
            f"                     Hotline: {self.referral_recommendation.hotline}",
            f"                     Urgency: {self.referral_recommendation.urgency}",
        ]
        if self.additional_referrals:
            lines.append("  Additional       :")
            for r in self.additional_referrals:
                lines.append(f"    - {r.name} ({r.hotline})")

        lines += ["", "  Recommended Actions:"]
        for action in self.recommended_actions:
            lines.append(f"    • {action}")

        if self.mandatory_report:
            lines += ["", "  !! MANDATORY CHILD PROTECTION REPORT REQUIRED (Children Act 2013)"]
        if self.urgency_flag:
            lines += ["  !! SAME-DAY ACTION REQUIRED"]

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level":   self.risk_label,
            "risk_level_bn": self.risk_label_bn,
            "risk_score":   round(self.risk_score, 2),
            "confidence":   round(self.confidence, 3),
            "modality_scores": {
                "sdq":  round(self.sdq_score,  2) if self.sdq_score  is not None else None,
                "cpss": round(self.cpss_score, 2) if self.cpss_score is not None else None,
                "csbi": round(self.csbi_score, 2) if self.csbi_score is not None else None,
            },
            "instruments_available":   self.instruments_available,
            "mandatory_report":        self.mandatory_report,
            "urgency_flag":            self.urgency_flag,
            "referral": {
                "primary": {
                    "name":    self.referral_recommendation.name,
                    "hotline": self.referral_recommendation.hotline,
                    "urgency": self.referral_recommendation.urgency,
                },
                "additional": [
                    {"name": r.name, "hotline": r.hotline}
                    for r in self.additional_referrals
                ],
            },
            "clinical_flags": [
                {"code": f.code, "description": f.description,
                 "source": f.source, "severity": f.severity}
                for f in self.clinical_flags
            ],
            "recommended_actions": self.recommended_actions,
        }


_RISK_THRESHOLDS: list[tuple[float, RiskLevel]] = [
    (75.0, RiskLevel.CRITICAL),
    (50.0, RiskLevel.HIGH),
    (25.0, RiskLevel.MODERATE),
    (0.0,  RiskLevel.LOW),
]

_INSTRUMENT_WEIGHTS: dict[str, float] = {
    "SDQ":  0.35,
    "CPSS": 0.40,
    "CSBI": 0.25,
}

# Flags that force risk level to CRITICAL regardless of composite score
_CRITICAL_OVERRIDE_FLAGS: set[str] = {
    "cpss_severe_ptsd",
    "cpss_ptsd_diagnostic_criteria_met",
    "csbi_critical_item_endorsed:CSBI_25",
    "csbi_critical_item_endorsed:CSBI_32",
    "csbi_critical_item_endorsed:CSBI_33",
    "sdq_depressed_mood_endorsed_certainly_true",
}

_MANDATORY_REPORT_FLAGS: set[str] = {
    "csbi_critical_range_total",
    "csbi_critical_item_endorsed:CSBI_25",
    "csbi_critical_item_endorsed:CSBI_32",
    "csbi_critical_item_endorsed:CSBI_33",
    "csbi_critical_item_endorsed:CSBI_06",
    "cpss_ptsd_diagnostic_criteria_met",
    "cpss_severe_ptsd",
}

_REFERRAL_BY_LEVEL: dict[RiskLevel, str] = {
    RiskLevel.CRITICAL: "occ",
    RiskLevel.HIGH:     "dss",
    RiskLevel.MODERATE: "school_counsellor",
    RiskLevel.LOW:      "school_counsellor",
}

_ADDITIONAL_REFERRALS_BY_LEVEL: dict[RiskLevel, list[str]] = {
    RiskLevel.CRITICAL: ["dss", "nmhh"],
    RiskLevel.HIGH:     ["nmhh", "community_mh"],
    RiskLevel.MODERATE: ["community_mh"],
    RiskLevel.LOW:      [],
}

_ACTIONS_BY_LEVEL: dict[RiskLevel, list[str]] = {
    RiskLevel.CRITICAL: [
        "Activate child protection protocol immediately",
        "Call OCC hotline 16767 — same-day referral",
        "Notify DSS and law enforcement under Children Act 2013",
        "Ensure safe placement assessment before child leaves",
        "Document all disclosures verbatim — do not interrogate",
        "Inform school authority and guardian separately",
        "Follow-up within 24 hours",
    ],
    RiskLevel.HIGH: [
        "Call DSS Child Helpline 1098 within 24 hours",
        "Refer to Upazila Mental Health Officer or psychiatrist",
        "Complete safety planning with child and caregiver",
        "Notify guardian — explain risk findings in simple language",
        "Conduct follow-up assessment within 1 week",
        "Inform school authority with child's permission where appropriate",
    ],
    RiskLevel.MODERATE: [
        "Refer to school counsellor or community mental health unit",
        "Schedule psychoeducation session with parent/guardian",
        "Coordinate with class teacher regarding school performance",
        "Review in 6 weeks or earlier if symptoms worsen",
        "Provide child with NMHH helpline number (16789)",
    ],
    RiskLevel.LOW: [
        "Provide psychoeducation leaflet to parent/guardian",
        "Advise routine monitoring by school teacher",
        "Schedule re-screening in 3 months",
        "Encourage involvement in school-based wellness activities",
    ],
}


class RiskCalculator:
    """Combines SDQ, CPSS, and CSBI scores into a composite risk assessment.

    Supports partial administration: if one or more instruments are missing,
    weights are re-normalised across available instruments and confidence is
    penalised proportionally.

    Usage::

        calc   = RiskCalculator()
        output = calc.calculate(
            sdq_result  = score_sdq(...),
            cpss_result = score_cpss(...),
            csbi_result = score_csbi(...),
            age=12, gender="female",
        )
        print(output.summary())
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        critical_override_flags: set[str] | None = None,
    ) -> None:
        self.weights = weights or dict(_INSTRUMENT_WEIGHTS)
        self.critical_flags = critical_override_flags or set(_CRITICAL_OVERRIDE_FLAGS)

        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

    def calculate(
        self,
        age:    int,
        gender: str = "unknown",
        sdq_result:  ScoreResult | None = None,
        cpss_result: ScoreResult | None = None,
        csbi_result: ScoreResult | None = None,
    ) -> RiskOutput:
        """Compute composite risk from available instrument results.

        At least one instrument result must be provided.
        """
        available: dict[str, ScoreResult] = {}
        if sdq_result  is not None: available["SDQ"]  = sdq_result
        if cpss_result is not None: available["CPSS"] = cpss_result
        if csbi_result is not None: available["CSBI"] = csbi_result

        if not available:
            raise ValueError(
                "At least one instrument result (SDQ, CPSS, or CSBI) must be provided."
            )

        composite, confidence = self._compute_composite(available)

        all_raw_flags: set[str] = set()
        for result in available.values():
            all_raw_flags.update(result.flags)

        clinical_flags = self._build_clinical_flags(all_raw_flags, available)
        risk_level = self._classify_risk(composite, all_raw_flags)
        mandatory_report = bool(all_raw_flags & _MANDATORY_REPORT_FLAGS)

        primary_ref_key  = _REFERRAL_BY_LEVEL[risk_level]
        additional_keys  = _ADDITIONAL_REFERRALS_BY_LEVEL[risk_level]
        primary_ref      = _REFERRALS[primary_ref_key]
        additional_refs  = [_REFERRALS[k] for k in additional_keys if k in _REFERRALS]
        actions          = list(_ACTIONS_BY_LEVEL[risk_level])

        if mandatory_report and "mandatory_report_action" not in "".join(actions):
            actions.insert(0, "File mandatory child protection report under Children Act 2013")

        urgency = (
            risk_level >= RiskLevel.HIGH
            or bool(all_raw_flags & {"cpss_severe_ptsd", "csbi_critical_range_total"})
        )

        return RiskOutput(
            risk_level              = risk_level,
            risk_score              = round(composite, 2),
            confidence              = round(confidence, 3),
            sdq_score               = sdq_result.normalised  if sdq_result  else None,
            cpss_score              = cpss_result.normalised if cpss_result else None,
            csbi_score              = csbi_result.normalised if csbi_result else None,
            instruments_available   = list(available.keys()),
            clinical_flags          = clinical_flags,
            referral_recommendation = primary_ref,
            additional_referrals    = additional_refs,
            recommended_actions     = actions,
            mandatory_report        = mandatory_report,
            urgency_flag            = urgency,
            age                     = age,
            gender                  = gender,
        )

    def _compute_composite(
        self,
        available: dict[str, ScoreResult],
    ) -> tuple[float, float]:
        """Return (composite_score_0_100, confidence_0_1).

        When instruments are missing, weights are re-normalised and confidence
        is penalised by the fraction of total weight that is absent.
        """
        total_weight = sum(self.weights.get(k, 0.0) for k in available)
        if total_weight == 0:
            return 0.0, 0.0

        composite = sum(
            result.normalised * self.weights.get(key, 0.0)
            for key, result in available.items()
        ) / total_weight

        max_possible_weight = sum(self.weights.values())
        confidence = total_weight / max_possible_weight

        return composite, confidence

    def _classify_risk(self, composite: float, raw_flags: set[str]) -> RiskLevel:
        if raw_flags & self.critical_flags:
            return RiskLevel.CRITICAL

        for threshold, level in _RISK_THRESHOLDS:
            if composite >= threshold:
                return level
        return RiskLevel.LOW

    def _build_clinical_flags(
        self,
        raw_flags: set[str],
        available: dict[str, ScoreResult],
    ) -> list[ClinicalFlag]:
        source_map = {
            "sdq_":  "SDQ",
            "cpss_": "CPSS",
            "csbi_": "CSBI",
        }

        flag_descriptions: dict[str, tuple[str, str]] = {
            "sdq_depressed_mood_endorsed_certainly_true": (
                "Child endorsed 'certainly true' on depressed mood item (SDQ_13).",
                "warning",
            ),
            "sdq_emotional_subscale_elevated": (
                "SDQ emotional subscale score ≥ 7 — severe emotional distress.",
                "warning",
            ),
            "cpss_ptsd_diagnostic_criteria_met": (
                "CPSS responses meet DSM diagnostic cluster criteria for PTSD.",
                "critical",
            ),
            "cpss_severe_ptsd": (
                "CPSS total ≥ 35 — severe PTSD symptom range.",
                "critical",
            ),
            "cpss_foreshortened_future_endorsed": (
                "Child reports feeling future plans will not come true (CPSS_12 ≥ 2).",
                "warning",
            ),
            "csbi_clinical_range_total": (
                "CSBI total score in clinical range — sexualized behaviour warrants investigation.",
                "critical",
            ),
        }

        csbi_item_descriptions: dict[str, str] = {
            "CSBI_05": "Masturbation with hand",
            "CSBI_06": "Inserting objects into private parts",
            "CSBI_25": "Attempting sexual acts with other children",
            "CSBI_30": "Tongue kissing with others",
            "CSBI_32": "Simulating intercourse with another child",
            "CSBI_33": "Making sexual propositions to younger children",
        }

        flags: list[ClinicalFlag] = []
        for raw in sorted(raw_flags):
            if raw.startswith("csbi_critical_item_endorsed:"):
                item_id = raw.split(":", 1)[1]
                desc = csbi_item_descriptions.get(
                    item_id,
                    f"Critical CSBI item {item_id} endorsed (frequency ≥ 2).",
                )
                flags.append(ClinicalFlag(
                    code=raw,
                    description=desc,
                    source="CSBI",
                    severity="critical",
                ))
                continue

            if raw in flag_descriptions:
                desc, sev = flag_descriptions[raw]
                source = next(
                    (v for k, v in source_map.items() if raw.startswith(k)),
                    "unknown",
                )
                flags.append(ClinicalFlag(
                    code=raw, description=desc, source=source, severity=sev,
                ))
            else:
                source = next(
                    (v for k, v in source_map.items() if raw.startswith(k)),
                    "unknown",
                )
                flags.append(ClinicalFlag(
                    code=raw,
                    description=f"Clinical alert: {raw.replace('_', ' ')}.",
                    source=source,
                    severity="warning",
                ))

        return flags

    def score_and_calculate(
        self,
        age:    int,
        gender: str,
        sdq_responses:  dict[str, int] | None = None,
        cpss_responses: dict[str, int] | None = None,
        csbi_responses: dict[str, int] | None = None,
    ) -> RiskOutput:
        """Score all provided instruments then calculate risk."""
        scorer = InstrumentScorer()
        sdq_result  = None
        cpss_result = None
        csbi_result = None

        if sdq_responses:
            sdq_result = scorer.score_sdq(sdq_responses, age, gender)
        if cpss_responses:
            cpss_result = scorer.score_cpss(cpss_responses, age, gender)
        if csbi_responses:
            csbi_result = scorer.score_csbi(csbi_responses, age, gender)

        return self.calculate(
            age=age, gender=gender,
            sdq_result=sdq_result,
            cpss_result=cpss_result,
            csbi_result=csbi_result,
        )
