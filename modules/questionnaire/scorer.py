"""Scoring engine for the BangladeshSDQ, CPSS, and CSBI_BD instruments.

For each instrument, InstrumentScorer:
  1. Applies reverse-scoring where flagged on individual items.
  2. Computes subscale raw totals.
  3. Normalises every subscale and the total to a 0–100 scale.
  4. Classifies severity using instrument-specific cutoffs.
  5. Returns a rich ScoreResult dataclass.

All scoring rules are sourced from:
  SDQ  — Goodman (1997, 2001); Bangladesh community pilot (2022).
  CPSS — Foa, Johnson, Feeny & Treadwell (2001).
  CSBI — Friedrich (1992, 1997); BD adaptation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from modules.questionnaire.instrument import (
    CPSS,
    CSBI_BD,
    BangladeshSDQ,
    BaseInstrument,
    Item,
    InstrumentType,
)


@dataclass
class SubscaleScore:
    name:       str
    raw:        int
    max_raw:    int
    normalised: float
    severity:   str


@dataclass
class ScoreResult:
    """Complete scoring output for one instrument administration."""
    instrument:     InstrumentType
    age:            int
    gender:         str
    raw_total:      int
    max_total:      int
    normalised:     float
    severity:       str
    subscales:      list[SubscaleScore]
    missing_ids:    list[str]   = field(default_factory=list)
    imputed:        bool        = False
    flags:          list[str]   = field(default_factory=list)
    n_items_scored: int         = 0

    def subscale(self, name: str) -> SubscaleScore | None:
        return next((s for s in self.subscales if s.name == name), None)

    def is_clinical(self) -> bool:
        return self.severity not in {"normal", "typical", "no_ptsd"}

    def summary(self) -> str:
        lines = [
            f"{self.instrument.value} Score Summary",
            f"  Age: {self.age}  Gender: {self.gender}",
            f"  Total: {self.raw_total}/{self.max_total}  "
            f"({self.normalised:.1f}/100)  Severity: {self.severity.upper()}",
        ]
        for ss in self.subscales:
            lines.append(
                f"    {ss.name:<22s}: {ss.raw:>3}/{ss.max_raw}  "
                f"({ss.normalised:.0f}/100)  [{ss.severity}]"
            )
        if self.flags:
            lines.append(f"  !! FLAGS: {', '.join(self.flags)}")
        if self.missing_ids:
            lines.append(f"  Missing items ({len(self.missing_ids)}): {self.missing_ids}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument":     self.instrument.value,
            "age":            self.age,
            "gender":         self.gender,
            "raw_total":      self.raw_total,
            "max_total":      self.max_total,
            "normalised":     round(self.normalised, 2),
            "severity":       self.severity,
            "n_items_scored": self.n_items_scored,
            "imputed":        self.imputed,
            "missing_ids":    self.missing_ids,
            "flags":          self.flags,
            "subscales": [
                {
                    "name":       ss.name,
                    "raw":        ss.raw,
                    "max_raw":    ss.max_raw,
                    "normalised": round(ss.normalised, 2),
                    "severity":   ss.severity,
                }
                for ss in self.subscales
            ],
        }


def _normalise(raw: int | float, max_val: int) -> float:
    """Map raw score to 0–100, clamped."""
    if max_val == 0:
        return 0.0
    return max(0.0, min(100.0, raw / max_val * 100.0))


def _classify(score: int, cutoffs: dict[str, tuple[int, int]]) -> str:
    """Return the severity label whose range contains *score*."""
    for label, (lo, hi) in cutoffs.items():
        if lo <= score <= hi:
            return label
    return max(cutoffs, key=lambda k: cutoffs[k][1])


def _prorate(
    items_in_subscale: list[Item],
    responded: dict[str, int],
    max_missing_frac: float = 0.20,
) -> tuple[int, list[str], bool]:
    """Prorate a subscale score when a minority of items are missing.

    Returns (scored_total, missing_ids, was_imputed).
    Raises ValueError if missing fraction exceeds max_missing_frac.
    """
    scored, missing = [], []
    for item in items_in_subscale:
        if item.item_id in responded:
            scored.append(item.score(responded[item.item_id]))
        else:
            missing.append(item.item_id)

    n_total = len(items_in_subscale)
    if n_total == 0:
        return 0, [], False

    if len(missing) / n_total > max_missing_frac:
        raise ValueError(
            f"Too many missing items ({len(missing)}/{n_total}) — "
            f"subscale score cannot be computed."
        )

    if not missing:
        return sum(scored), [], False

    # Scale observed sum up proportionally
    mean_scored = sum(scored) / len(scored) if scored else 0.0
    prorated = round(mean_scored * n_total)
    return prorated, missing, True


_SDQ_SUBSCALE_CUTOFFS: dict[str, dict[str, tuple[int, int]]] = {
    "emotional":     {"normal": (0, 3),  "borderline": (4, 5),  "abnormal": (6, 10)},
    "conduct":       {"normal": (0, 2),  "borderline": (3, 4),  "abnormal": (5, 10)},
    "hyperactivity": {"normal": (0, 5),  "borderline": (6, 7),  "abnormal": (8, 10)},
    "peer":          {"normal": (0, 2),  "borderline": (3, 4),  "abnormal": (5, 10)},
    "prosocial":     {"normal": (6, 10), "borderline": (5, 5),  "abnormal": (0, 4)},
}

_SDQ_FLAG_ITEMS: set[str] = {
    "SDQ_13",   # "often unhappy/depressed/tearful" — depression marker
}


def score_sdq(
    instrument: BangladeshSDQ,
    responses: dict[str, int],
    age:    int,
    gender: str = "unknown",
) -> ScoreResult:
    ok, msg = instrument.validate_age(age)
    if not ok:
        raise ValueError(msg)

    all_missing: list[str] = []
    any_imputed = False
    flags: list[str] = []
    subscale_scores: list[SubscaleScore] = []

    for ss_name, item_ids in instrument._subscale_map.items():
        items_in_ss = [instrument.get_item(iid) for iid in item_ids
                       if instrument.get_item(iid) is not None]
        raw, missing, imputed = _prorate(items_in_ss, responses)
        all_missing.extend(missing)
        if imputed:
            any_imputed = True

        max_raw = instrument.SUBSCALE_MAX[ss_name]
        norm = _normalise(raw, max_raw)
        cutoffs = _SDQ_SUBSCALE_CUTOFFS.get(ss_name, {"normal": (0, max_raw)})
        severity = _classify(raw, cutoffs)

        subscale_scores.append(SubscaleScore(
            name=ss_name, raw=raw, max_raw=max_raw,
            normalised=norm, severity=severity,
        ))

    # Total difficulties = sum of four non-prosocial subscales
    difficulties_subs = {
        ss.name: ss for ss in subscale_scores
        if ss.name != "prosocial"
    }
    total_raw = sum(ss.raw for ss in difficulties_subs.values())
    total_max = instrument.TOTAL_DIFFICULTIES_MAX
    total_norm = _normalise(total_raw, total_max)
    overall_severity = _classify(total_raw, instrument.CUTOFFS)

    if "SDQ_13" in responses and responses["SDQ_13"] == 2:
        flags.append("sdq_depressed_mood_endorsed_certainly_true")
    if difficulties_subs.get("emotional") and difficulties_subs["emotional"].raw >= 7:
        flags.append("sdq_emotional_subscale_elevated")

    n_scored = sum(1 for iid in responses if instrument.get_item(iid) is not None)

    return ScoreResult(
        instrument=InstrumentType.SDQ,
        age=age,
        gender=gender,
        raw_total=total_raw,
        max_total=total_max,
        normalised=total_norm,
        severity=overall_severity,
        subscales=subscale_scores,
        missing_ids=all_missing,
        imputed=any_imputed,
        flags=flags,
        n_items_scored=n_scored,
    )


_CPSS_SUBSCALE_CUTOFFS: dict[str, dict[str, tuple[int, int]]] = {
    "re_experiencing": {
        "no_ptsd": (0, 3), "subclinical": (4, 6),
        "clinical": (7, 10), "severe": (11, 15),
    },
    "avoidance": {
        "no_ptsd": (0, 4), "subclinical": (5, 8),
        "clinical": (9, 14), "severe": (15, 21),
    },
    "arousal": {
        "no_ptsd": (0, 3), "subclinical": (4, 6),
        "clinical": (7, 10), "severe": (11, 15),
    },
}

_CPSS_RE_EXPERIENCING_IDS = {"CPSS_01", "CPSS_02", "CPSS_03", "CPSS_04", "CPSS_05"}
_CPSS_AVOIDANCE_IDS       = {"CPSS_06", "CPSS_07", "CPSS_08", "CPSS_09",
                              "CPSS_10", "CPSS_11", "CPSS_12"}
_CPSS_AROUSAL_IDS         = {"CPSS_13", "CPSS_14", "CPSS_15", "CPSS_16", "CPSS_17"}


def score_cpss(
    instrument: CPSS,
    responses: dict[str, int],
    age:    int,
    gender: str = "unknown",
) -> ScoreResult:
    ok, msg = instrument.validate_age(age)
    if not ok:
        raise ValueError(msg)

    all_missing: list[str] = []
    any_imputed = False
    flags: list[str] = []
    subscale_scores: list[SubscaleScore] = []

    for ss_name, item_ids in instrument._subscale_map.items():
        items_in_ss = [instrument.get_item(iid) for iid in item_ids
                       if instrument.get_item(iid) is not None]
        raw, missing, imputed = _prorate(items_in_ss, responses)
        all_missing.extend(missing)
        if imputed:
            any_imputed = True

        max_raw = instrument.SUBSCALE_MAX[ss_name]
        norm = _normalise(raw, max_raw)
        severity = _classify(raw, _CPSS_SUBSCALE_CUTOFFS[ss_name])

        subscale_scores.append(SubscaleScore(
            name=ss_name, raw=raw, max_raw=max_raw,
            normalised=norm, severity=severity,
        ))

    total_raw = sum(ss.raw for ss in subscale_scores)
    total_max = instrument.TOTAL_MAX
    total_norm = _normalise(total_raw, total_max)
    overall_severity = _classify(total_raw, instrument.CUTOFFS)

    # PTSD diagnostic cluster check: need ≥1 re-experiencing + ≥3 avoidance + ≥2 arousal
    re_count = sum(1 for iid in _CPSS_RE_EXPERIENCING_IDS if responses.get(iid, 0) >= 1)
    av_count = sum(1 for iid in _CPSS_AVOIDANCE_IDS       if responses.get(iid, 0) >= 1)
    ar_count = sum(1 for iid in _CPSS_AROUSAL_IDS         if responses.get(iid, 0) >= 1)
    if re_count >= 1 and av_count >= 3 and ar_count >= 2:
        flags.append("cpss_ptsd_diagnostic_criteria_met")
    if total_raw >= 35:
        flags.append("cpss_severe_ptsd")
    if responses.get("CPSS_12", 0) >= 2:
        flags.append("cpss_foreshortened_future_endorsed")

    n_scored = sum(1 for iid in responses if instrument.get_item(iid) is not None)

    return ScoreResult(
        instrument=InstrumentType.CPSS,
        age=age,
        gender=gender,
        raw_total=total_raw,
        max_total=total_max,
        normalised=total_norm,
        severity=overall_severity,
        subscales=subscale_scores,
        missing_ids=all_missing,
        imputed=any_imputed,
        flags=flags,
        n_items_scored=n_scored,
    )


_CSBI_SUBSCALE_CUTOFFS_CHILD: dict[str, dict[str, tuple[int, int]]] = {
    "self_stimulation":     {"typical": (0, 5),  "elevated": (6, 8),  "clinical": (9, 21)},
    "boundary_problems":    {"typical": (0, 4),  "elevated": (5, 7),  "clinical": (8, 21)},
    "sexual_anxiety":       {"typical": (0, 3),  "elevated": (4, 5),  "clinical": (6, 18)},
    "sexual_interest":      {"typical": (0, 5),  "elevated": (6, 8),  "clinical": (9, 24)},
    "sexual_intrusiveness": {"typical": (0, 3),  "elevated": (4, 5),  "clinical": (6, 15)},
    "sexual_knowledge":     {"typical": (0, 2),  "elevated": (3, 4),  "clinical": (5, 9)},
    "voyeuristic_behavior": {"typical": (0, 1),  "elevated": (2, 3),  "clinical": (4, 6)},
}

_CSBI_SUBSCALE_CUTOFFS_ADOLESCENT: dict[str, dict[str, tuple[int, int]]] = {
    "self_stimulation":     {"typical": (0, 7),  "elevated": (8, 11), "clinical": (12, 21)},
    "boundary_problems":    {"typical": (0, 6),  "elevated": (7, 9),  "clinical": (10, 21)},
    "sexual_anxiety":       {"typical": (0, 4),  "elevated": (5, 7),  "clinical": (8, 18)},
    "sexual_interest":      {"typical": (0, 7),  "elevated": (8, 10), "clinical": (11, 24)},
    "sexual_intrusiveness": {"typical": (0, 4),  "elevated": (5, 6),  "clinical": (7, 15)},
    "sexual_knowledge":     {"typical": (0, 3),  "elevated": (4, 5),  "clinical": (6, 9)},
    "voyeuristic_behavior": {"typical": (0, 2),  "elevated": (3, 3),  "clinical": (4, 6)},
}

_CSBI_CRITICAL_ITEMS: set[str] = {
    "CSBI_05",
    "CSBI_06",
    "CSBI_25",
    "CSBI_30",
    "CSBI_32",
    "CSBI_33",
}


def score_csbi(
    instrument: CSBI_BD,
    responses: dict[str, int],
    age:    int,
    gender: str = "unknown",
) -> ScoreResult:
    """Score a CSBI_BD administration.

    Only items applicable to the child's age are scored; non-applicable items
    in *responses* are silently skipped.
    """
    ok, msg = instrument.validate_age(age)
    if not ok:
        raise ValueError(msg)

    applicable_items = instrument.get_items_for_age(age)
    applicable_ids   = {it.item_id for it in applicable_items}
    filtered_responses = {k: v for k, v in responses.items() if k in applicable_ids}

    cutoffs_by_subscale = (
        _CSBI_SUBSCALE_CUTOFFS_CHILD if age <= 11
        else _CSBI_SUBSCALE_CUTOFFS_ADOLESCENT
    )

    all_missing: list[str] = []
    any_imputed = False
    flags: list[str] = []
    subscale_scores: list[SubscaleScore] = []

    for ss_name, item_ids in instrument._subscale_map.items():
        items_in_ss = [
            instrument.get_item(iid) for iid in item_ids
            if instrument.get_item(iid) is not None
            and instrument.get_item(iid).is_applicable(age)   # type: ignore[union-attr]
        ]
        if not items_in_ss:
            continue

        raw, missing, imputed = _prorate(items_in_ss, filtered_responses)
        all_missing.extend(missing)
        if imputed:
            any_imputed = True

        max_raw = len(items_in_ss) * 3
        norm = _normalise(raw, max_raw)
        severity = _classify(raw, cutoffs_by_subscale.get(ss_name, {"typical": (0, max_raw)}))

        subscale_scores.append(SubscaleScore(
            name=ss_name, raw=raw, max_raw=max_raw,
            normalised=norm, severity=severity,
        ))

    total_raw = sum(ss.raw for ss in subscale_scores)
    total_max = sum(ss.max_raw for ss in subscale_scores)
    total_norm = _normalise(total_raw, total_max)
    overall_cutoffs = (
        instrument.CUTOFFS_CHILD if age <= 11
        else instrument.CUTOFFS_ADOLESCENT
    )
    overall_severity = _classify(total_raw, overall_cutoffs)

    for item_id in _CSBI_CRITICAL_ITEMS:
        if filtered_responses.get(item_id, 0) >= 2:
            item = instrument.get_item(item_id)
            if item and item.is_applicable(age):
                flags.append(f"csbi_critical_item_endorsed:{item_id}")

    if overall_severity == "clinical":
        flags.append("csbi_clinical_range_total")

    return ScoreResult(
        instrument=InstrumentType.CSBI,
        age=age,
        gender=gender,
        raw_total=total_raw,
        max_total=total_max,
        normalised=total_norm,
        severity=overall_severity,
        subscales=subscale_scores,
        missing_ids=all_missing,
        imputed=any_imputed,
        flags=flags,
        n_items_scored=len(filtered_responses),
    )


class InstrumentScorer:
    """Unified scoring facade for all three instruments.

    Usage::

        scorer = InstrumentScorer()
        result = scorer.score(
            instrument_type="SDQ",
            responses={"SDQ_01": 2, "SDQ_02": 1, ...},
            age=12,
            gender="female",
        )
    """

    def __init__(self) -> None:
        self._sdq  = BangladeshSDQ()
        self._cpss = CPSS()
        self._csbi = CSBI_BD()

    def score(
        self,
        instrument_type: str,
        responses: dict[str, int],
        age:    int,
        gender: str = "unknown",
    ) -> ScoreResult:
        key = instrument_type.upper()
        if key == "SDQ":
            return score_sdq(self._sdq, responses, age, gender)
        if key == "CPSS":
            return score_cpss(self._cpss, responses, age, gender)
        if key in {"CSBI", "CSBI_BD"}:
            return score_csbi(self._csbi, responses, age, gender)
        raise ValueError(
            f"Unknown instrument type {instrument_type!r}. "
            f"Valid options: 'SDQ', 'CPSS', 'CSBI'."
        )

    def score_sdq(self, responses: dict[str, int], age: int,
                  gender: str = "unknown") -> ScoreResult:
        return score_sdq(self._sdq, responses, age, gender)

    def score_cpss(self, responses: dict[str, int], age: int,
                   gender: str = "unknown") -> ScoreResult:
        return score_cpss(self._cpss, responses, age, gender)

    def score_csbi(self, responses: dict[str, int], age: int,
                   gender: str = "unknown") -> ScoreResult:
        return score_csbi(self._csbi, responses, age, gender)
