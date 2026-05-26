"""Session-level emotion pattern classification from AU feature vectors."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from modules.facial_analysis.expression_detector import AU_LABELS

logger = logging.getLogger(__name__)

# AU index positions in the canonical AU_LABELS list
_IDX = {au: i for i, au in enumerate(AU_LABELS)}
# AU_LABELS = ["AU4","AU6","AU7","AU12","AU15","AU17","AU20","AU43"]

# ── Dominant emotion rules (AU index, weight) ─────────────────────────────────
# Each emotion is scored by a weighted sum of its constituent AUs.
_EMOTION_RULES: dict[str, list[tuple[str, float]]] = {
    "fear":     [("AU4", 0.25), ("AU7", 0.25), ("AU20", 0.30), ("AU43", 0.20)],
    "sadness":  [("AU4", 0.30), ("AU15", 0.35), ("AU17", 0.20), ("AU6", 0.15)],
    "anger":    [("AU4", 0.35), ("AU7", 0.35), ("AU12", 0.15), ("AU17", 0.15)],
    "neutral":  [],   # fallback — low overall activation
}

# Risk contribution weights for each session pattern
_RISK_WEIGHTS = {
    "flat_affect":     0.30,
    "freeze":          0.40,
    "hypervigilance":  0.30,
}


# ═══════════════════════════════════════════════════════════════════════════════
# SessionEmotionResult
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SessionEmotionResult:
    dominant_emotion:      str
    flat_affect_score:     float
    freeze_score:          float
    hypervigilance_score:  float
    risk_contribution:     float
    n_frames:              int

    def __repr__(self) -> str:
        return (
            f"SessionEmotionResult(\n"
            f"  dominant_emotion={self.dominant_emotion!r},\n"
            f"  flat_affect_score={self.flat_affect_score:.4f},\n"
            f"  freeze_score={self.freeze_score:.4f},\n"
            f"  hypervigilance_score={self.hypervigilance_score:.4f},\n"
            f"  risk_contribution={self.risk_contribution:.4f},\n"
            f"  n_frames={self.n_frames}\n"
            f")"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dominant_emotion":     self.dominant_emotion,
            "flat_affect_score":    self.flat_affect_score,
            "freeze_score":         self.freeze_score,
            "hypervigilance_score": self.hypervigilance_score,
            "risk_contribution":    self.risk_contribution,
            "n_frames":             self.n_frames,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EmotionSequenceClassifier
# ═══════════════════════════════════════════════════════════════════════════════

class EmotionSequenceClassifier:
    """Rule-based session emotion classifier; accepts AU score dicts or 32-dim feature vectors."""

    def __init__(
        self,
        flat_affect_threshold: float = 0.20,
        freeze_au43_threshold: float = 0.40,
        hypervig_threshold:    float = 0.35,
    ) -> None:
        self.flat_affect_threshold = flat_affect_threshold
        self.freeze_au43_threshold = freeze_au43_threshold
        self.hypervig_threshold    = hypervig_threshold

    # ------------------------------------------------------------------
    def _to_au_matrix(
        self,
        vectors: list[dict[str, float]] | list[np.ndarray],
    ) -> np.ndarray:
        rows: list[np.ndarray] = []
        for v in vectors:
            if isinstance(v, dict):
                row = np.array([v.get(au, 0.0) for au in AU_LABELS], dtype=np.float32)
            else:
                arr = np.asarray(v, dtype=np.float32).flatten()
                row = arr[:8]  # first 8 elements are always raw AUs
            rows.append(row)
        return np.stack(rows) if rows else np.zeros((0, 8), dtype=np.float32)

    # ------------------------------------------------------------------
    def _flat_affect_score(self, au_matrix: np.ndarray) -> float:
        """Low mean activation → high flat affect.  Inverted & normalised."""
        if len(au_matrix) == 0:
            return 0.0
        mean_activation = float(au_matrix.mean())
        # Score rises as activation falls below the threshold
        score = max(0.0, self.flat_affect_threshold - mean_activation) / self.flat_affect_threshold
        return float(np.clip(score, 0.0, 1.0))

    def _freeze_score(self, au_matrix: np.ndarray) -> float:
        """AU43 dominance + low AU12 (no smile) → freeze pattern."""
        if len(au_matrix) == 0:
            return 0.0
        au43_mean = float(au_matrix[:, _IDX["AU43"]].mean())
        au12_mean = float(au_matrix[:, _IDX["AU12"]].mean())
        # AU43 elevated and AU12 suppressed
        au43_component = float(np.clip(au43_mean / max(self.freeze_au43_threshold, 1e-6), 0.0, 1.0))
        au12_suppression = float(np.clip(1.0 - au12_mean * 3.0, 0.0, 1.0))
        return float(np.clip((au43_component * 0.60 + au12_suppression * 0.40), 0.0, 1.0))

    def _hypervigilance_score(self, au_matrix: np.ndarray) -> float:
        """Elevated AU4 + AU7 → sustained brow tension / lid tightening."""
        if len(au_matrix) == 0:
            return 0.0
        au4_mean = float(au_matrix[:, _IDX["AU4"]].mean())
        au7_mean = float(au_matrix[:, _IDX["AU7"]].mean())
        combined = (au4_mean + au7_mean) / 2.0
        score    = float(np.clip(combined / max(self.hypervig_threshold, 1e-6), 0.0, 1.0))
        return score

    def _dominant_emotion(self, au_matrix: np.ndarray) -> str:
        """Score each emotion against session mean AU values."""
        if len(au_matrix) == 0:
            return "neutral"
        mean_aus = au_matrix.mean(axis=0)  # (8,)
        au_dict  = {au: float(mean_aus[i]) for i, au in enumerate(AU_LABELS)}

        scores: dict[str, float] = {}
        for emotion, rules in _EMOTION_RULES.items():
            if not rules:
                scores[emotion] = 0.0
                continue
            scores[emotion] = sum(w * au_dict.get(au, 0.0) for au, w in rules)

        # 'neutral' wins when all named emotions score below a low bar
        max_named = max(scores[e] for e in ("fear", "sadness", "anger"))
        if max_named < 0.15:
            return "neutral"

        return max(("fear", "sadness", "anger"), key=lambda e: scores[e])

    def _risk_contribution(
        self,
        flat: float,
        freeze: float,
        hypervig: float,
    ) -> float:
        score = (
            _RISK_WEIGHTS["flat_affect"]    * flat
            + _RISK_WEIGHTS["freeze"]       * freeze
            + _RISK_WEIGHTS["hypervigilance"] * hypervig
        )
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    def classify(
        self,
        au_vectors: list[dict[str, float]] | list[np.ndarray],
    ) -> SessionEmotionResult:
        if not au_vectors:
            logger.warning("classify() called with empty au_vectors — returning zeros.")
            return SessionEmotionResult(
                dominant_emotion      = "neutral",
                flat_affect_score     = 0.0,
                freeze_score          = 0.0,
                hypervigilance_score  = 0.0,
                risk_contribution     = 0.0,
                n_frames              = 0,
            )

        mat      = self._to_au_matrix(au_vectors)
        flat     = self._flat_affect_score(mat)
        freeze   = self._freeze_score(mat)
        hypervig = self._hypervigilance_score(mat)
        dominant = self._dominant_emotion(mat)
        risk     = self._risk_contribution(flat, freeze, hypervig)

        return SessionEmotionResult(
            dominant_emotion      = dominant,
            flat_affect_score     = round(flat, 4),
            freeze_score          = round(freeze, 4),
            hypervigilance_score  = round(hypervig, 4),
            risk_contribution     = round(risk, 4),
            n_frames              = len(au_vectors),
        )
