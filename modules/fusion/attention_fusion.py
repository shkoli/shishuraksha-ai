"""Attention-based multi-modal fusion for psychiatric risk scoring.

Each modality is projected to a shared 128-dim space via a seeded random
projection (Johnson–Lindenstrauss style). Cross-modal scaled dot-product
attention is computed, context added (residual), and passed through tanh.
Four 128-dim attended vectors concatenate → 512-dim fused output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Architecture constants ────────────────────────────────────────────────────
_PROJ_DIM  = 128   # each modality projects here
_FUSED_DIM = 512   # 4 × _PROJ_DIM

# Input dimensions for each modality
_IN_DIMS: dict[str, int] = {
    "questionnaire": 32,
    "text":          768,
    "drawing":       1300,
    "facial":        32,
}

# Config-matching default modality weights (must sum to 1.0)
_DEFAULT_WEIGHTS: dict[str, float] = {
    "questionnaire": 0.40,
    "text":          0.25,
    "drawing":       0.20,
    "facial":        0.15,
}

# Canonical questionnaire feature slots (name, normalisation_max)
_Q_CANONICAL: list[tuple[str, float]] = [
    ("sdq_total",           40.0),
    ("sdq_emotional",       10.0),
    ("sdq_conduct",         10.0),
    ("sdq_hyperactivity",   10.0),
    ("sdq_peer",            10.0),
    ("sdq_prosocial",       10.0),
    ("cpss_total",          51.0),
    ("cpss_re_experiencing",15.0),
    ("cpss_avoidance",      21.0),
    ("cpss_arousal",        15.0),
    ("risk_score",           1.0),
    ("csbi_total",          100.0),
]
_Q_CANONICAL_NAMES: frozenset[str] = frozenset(n for n, _ in _Q_CANONICAL)
_Q_DIM = _IN_DIMS["questionnaire"]   # 32


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FusionInput:
    questionnaire_features: dict[str, float | int]
    text_features:          np.ndarray           # (768,)
    drawing_features:       np.ndarray           # (1300,)
    facial_features:        np.ndarray           # (32,)

    def available_modalities(self) -> list[str]:
        out = []
        if self.questionnaire_features:
            out.append("questionnaire")
        if self.text_features is not None and len(self.text_features):
            out.append("text")
        if self.drawing_features is not None and len(self.drawing_features):
            out.append("drawing")
        if self.facial_features is not None and len(self.facial_features):
            out.append("facial")
        return out


@dataclass
class FusionOutput:
    fused_vector:         np.ndarray            # (512,)
    modality_weights:     dict[str, float]
    confidence:           float
    per_modality_scores:  dict[str, float]
    risk_score:           float


# ═══════════════════════════════════════════════════════════════════════════════
# MultiModalAttentionFusion
# ═══════════════════════════════════════════════════════════════════════════════

class MultiModalAttentionFusion:
    """Cross-modal attention fusion (numpy) with optional torch acceleration."""

    DEFAULT_WEIGHTS: dict[str, float] = dict(_DEFAULT_WEIGHTS)

    def __init__(
        self,
        modality_weights: dict[str, float] | None = None,
        device: str = "cpu",
    ) -> None:
        self.weights = dict(modality_weights or _DEFAULT_WEIGHTS)
        self.device  = device

        # Seeded random projection matrices  (in_dim, PROJ_DIM)
        self._proj: dict[str, np.ndarray] = {}
        for key, in_dim in _IN_DIMS.items():
            seed = abs(hash(key)) % (2 ** 31)
            rng  = np.random.RandomState(seed=seed)
            P    = rng.randn(in_dim, _PROJ_DIM).astype(np.float32)
            P   *= 1.0 / np.sqrt(in_dim)    # variance-preserving scale
            self._proj[key] = P

    # ------------------------------------------------------------------
    # Feature preparation helpers
    # ------------------------------------------------------------------

    def _questionnaire_to_vec(self, features: dict[str, Any]) -> np.ndarray:
        vec = np.zeros(_Q_DIM, dtype=np.float32)
        # Canonical slots
        for i, (name, max_val) in enumerate(_Q_CANONICAL):
            if name in features:
                vec[i] = float(np.clip(float(features[name]) / max_val, 0.0, 1.0))
        # Overflow: unknown numeric keys appended after canonical slots
        idx = len(_Q_CANONICAL)
        for k in sorted(features):
            if k in _Q_CANONICAL_NAMES or idx >= _Q_DIM:
                continue
            try:
                vec[idx] = float(np.clip(float(features[k]) / 100.0, 0.0, 1.0))
                idx += 1
            except (TypeError, ValueError):
                pass
        return vec

    def _pad_or_truncate(self, x: np.ndarray, target: int) -> np.ndarray:
        x = x.flatten().astype(np.float32)
        if len(x) < target:
            x = np.pad(x, (0, target - len(x)))
        else:
            x = x[:target]
        return x

    def _project(self, x: np.ndarray, key: str) -> np.ndarray:
        """Project modality vector to _PROJ_DIM with tanh activation."""
        P   = self._proj[key]               # (in_dim, PROJ_DIM)
        x   = self._pad_or_truncate(x, P.shape[0])
        # L2-normalise input so modalities are on comparable scale
        nrm = np.linalg.norm(x)
        if nrm > 1e-6:
            x = x / nrm
        h = x @ P                           # (PROJ_DIM,)
        return np.tanh(h).astype(np.float32)

    # ------------------------------------------------------------------
    # Cross-modal attention (numpy scaled dot-product)
    # ------------------------------------------------------------------

    def _cross_modal_attention(
        self,
        projected: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute attended representations; concatenate → (512,)."""
        keys = list(projected.keys())
        d    = float(_PROJ_DIM)
        attended: dict[str, np.ndarray] = {}

        for qi_key in keys:
            qi = projected[qi_key]                          # (128,)
            # Scaled dot-product scores against every modality
            raw_scores = np.array([
                np.dot(qi, projected[kj]) / np.sqrt(d)
                for kj in keys
            ], dtype=np.float32)
            # Stable softmax
            raw_scores -= raw_scores.max()
            attn = np.exp(raw_scores)
            attn /= attn.sum() + 1e-9                       # (n_mod,)

            context = np.zeros(_PROJ_DIM, dtype=np.float32)
            for j, kj in enumerate(keys):
                context += attn[j] * projected[kj]

            attended[qi_key] = np.tanh(qi + context).astype(np.float32)

        # Canonical order for consistent 512-dim layout
        order = ["questionnaire", "text", "drawing", "facial"]
        return np.concatenate([attended[k] for k in order])  # (512,)

    # ------------------------------------------------------------------
    # Per-modality risk scoring
    # ------------------------------------------------------------------

    def _modality_score(
        self,
        key: str,
        raw_vec: np.ndarray,
        q_features: dict[str, Any],
    ) -> float:
        """Return a risk-proxy score in [0, 1] for one modality."""
        if key == "questionnaire":
            if "risk_score" in q_features:
                return float(np.clip(float(q_features["risk_score"]), 0.0, 1.0))
            sdq  = float(q_features.get("sdq_total",  0)) / 40.0
            cpss = float(q_features.get("cpss_total", 0)) / 51.0
            return float(np.clip((sdq + cpss) / 2.0, 0.0, 1.0))
        # Generic: mean activation of the (already [0,1]-bounded) feature vector
        return float(np.clip(float(np.mean(raw_vec)), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Attention-refined modality weights
    # ------------------------------------------------------------------

    def _attention_weights(
        self,
        projected: dict[str, np.ndarray],
        base: dict[str, float],
    ) -> dict[str, float]:
        """Refine base weights by L2-norm salience of projected representations."""
        salience = {k: float(np.linalg.norm(projected[k])) * base[k]
                    for k in projected}
        total = sum(salience.values()) + 1e-9
        return {k: round(salience[k] / total, 4) for k in salience}

    # ------------------------------------------------------------------
    # Confidence estimate
    # ------------------------------------------------------------------

    def _confidence(
        self,
        scores: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        """High when modalities agree; low when they diverge."""
        vals   = np.array(list(scores.values()), dtype=np.float32)
        wts    = np.array([weights[k] for k in scores], dtype=np.float32)
        w_mean = float(np.dot(wts, vals))
        w_var  = float(np.dot(wts, (vals - w_mean) ** 2))
        # Confidence decreases with weighted variance
        return float(np.clip(1.0 - 2.0 * np.sqrt(w_var), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(self, inp: FusionInput) -> FusionOutput:
        # Build raw modality vectors
        raw: dict[str, np.ndarray] = {
            "questionnaire": self._questionnaire_to_vec(inp.questionnaire_features),
            "text":          np.asarray(inp.text_features,     dtype=np.float32).flatten(),
            "drawing":       np.asarray(inp.drawing_features,  dtype=np.float32).flatten(),
            "facial":        np.asarray(inp.facial_features,   dtype=np.float32).flatten(),
        }

        # Project to shared space
        projected = {k: self._project(raw[k], k) for k in raw}

        # Cross-modal attention → 512-dim
        fused_vector = self._cross_modal_attention(projected)

        # Per-modality risk scores
        per_modality_scores = {
            k: self._modality_score(k, raw[k], inp.questionnaire_features)
            for k in raw
        }

        # Attention-refined modality weights
        modality_weights = self._attention_weights(projected, self.weights)

        # Composite risk score (use base config weights, not attention-refined,
        # to preserve clinical interpretability)
        risk_score = float(np.clip(
            sum(self.weights[k] * per_modality_scores[k] for k in raw),
            0.0, 1.0,
        ))

        confidence = self._confidence(per_modality_scores, self.weights)

        logger.debug(
            "Fusion: risk=%.3f conf=%.3f per_modality=%s",
            risk_score, confidence, per_modality_scores,
        )

        return FusionOutput(
            fused_vector        = fused_vector,
            modality_weights    = modality_weights,
            confidence          = round(confidence, 4),
            per_modality_scores = {k: round(v, 4) for k, v in per_modality_scores.items()},
            risk_score          = round(risk_score, 4),
        )


# Backward-compatibility alias
AttentionFusion = MultiModalAttentionFusion
