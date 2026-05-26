"""Trauma indicator detection from Bengali clinical narratives."""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── NFC helper ────────────────────────────────────────────────────────────────

def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# ── Rule-based keyword sets ───────────────────────────────────────────────────

_TRAUMA_WORDS: frozenset[str] = frozenset(_nfc(w) for w in [
    "নির্যাতন", "নির্যাতিত", "আঘাত", "মারধর", "মেরেছিল",
    "জোর", "জোরপূর্বক", "ধর্ষণ", "যৌন", "অশ্লীল",
    "ছুঁয়েছিল", "স্পর্শ", "হুমকি", "বাঁচাও", "লুকিয়ে",
    "ভয়", "আতঙ্ক", "কষ্ট", "যন্ত্রণা", "অসহায়",
    "কেউ বিশ্বাস করেনি", "গোপন", "চুপ",
])

_STRONG_TRAUMA_WORDS: frozenset[str] = frozenset(_nfc(w) for w in [
    "নির্যাতন", "ধর্ষণ", "যৌন নির্যাতন", "জোরপূর্বক",
    "মারধর", "বাঁচাও",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TraumaDetectionResult
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TraumaDetectionResult:
    trauma_prob:        float
    key_phrases:        list[str]
    confidence:         float
    is_trauma_positive: bool
    mode:               str = "rule_based"

    def summary(self) -> str:
        label = "TRAUMA" if self.is_trauma_positive else "NON-TRAUMA"
        phrases = ", ".join(self.key_phrases) if self.key_phrases else "—"
        return (
            f"[{label}] prob={self.trauma_prob:.3f} "
            f"conf={self.confidence:.3f} ({self.mode})\n"
            f"  key_phrases: {phrases}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TraumaIndicatorClassifier
# ═══════════════════════════════════════════════════════════════════════════════

class TraumaIndicatorClassifier:
    """BanglaBERT binary trauma classifier with rule-based fallback."""

    DEFAULT_MODEL: str = "csebuetnlp/banglabert"

    def __init__(
        self,
        model_name: str  | None = None,
        threshold:  float        = 0.50,
        device:     str          = "cpu",
        offline:    bool         = False,
    ) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self.threshold  = threshold
        self.device     = device
        self.offline    = offline

        self._tokenizer: Any  = None
        self._model:     Any  = None
        self._available: bool = False

        self._try_load()

    # ------------------------------------------------------------------
    def _try_load(self) -> None:
        try:
            from transformers import AutoTokenizer, AutoModel  # noqa: PLC0415
            import torch                                        # noqa: PLC0415

            hf_kwargs: dict[str, Any] = {}
            if self.offline:
                hf_kwargs["local_files_only"] = True

            logger.info("TraumaIndicatorClassifier: loading %s …", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, **hf_kwargs
            )
            self._model = AutoModel.from_pretrained(
                self.model_name, **hf_kwargs
            )
            self._model.to(self.device)
            self._model.eval()
            self._available = True
            logger.info("TraumaIndicatorClassifier ready (neural mode).")

        except ImportError:
            logger.warning(
                "transformers/torch not installed — TraumaIndicatorClassifier "
                "running in rule-based mode."
            )
        except OSError as exc:
            logger.warning(
                "BanglaBERT not found at %s: %s — rule-based mode.",
                self.model_name, exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TraumaIndicatorClassifier load failed: %s — rule-based mode.", exc
            )

    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    def _tokenize_words(self, text: str) -> list[str]:
        text = _nfc(text)
        raw = text.split()
        tokens: list[str] = []
        for tok in raw:
            tok = tok.strip("।॥.,;:!?'\"""()[]{}—–")
            tok = _nfc(tok)
            if tok:
                tokens.append(tok)
        return tokens

    # ------------------------------------------------------------------
    # Rule-based path
    # ------------------------------------------------------------------
    def _rule_based_predict(self, text: str) -> TraumaDetectionResult:
        tokens = self._tokenize_words(text)
        n = max(1, len(tokens))

        weighted_hits: list[tuple[float, str]] = []
        for tok in tokens:
            if tok in _STRONG_TRAUMA_WORDS:
                weighted_hits.append((2.0, tok))
            elif tok in _TRAUMA_WORDS:
                weighted_hits.append((1.0, tok))

        total_weight = sum(w for w, _ in weighted_hits)
        # Normalise: 3 strong hits in a 10-token text → ~0.6 probability
        raw_score = min(1.0, total_weight / (n * 0.5 + 1e-9))

        # Sigmoid-like squashing so very short texts don't spike
        import math
        trauma_prob = 1.0 / (1.0 + math.exp(-8.0 * (raw_score - 0.3)))
        trauma_prob = float(np.clip(trauma_prob, 0.0, 1.0))

        # Key phrases: top-5 by weight (deduplicated, preserving order)
        seen: set[str] = set()
        sorted_hits = sorted(weighted_hits, key=lambda x: -x[0])
        key_phrases: list[str] = []
        for _, phrase in sorted_hits:
            if phrase not in seen:
                key_phrases.append(phrase)
                seen.add(phrase)
            if len(key_phrases) == 5:
                break

        confidence = min(1.0, total_weight / 3.0) if total_weight > 0 else (
            1.0 - trauma_prob  # confident it's non-trauma
        )

        return TraumaDetectionResult(
            trauma_prob        = trauma_prob,
            key_phrases        = key_phrases,
            confidence         = float(np.clip(confidence, 0.0, 1.0)),
            is_trauma_positive = trauma_prob >= self.threshold,
            mode               = "rule_based",
        )

    # ------------------------------------------------------------------
    # Neural path
    # ------------------------------------------------------------------
    def _neural_predict(self, text: str) -> TraumaDetectionResult:
        try:
            import torch  # noqa: PLC0415

            enc_text = _nfc(text)
            inputs = self._tokenizer(
                enc_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs, output_attentions=True)

            # CLS embedding → linear probe for trauma probability
            cls_vec = (
                outputs.last_hidden_state[:, 0, :]
                .squeeze(0).cpu().numpy().astype(np.float32)
            )
            # Without a fine-tuned head, use L2-norm of CLS as a proxy score,
            # scaled to [0,1].  This will be replaced when a head is loaded.
            norm = float(np.linalg.norm(cls_vec))
            # Heuristic calibration (placeholder until fine-tuned head is wired)
            trauma_prob = float(np.clip(norm / 30.0, 0.0, 1.0))

            # Attention-based key phrases: mean over heads, last layer, CLS row
            last_attn = outputs.attentions[-1]       # (1, H, S, S)
            cls_row = (
                last_attn[0].mean(dim=0)[0, :]       # (S,)
                .cpu().numpy().astype(np.float32)
            )
            subword_tokens: list[str] = self._tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0].tolist()
            )
            # Pair each subword with its attention score, filter specials
            scored = [
                (float(score), tok)
                for score, tok in zip(cls_row, subword_tokens)
                if tok not in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>")
                   and not tok.startswith("##")
            ]
            scored.sort(key=lambda x: -x[0])
            key_phrases = [tok for _, tok in scored[:5]]

            confidence = float(np.clip(abs(trauma_prob - 0.5) * 2, 0.0, 1.0))

            return TraumaDetectionResult(
                trauma_prob        = trauma_prob,
                key_phrases        = key_phrases,
                confidence         = confidence,
                is_trauma_positive = trauma_prob >= self.threshold,
                mode               = "neural",
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("Neural predict failed: %s — falling back to rule-based.", exc)
            result = self._rule_based_predict(text)
            return result

    # ------------------------------------------------------------------
    def predict(self, text: str) -> TraumaDetectionResult:
        if self._available:
            return self._neural_predict(text)
        return self._rule_based_predict(text)

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> list[TraumaDetectionResult]:
        """Classify a list of narratives."""
        return [self.predict(t) for t in texts]


# ── Backward-compatibility alias ──────────────────────────────────────────────
TraumaDetector = TraumaIndicatorClassifier
