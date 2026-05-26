"""Bengali NLP preprocessing and BanglaBERT embedding utilities.

BengaliTextPreprocessor — Unicode NFC normalisation, zero-width character removal,
    noise stripping, and whitespace-based tokenisation for clinical Bengali text.
BanglaBERTEmbedder — Loads csebuetnlp/banglabert and extracts 768-dim CLS embeddings.
    Falls back silently to zero-vectors when transformers is not installed
    or the model cannot be downloaded (offline / air-gapped deployment).
BengaliNLPPipeline — Unified facade combining preprocessor + embedder.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Zero-width characters frequently injected by Bengali input methods / copy-paste
_ZERO_WIDTH = frozenset('​‌‍‎‏﻿')

_BN_DIGIT_TABLE = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
_PUNCTUATION_RE = re.compile(r'[।॥,;:!?\'\"()\[\]{}]+')
_URL_RE = re.compile(r'https?://\S+|www\.\S+')
_WS_RE = re.compile(r'\s+')


def _nfc(text: str) -> str:
    return unicodedata.normalize('NFC', text)


@dataclass
class PreprocessedText:
    original:      str
    normalised:    str
    tokens:        list[str]
    n_tokens:      int
    has_bengali:   bool
    has_latin:     bool


class BengaliTextPreprocessor:
    """Normalises and tokenises Bengali clinical text."""

    def __init__(
        self,
        normalise_digits: bool = False,
        remove_urls:      bool = True,
        lowercase_latin:  bool = True,
        strip_html:       bool = True,
    ) -> None:
        self.normalise_digits = normalise_digits
        self.remove_urls      = remove_urls
        self.lowercase_latin  = lowercase_latin
        self.strip_html       = strip_html

    def normalize(self, text: str) -> str:
        text = _nfc(text)
        text = ''.join(ch for ch in text if ch not in _ZERO_WIDTH)
        if self.strip_html:
            text = re.sub(r'<[^>]+>', ' ', text)
        if self.remove_urls:
            text = _URL_RE.sub(' ', text)
        if self.lowercase_latin:
            text = ''.join(
                ch.lower() if ch.isascii() and ch.isalpha() else ch
                for ch in text
            )
        if self.normalise_digits:
            text = text.translate(_BN_DIGIT_TABLE)
        text = _WS_RE.sub(' ', text).strip()
        return text

    def remove_noise(self, text: str) -> str:
        """Replace Bengali dandas, brackets, and punctuation with spaces."""
        text = _PUNCTUATION_RE.sub(' ', text)
        return _WS_RE.sub(' ', text).strip()

    def tokenize(self, text: str) -> list[str]:
        # Replace dandas with spaces so they don't attach to adjacent words
        text = re.sub(r'[।॥]', ' ', text)
        raw_tokens = text.split()
        tokens = [t.strip('.,;:!?\'"()[]{}') for t in raw_tokens]
        return [_nfc(t) for t in tokens if t]

    def preprocess(self, text: str) -> PreprocessedText:
        normalised = self.normalize(text)
        tokens     = self.tokenize(normalised)
        has_bengali = any('ঀ' <= ch <= '৿' for ch in normalised)
        has_latin   = any(ch.isascii() and ch.isalpha() for ch in normalised)
        return PreprocessedText(
            original    = text,
            normalised  = normalised,
            tokens      = tokens,
            n_tokens    = len(tokens),
            has_bengali = has_bengali,
            has_latin   = has_latin,
        )

    def batch_preprocess(self, texts: list[str]) -> list[PreprocessedText]:
        return [self.preprocess(t) for t in texts]


class BanglaBERTEmbedder:
    """Extracts 768-dim CLS embeddings via csebuetnlp/banglabert.

    When offline=True, loads from local HuggingFace cache only. Falls back to
    zero-vectors if transformers is not installed or the model isn't cached.
    """

    EMBEDDING_DIM: int = 768
    DEFAULT_MODEL: str = "csebuetnlp/banglabert"

    def __init__(
        self,
        model_name: str  | None = None,
        device:     str         = "cpu",
        offline:    bool        = False,
    ) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device     = device
        self.offline    = offline

        self._tokenizer: Any = None
        self._model:     Any = None
        self._available: bool = False

        self._try_load()

    def _try_load(self) -> None:
        try:
            from transformers import AutoTokenizer, AutoModel  # noqa: PLC0415
            import torch                                        # noqa: PLC0415

            hf_kwargs: dict[str, Any] = {}
            if self.offline:
                hf_kwargs["local_files_only"] = True

            logger.info("Loading BanglaBERT tokeniser from %s …", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **hf_kwargs)
            logger.info("Loading BanglaBERT model …")
            self._model = AutoModel.from_pretrained(self.model_name, **hf_kwargs)
            self._model.to(self.device)
            self._model.eval()
            self._available = True
            logger.info("BanglaBERT ready (%s) on device=%s", self.model_name, self.device)

        except ImportError:
            logger.warning(
                "transformers / torch not installed. "
                "BanglaBERTEmbedder running in stub mode (zero vectors)."
            )
        except OSError as exc:
            logger.warning(
                "BanglaBERT model not found at %s: %s. Running in stub mode.",
                self.model_name, exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("BanglaBERT failed to load: %s. Running in stub mode.", exc)

    @property
    def is_available(self) -> bool:
        return self._available

    def embed(self, text: str, max_length: int = 512) -> np.ndarray:
        """Return the 768-dim CLS embedding for *text*, or zeros if unavailable."""
        if not self._available:
            return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)

        try:
            import torch  # noqa: PLC0415

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            cls_vec: np.ndarray = (
                outputs.last_hidden_state[:, 0, :]
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            return cls_vec

        except Exception as exc:  # noqa: BLE001
            logger.error("BanglaBERT embed() failed: %s", exc)
            return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)

    def embed_with_attentions(
        self,
        text: str,
        max_length: int = 512,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Return (cls_embedding, tokens, per_token_attention).

        Attention is averaged across all heads in the last encoder layer.
        """
        if not self._available:
            tokens = text.split()
            attention = np.zeros(len(tokens), dtype=np.float32)
            return np.zeros(self.EMBEDDING_DIM, dtype=np.float32), tokens, attention

        try:
            import torch  # noqa: PLC0415

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs, output_attentions=True)

            cls_vec: np.ndarray = (
                outputs.last_hidden_state[:, 0, :]
                .squeeze(0).cpu().numpy().astype(np.float32)
            )
            # outputs.attentions: tuple of (batch, heads, seq, seq) per layer
            last_attn = outputs.attentions[-1]           # (1, H, S, S)
            cls_attention = (
                last_attn[0].mean(dim=0)[0, :]           # (S,)
                .cpu().numpy().astype(np.float32)
            )
            subword_tokens: list[str] = self._tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0].tolist()
            )
            return cls_vec, subword_tokens, cls_attention

        except Exception as exc:  # noqa: BLE001
            logger.error("embed_with_attentions() failed: %s", exc)
            tokens = text.split()
            return (
                np.zeros(self.EMBEDDING_DIM, dtype=np.float32),
                tokens,
                np.zeros(len(tokens), dtype=np.float32),
            )

    def embed_batch(
        self,
        texts:      list[str],
        max_length: int = 512,
        batch_size: int = 8,
    ) -> np.ndarray:
        """Embed a list of texts; returns ndarray of shape (N, 768)."""
        if not texts:
            return np.zeros((0, self.EMBEDDING_DIM), dtype=np.float32)
        results: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            results.extend(self.embed(t, max_length) for t in chunk)
        return np.stack(results)


class BengaliNLPPipeline:
    """Unified preprocessing + BanglaBERT embedding pipeline."""

    def __init__(
        self,
        model_name: str  | None = None,
        device:     str         = "cpu",
        offline:    bool        = False,
    ) -> None:
        self.preprocessor = BengaliTextPreprocessor()
        self.embedder      = BanglaBERTEmbedder(model_name, device, offline)

    @property
    def is_available(self) -> bool:
        return self.embedder.is_available

    def preprocess(self, text: str) -> PreprocessedText:
        return self.preprocessor.preprocess(text)

    def tokenize(self, text: str) -> list[str]:
        return self.preprocessor.tokenize(self.preprocessor.normalize(text))

    def embed(self, text: str, max_length: int = 512) -> np.ndarray:
        """Preprocess then embed; returns 768-dim CLS vector."""
        normalised = self.preprocessor.normalize(text)
        return self.embedder.embed(normalised, max_length)

    def embed_with_attentions(
        self, text: str, max_length: int = 512
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        normalised = self.preprocessor.normalize(text)
        return self.embedder.embed_with_attentions(normalised, max_length)

    def embed_batch(
        self, texts: list[str], max_length: int = 512, batch_size: int = 8
    ) -> np.ndarray:
        normalised = [self.preprocessor.normalize(t) for t in texts]
        return self.embedder.embed_batch(normalised, max_length, batch_size)
