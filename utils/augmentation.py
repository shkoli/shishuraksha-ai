"""Data augmentation utilities for drawing and text modalities.

Augmentation is applied only during training to improve model
generalisation on limited clinical datasets.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class DrawingAugmentor:
    """Augmentation pipeline for child drawing images."""

    def __init__(
        self,
        p_flip: float = 0.0,
        p_rotate: float = 0.3,
        p_colour_jitter: float = 0.2,
        p_elastic: float = 0.1,
        max_rotation: float = 10.0,
    ) -> None:
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_colour_jitter = p_colour_jitter
        self.p_elastic = p_elastic
        self.max_rotation = max_rotation
        self._transform: Any = None
        self._build()

    def _build(self) -> None:
        """Compose the augmentation pipeline using torchvision/albumentations."""
        ...

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply the augmentation pipeline to a single image."""
        ...

    def augment_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Apply augmentation to a list of images."""
        ...


class TextAugmentor:
    """Text-level augmentation for Bengali clinical narratives.

    Applies synonym replacement, back-translation, and sentence shuffling
    to increase training data diversity without altering clinical meaning.
    """

    def __init__(
        self,
        p_synonym: float = 0.1,
        p_insert: float = 0.05,
        p_swap: float = 0.05,
        p_delete: float = 0.05,
    ) -> None:
        self.p_synonym = p_synonym
        self.p_insert = p_insert
        self.p_swap = p_swap
        self.p_delete = p_delete

    def synonym_replace(self, text: str, n: int = 1) -> str:
        """Replace n random non-stopwords with Bengali synonyms."""
        ...

    def random_insert(self, text: str, n: int = 1) -> str:
        """Insert n random synonym words at random positions."""
        ...

    def random_swap(self, text: str, n: int = 1) -> str:
        """Swap n random word pairs in the text."""
        ...

    def random_delete(self, text: str, p: float = 0.05) -> str:
        """Delete each word with probability p."""
        ...

    def augment(self, text: str) -> str:
        """Apply all configured augmentation operations to a text sample."""
        ...

    def augment_batch(self, texts: list[str]) -> list[str]:
        """Augment a list of text samples."""
        ...
