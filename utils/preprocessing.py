"""Shared data preprocessing utilities for all modalities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class TabularPreprocessor:
    """Preprocesses tabular questionnaire feature matrices."""

    def __init__(
        self,
        strategy: str = "median",
        scale: bool = True,
    ) -> None:
        self.strategy = strategy
        self.scale = scale
        self._imputer: Any = None
        self._scaler: Any = None
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        ...

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    def load(cls, path: str | Path) -> "TabularPreprocessor":
        ...


class ImagePreprocessor:
    """Preprocesses drawing images for CNN inference."""

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        normalise: bool = True,
    ) -> None:
        self.image_size = image_size
        self.normalise = normalise
        self._mean = np.array([0.485, 0.456, 0.406])
        self._std = np.array([0.229, 0.224, 0.225])

    def load_image(self, path: str | Path) -> np.ndarray:
        ...

    def resize(self, image: np.ndarray) -> np.ndarray:
        ...

    def normalise_image(self, image: np.ndarray) -> np.ndarray:
        ...

    def to_tensor(self, image: np.ndarray) -> Any:
        ...

    def preprocess(self, path: str | Path) -> Any:
        ...


class VideoPreprocessor:
    """Extracts and preprocesses frames from facial video clips."""

    def __init__(
        self,
        sample_rate: int = 5,
        image_size: tuple[int, int] = (112, 112),
    ) -> None:
        self.sample_rate = sample_rate
        self.image_size = image_size

    def extract_frames(self, video_path: str | Path) -> list[np.ndarray]:
        ...

    def preprocess_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        ...
