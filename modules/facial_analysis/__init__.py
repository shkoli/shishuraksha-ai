"""Facial analysis module — expression detection and emotion classification for screening."""

from modules.facial_analysis.expression_detector import (
    FacialFeatureExtractor,
    AUExtractionResult,
    AU_LABELS,
)
from modules.facial_analysis.emotion_classifier import (
    EmotionSequenceClassifier,
    SessionEmotionResult,
)

__all__ = [
    "FacialFeatureExtractor",
    "AUExtractionResult",
    "AU_LABELS",
    "EmotionSequenceClassifier",
    "SessionEmotionResult",
]
