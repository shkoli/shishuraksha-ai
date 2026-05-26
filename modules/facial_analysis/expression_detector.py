"""Facial Action Unit extraction via MediaPipe Face Mesh.

Extracts 8 trauma-relevant AUs (AU4/6/7/12/15/17/20/43) from 468 landmarks.
AU scores are Euclidean-distance approximations normalised by inter-ocular width
for scale invariance. Falls back to zero vectors when mediapipe is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── AU labels in canonical order ─────────────────────────────────────────────
AU_LABELS: list[str] = ["AU4", "AU6", "AU7", "AU12", "AU15", "AU17", "AU20", "AU43"]
AU_DIM = len(AU_LABELS)          # 8
FEATURE_DIM = 32                 # 8 raw AUs + 8 squared + 8 cross-products + 8 stats


# ── MediaPipe landmark indices (Face Mesh 468-point model) ───────────────────
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# Indices verified against the canonical face model topology.

_LM = {
    # Brow landmarks
    "left_brow_inner":   107,
    "left_brow_outer":   46,
    "right_brow_inner":  336,
    "right_brow_outer":  276,
    "left_brow_mid":     52,
    "right_brow_mid":    282,

    # Eyelid landmarks
    "left_eye_upper":    159,
    "left_eye_lower":    145,
    "right_eye_upper":   386,
    "right_eye_lower":   374,
    "left_eye_inner":    133,
    "left_eye_outer":    33,
    "right_eye_inner":   362,
    "right_eye_outer":   263,

    # Cheek
    "left_cheek":        116,
    "right_cheek":       345,

    # Lip corners and edges
    "lip_left":          61,
    "lip_right":         291,
    "lip_upper_mid":     13,
    "lip_lower_mid":     14,
    "lip_upper_left":    37,
    "lip_upper_right":   267,
    "lip_lower_left":    84,
    "lip_lower_right":   314,

    # Chin
    "chin_mid":          175,
    "chin_lower":        199,

    # Nose (reference)
    "nose_tip":          4,
    "nose_bridge":       6,
}


def _pt(lms: Any, idx: int) -> np.ndarray:
    """Return (x, y) for MediaPipe landmark at index."""
    lm = lms.landmark[idx]
    return np.array([lm.x, lm.y], dtype=np.float32)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _sigmoid(x: float, k: float = 10.0, x0: float = 0.5) -> float:
    """Soft threshold to map raw ratios into [0, 1]."""
    import math
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


# ═══════════════════════════════════════════════════════════════════════════════
# AUExtractionResult
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AUExtractionResult:
    au_scores:      dict[str, float]
    feature_vector: np.ndarray
    face_detected:  bool

    def __repr__(self) -> str:
        status = "face_detected" if self.face_detected else "NO FACE"
        au_str = "  " + "\n  ".join(f"{k}: {v:.4f}" for k, v in self.au_scores.items())
        return (
            f"AUExtractionResult({status})\n"
            f"{au_str}\n"
            f"  feature_vector: shape={self.feature_vector.shape}, "
            f"norm={np.linalg.norm(self.feature_vector):.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FacialFeatureExtractor
# ═══════════════════════════════════════════════════════════════════════════════

class FacialFeatureExtractor:
    """Extracts 8 trauma-relevant Action Units using MediaPipe Face Mesh."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence:  float = 0.5,
        static_image_mode:        bool  = True,
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence  = min_tracking_confidence
        self.static_image_mode        = static_image_mode

        self._face_mesh: Any  = None
        self._available: bool = False

        self._try_load()

    # ------------------------------------------------------------------
    def _try_load(self) -> None:
        try:
            import mediapipe as mp  # noqa: PLC0415

            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode        = self.static_image_mode,
                max_num_faces            = 1,
                refine_landmarks         = True,
                min_detection_confidence = self.min_detection_confidence,
                min_tracking_confidence  = self.min_tracking_confidence,
            )
            self._available = True
            logger.info("FacialFeatureExtractor ready (MediaPipe Face Mesh).")
        except ImportError:
            logger.warning(
                "mediapipe not installed — FacialFeatureExtractor in stub mode (zero vectors)."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("FacialFeatureExtractor load failed: %s — stub mode.", exc)

    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    def _load_image(self, source: Any) -> np.ndarray:
        if isinstance(source, (str, Path)):
            import cv2  # noqa: PLC0415
            bgr = cv2.imread(str(source))
            if bgr is None:
                raise FileNotFoundError(f"Image not found: {source}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        arr = np.asarray(source, dtype=np.uint8)
        # Assume BGR if loaded via OpenCV convention, else RGB
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr  # caller must ensure RGB for MediaPipe

    # ------------------------------------------------------------------
    # AU computation helpers
    # ------------------------------------------------------------------

    def _reference_dist(self, lms: Any) -> float:
        """Inter-ocular distance as scale-invariant reference."""
        left  = _pt(lms, _LM["left_eye_outer"])
        right = _pt(lms, _LM["right_eye_outer"])
        d = _dist(left, right)
        return d if d > 1e-6 else 1.0

    def _au4(self, lms: Any, ref: float) -> float:
        """Brow Lowerer: mean brow-to-eye-upper distance (low = depressed)."""
        left_brow  = _pt(lms, _LM["left_brow_mid"])
        right_brow = _pt(lms, _LM["right_brow_mid"])
        left_eye   = _pt(lms, _LM["left_eye_upper"])
        right_eye  = _pt(lms, _LM["right_eye_upper"])
        raw = (_dist(left_brow, left_eye) + _dist(right_brow, right_eye)) / (2.0 * ref)
        # Smaller gap → higher AU4 score
        return float(np.clip(1.0 - raw * 4.0, 0.0, 1.0))

    def _au6(self, lms: Any, ref: float) -> float:
        """Cheek Raiser: cheek-to-eye-lower elevation."""
        left_cheek  = _pt(lms, _LM["left_cheek"])
        right_cheek = _pt(lms, _LM["right_cheek"])
        left_eye    = _pt(lms, _LM["left_eye_lower"])
        right_eye   = _pt(lms, _LM["right_eye_lower"])
        raw = (_dist(left_cheek, left_eye) + _dist(right_cheek, right_eye)) / (2.0 * ref)
        # Cheek raised → shorter distance
        return float(np.clip(1.0 - raw * 3.0, 0.0, 1.0))

    def _au7(self, lms: Any, ref: float) -> float:
        """Lid Tightener: upper–lower eyelid aperture (narrow = tense)."""
        left_ap  = _dist(_pt(lms, _LM["left_eye_upper"]),  _pt(lms, _LM["left_eye_lower"]))
        right_ap = _dist(_pt(lms, _LM["right_eye_upper"]), _pt(lms, _LM["right_eye_lower"]))
        raw = (left_ap + right_ap) / (2.0 * ref)
        # Narrow aperture → high lid tension
        return float(np.clip(1.0 - raw * 8.0, 0.0, 1.0))

    def _au12(self, lms: Any, ref: float) -> float:
        """Lip Corner Puller: lateral displacement of lip corners."""
        left  = _pt(lms, _LM["lip_left"])
        right = _pt(lms, _LM["lip_right"])
        raw = _dist(left, right) / ref
        return float(np.clip((raw - 0.3) * 3.0, 0.0, 1.0))

    def _au15(self, lms: Any, ref: float) -> float:
        """Lip Corner Depressor: downward drop of lip corners vs mid-upper lip."""
        left_corner  = _pt(lms, _LM["lip_lower_left"])
        right_corner = _pt(lms, _LM["lip_lower_right"])
        upper_mid    = _pt(lms, _LM["lip_upper_mid"])
        drop = ((left_corner[1] + right_corner[1]) / 2.0) - upper_mid[1]
        raw  = drop / ref
        return float(np.clip(raw * 5.0, 0.0, 1.0))

    def _au17(self, lms: Any, ref: float) -> float:
        """Chin Raiser: chin boss to lower lip distance (small = raised)."""
        chin = _pt(lms, _LM["chin_mid"])
        lip  = _pt(lms, _LM["lip_lower_mid"])
        raw  = _dist(chin, lip) / ref
        return float(np.clip(1.0 - raw * 5.0, 0.0, 1.0))

    def _au20(self, lms: Any, ref: float) -> float:
        """Lip Stretcher: lip width relative to face width."""
        left  = _pt(lms, _LM["lip_left"])
        right = _pt(lms, _LM["lip_right"])
        raw   = _dist(left, right) / ref
        return float(np.clip((raw - 0.45) * 4.0, 0.0, 1.0))

    def _au43(self, lms: Any, ref: float) -> float:
        """Eyes Closed: very small eye aperture indicates closure/blink."""
        left_ap  = _dist(_pt(lms, _LM["left_eye_upper"]),  _pt(lms, _LM["left_eye_lower"]))
        right_ap = _dist(_pt(lms, _LM["right_eye_upper"]), _pt(lms, _LM["right_eye_lower"]))
        mean_ap  = (left_ap + right_ap) / 2.0
        raw      = mean_ap / ref
        # Very small aperture → high AU43
        return float(np.clip(1.0 - raw * 12.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _compute_aus(self, lms: Any) -> dict[str, float]:
        ref = self._reference_dist(lms)
        return {
            "AU4":  self._au4(lms, ref),
            "AU6":  self._au6(lms, ref),
            "AU7":  self._au7(lms, ref),
            "AU12": self._au12(lms, ref),
            "AU15": self._au15(lms, ref),
            "AU17": self._au17(lms, ref),
            "AU20": self._au20(lms, ref),
            "AU43": self._au43(lms, ref),
        }

    def _build_feature_vector(self, au_scores: dict[str, float]) -> np.ndarray:
        """Build 32-dim vector: raw(8) + squared(8) + pairwise-products(8) + stats(8)."""
        raw = np.array([au_scores[k] for k in AU_LABELS], dtype=np.float32)
        squared = raw ** 2

        # 8 clinically-motivated cross-products
        cross = np.array([
            au_scores["AU4"]  * au_scores["AU7"],    # brow lowerer × lid tightener (fear/anger)
            au_scores["AU4"]  * au_scores["AU15"],   # brow lowerer × lip depressor (sadness)
            au_scores["AU6"]  * au_scores["AU12"],   # cheek raiser × lip puller   (Duchenne smile)
            au_scores["AU20"] * au_scores["AU43"],   # lip stretcher × eyes closed  (freeze)
            au_scores["AU7"]  * au_scores["AU43"],   # lid tightener × eyes closed  (hypervigilance)
            au_scores["AU15"] * au_scores["AU17"],   # lip depressor × chin raiser  (distress)
            au_scores["AU4"]  * au_scores["AU20"],   # brow lowerer × lip stretcher (fear)
            au_scores["AU12"] * au_scores["AU15"],   # lip puller × lip depressor   (conflict)
        ], dtype=np.float32)

        # 8 global statistics
        stats = np.array([
            raw.mean(),
            raw.std(),
            raw.max(),
            raw.min(),
            float(np.percentile(raw, 25)),
            float(np.percentile(raw, 75)),
            float((raw > 0.5).sum()),        # count of high-activation AUs
            float(np.sum(raw)),              # total activation mass
        ], dtype=np.float32)

        return np.concatenate([raw, squared, cross, stats])  # 8+8+8+8 = 32

    # ------------------------------------------------------------------
    def _zero_result(self) -> AUExtractionResult:
        return AUExtractionResult(
            au_scores      = {k: 0.0 for k in AU_LABELS},
            feature_vector = np.zeros(FEATURE_DIM, dtype=np.float32),
            face_detected  = False,
        )

    # ------------------------------------------------------------------
    def extract(self, source: Any) -> AUExtractionResult:
        if not self._available:
            return self._zero_result()

        try:
            rgb = self._load_image(source)
            results = self._face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                logger.debug("No face detected in image.")
                return self._zero_result()

            lms       = results.multi_face_landmarks[0]
            au_scores = self._compute_aus(lms)
            fv        = self._build_feature_vector(au_scores)

            return AUExtractionResult(
                au_scores      = au_scores,
                feature_vector = fv,
                face_detected  = True,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("extract() failed: %s", exc)
            return self._zero_result()

    def extract_batch(self, sources: list[Any]) -> list[AUExtractionResult]:
        """Extract features from a list of image sources."""
        return [self.extract(s) for s in sources]
