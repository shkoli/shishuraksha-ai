"""Drawing-based psychiatric risk classification with Grad-CAM explainability."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from modules.drawing_analysis.feature_extractor import (
    CNN_DIM,
    MARKER_DIM,
    TOTAL_DIM,
    DrawingFeatureExtractor,
)
from modules.drawing_analysis.trauma_markers import MARKERS, HTPMarkerExtractor

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Output dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DrawingRiskOutput:
    risk_score:        float                     # sigmoid output in [0, 1]
    activated_markers: list[str]                 # HTP markers present in input
    gradcam_map:       np.ndarray                # (H, W) saliency map, or zeros stub
    burden_score:      float   = 0.0             # from HTPMarkerExtractor
    mode:              str     = "neural"        # 'neural' | 'stub'

    def __repr__(self) -> str:
        markers = ", ".join(self.activated_markers) if self.activated_markers else "none"
        return (
            f"DrawingRiskOutput(\n"
            f"  risk_score={self.risk_score:.4f},\n"
            f"  burden_score={self.burden_score:.2f},\n"
            f"  activated_markers=[{markers}],\n"
            f"  gradcam_map.shape={self.gradcam_map.shape},\n"
            f"  mode={self.mode!r}\n"
            f")"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Classifier head (pure nn.Module, defined only when torch is available)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_head() -> Any:
    """Return a 1300→256→64→1 classifier nn.Sequential."""
    import torch.nn as nn  # noqa: PLC0415

    return nn.Sequential(
        nn.Linear(TOTAL_DIM, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DrawingRiskClassifier
# ═══════════════════════════════════════════════════════════════════════════════

class DrawingRiskClassifier:
    """EfficientNet-B0 + 1300→256→64→1 head with Grad-CAM; falls back to HTP burden score."""

    def __init__(
        self,
        device:    str   = "cpu",
        threshold: float = 0.50,
        offline:   bool  = False,
    ) -> None:
        self.device    = device
        self.threshold = threshold

        self._feature_extractor = DrawingFeatureExtractor(device=device, offline=offline)
        self._marker_extractor  = HTPMarkerExtractor()

        self._head:      Any  = None
        self._available: bool = False

        # Grad-CAM hook storage
        self._fmap:  Any = None   # forward activations
        self._grad:  Any = None   # backward gradients

        self._try_build()

    # ------------------------------------------------------------------
    def _try_build(self) -> None:
        try:
            import torch  # noqa: PLC0415

            self._head = _build_head()
            self._head.to(self.device)
            self._head.eval()

            # Register Grad-CAM hooks on last conv block of the backbone
            if self._feature_extractor.is_available:
                last_layer = self._feature_extractor._model.features[-1]

                def _save_fmap(module, inp, out):
                    self._fmap = out

                def _save_grad(module, grad_in, grad_out):
                    self._grad = grad_out[0]

                last_layer.register_forward_hook(_save_fmap)
                last_layer.register_backward_hook(_save_grad)

            self._available = True
            logger.info("DrawingRiskClassifier ready (device=%s).", self.device)

        except ImportError as exc:
            logger.warning(
                "torch not installed: %s — stub mode (rule-based risk score).", exc
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("DrawingRiskClassifier build failed: %s — stub mode.", exc)

    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    def _gradcam_map(self, image_tensor: Any) -> np.ndarray:
        """Compute Grad-CAM saliency map; returns (7, 7) float32 array."""
        try:
            import torch  # noqa: PLC0415
            import torch.nn.functional as F  # noqa: PLC0415

            self._fmap = None
            self._grad = None

            # Enable gradient tracking for Grad-CAM
            image_tensor = image_tensor.requires_grad_(True)
            cnn_out = self._feature_extractor._model.features(image_tensor)
            # Global average pool (same as the backbone does)
            pooled = cnn_out.mean(dim=[2, 3])           # (1, 1280)
            score  = self._head(
                torch.cat([pooled, torch.zeros(1, MARKER_DIM).to(self.device)], dim=1)
            )
            score.backward()

            if self._fmap is None or self._grad is None:
                raise RuntimeError("Hooks did not fire")

            # Grad-CAM: weight channels by pooled gradient
            weights = self._grad.mean(dim=[0, 2, 3])    # (C,)
            cam     = (weights[:, None, None] * self._fmap.squeeze(0)).sum(dim=0)
            cam     = F.relu(cam).cpu().detach().numpy().astype(np.float32)
            cam_max = cam.max()
            if cam_max > 0:
                cam = cam / cam_max
            return cam

        except Exception as exc:  # noqa: BLE001
            logger.debug("Grad-CAM failed: %s", exc)
            return np.zeros((7, 7), dtype=np.float32)

    # ------------------------------------------------------------------
    def predict(
        self,
        image:      Any,
        htp_vector: list[int] | np.ndarray | None = None,
    ) -> DrawingRiskOutput:
        if htp_vector is None:
            htp_arr = np.zeros(MARKER_DIM, dtype=np.float32)
        else:
            htp_arr = np.array(htp_vector, dtype=np.float32).flatten()

        marker_result = self._marker_extractor.extract(htp_arr.astype(int).tolist())

        if not self._available or not self._feature_extractor.is_available:
            # Rule-based stub: scale burden_score to [0, 1]
            risk_score = float(np.clip(marker_result.burden_score / 100.0, 0.0, 1.0))
            return DrawingRiskOutput(
                risk_score        = risk_score,
                activated_markers = marker_result.present_markers,
                gradcam_map       = np.zeros((7, 7), dtype=np.float32),
                burden_score      = marker_result.burden_score,
                mode              = "stub",
            )

        try:
            import torch  # noqa: PLC0415

            pil_img    = self._feature_extractor._load_image(image)
            cnn_vec    = self._feature_extractor._cnn_embed(pil_img)
            combined   = np.concatenate([cnn_vec, htp_arr])
            tensor_in  = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logit = self._head(tensor_in)
            risk_score = float(torch.sigmoid(logit).item())

            # Grad-CAM requires its own forward pass with gradients
            img_tensor = self._feature_extractor._transform(pil_img).unsqueeze(0).to(self.device)
            gradcam    = self._gradcam_map(img_tensor)

            return DrawingRiskOutput(
                risk_score        = risk_score,
                activated_markers = marker_result.present_markers,
                gradcam_map       = gradcam,
                burden_score      = marker_result.burden_score,
                mode              = "neural",
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("DrawingRiskClassifier.predict() failed: %s", exc)
            risk_score = float(np.clip(marker_result.burden_score / 100.0, 0.0, 1.0))
            return DrawingRiskOutput(
                risk_score        = risk_score,
                activated_markers = marker_result.present_markers,
                gradcam_map       = np.zeros((7, 7), dtype=np.float32),
                burden_score      = marker_result.burden_score,
                mode              = "stub",
            )

    def save(self, path: str | Path) -> None:
        try:
            import torch  # noqa: PLC0415

            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._head.state_dict(), path)
            logger.info("DrawingRiskClassifier head saved to %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.error("save() failed: %s", exc)

    def load_head(self, path: str | Path) -> None:
        try:
            import torch  # noqa: PLC0415

            state = torch.load(path, map_location=self.device)
            self._head.load_state_dict(state)
            self._head.eval()
            logger.info("Loaded head weights from %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.error("load_head() failed: %s", exc)
