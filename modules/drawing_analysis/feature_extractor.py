"""EfficientNet-B0 CNN feature extraction from child drawings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CNN_DIM    = 1280   # EfficientNet-B0 penultimate feature dimension
MARKER_DIM = 20
TOTAL_DIM  = CNN_DIM + MARKER_DIM   # 1300


class DrawingFeatureExtractor:
    """EfficientNet-B0 feature extraction with HTP marker concatenation; zero-stub when torch unavailable."""

    def __init__(self, device: str = "cpu", offline: bool = False) -> None:
        self.device  = device
        self.offline = offline

        self._model:     Any  = None
        self._transform: Any  = None
        self._available: bool = False

        self._try_load()

    # ------------------------------------------------------------------
    def _try_load(self) -> None:
        try:
            import torch                                              # noqa: PLC0415
            import torchvision.models as tvm                         # noqa: PLC0415
            import torchvision.transforms as T                       # noqa: PLC0415

            weights_arg: Any
            if self.offline:
                weights_arg = None          # use cached or random init
            else:
                from torchvision.models import EfficientNet_B0_Weights  # noqa: PLC0415
                weights_arg = EfficientNet_B0_Weights.IMAGENET1K_V1

            model = tvm.efficientnet_b0(weights=weights_arg)
            # Remove the classifier head; keep feature extraction layers only
            model.classifier = torch.nn.Identity()
            model.to(self.device)
            model.eval()
            self._model = model

            self._transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225],
                ),
            ])
            self._available = True
            logger.info("DrawingFeatureExtractor ready (EfficientNet-B0, device=%s).", self.device)

        except ImportError as exc:
            logger.warning(
                "torchvision/torch not installed: %s — stub mode (zero CNN vectors).", exc
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DrawingFeatureExtractor load failed: %s — stub mode.", exc
            )

    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    def _load_image(self, source: Any) -> Any:
        from PIL import Image  # noqa: PLC0415

        if isinstance(source, (str, Path)):
            img = Image.open(source)
        elif isinstance(source, np.ndarray):
            img = Image.fromarray(source)
        else:
            img = source  # assume PIL already

        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _cnn_embed(self, pil_image: Any) -> np.ndarray:
        if not self._available:
            return np.zeros(CNN_DIM, dtype=np.float32)

        try:
            import torch  # noqa: PLC0415

            tensor = self._transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self._model(tensor)
            vec = out.squeeze(0).cpu().numpy().astype(np.float32)
            # EfficientNet-B0 classifier-removed output may still need avg-pool
            if vec.ndim > 1:
                vec = vec.mean(axis=tuple(range(1, vec.ndim)))
            return vec.flatten()[:CNN_DIM]

        except Exception as exc:  # noqa: BLE001
            logger.error("CNN embed failed: %s", exc)
            return np.zeros(CNN_DIM, dtype=np.float32)

    # ------------------------------------------------------------------
    def extract_cnn(self, source: Any) -> np.ndarray:
        pil_img = self._load_image(source)
        return self._cnn_embed(pil_img)

    def extract_combined(
        self,
        source: Any,
        htp_vector: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        cnn_vec = self.extract_cnn(source)

        if htp_vector is None:
            htp_arr = np.zeros(MARKER_DIM, dtype=np.float32)
        else:
            htp_arr = np.array(htp_vector, dtype=np.float32).flatten()
            if len(htp_arr) != MARKER_DIM:
                raise ValueError(f"htp_vector must be length {MARKER_DIM}, got {len(htp_arr)}")

        return np.concatenate([cnn_vec, htp_arr])

    def preprocess_image(self, source: Any) -> np.ndarray:
        """Return a normalised (224, 224, 3) float32 array — useful for inspection."""
        pil_img = self._load_image(source)
        arr = np.array(pil_img.resize((224, 224)), dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (arr - mean) / std
