"""XAI module — explainability tools for multi-modal psychiatric screening models."""

from xai.shap_explainer import SHAPExplainer
from xai.report_generator import ClinicalReportGenerator, XAIReportGenerator
from xai.lime_explainer import LIMEExplainer

# gradcam.py imports torch at module level; guard so the package is still
# importable on machines without torch/CUDA installed.
try:
    from xai.gradcam import GradCAM
except ImportError:
    GradCAM = None  # type: ignore[assignment,misc]

from xai.attention_visualizer import AttentionVisualizer

__all__ = [
    "SHAPExplainer",
    "ClinicalReportGenerator",
    "XAIReportGenerator",
    "LIMEExplainer",
    "GradCAM",
    "AttentionVisualizer",
]
