"""Fusion module — attention-based multi-modal integration for psychiatric risk scoring."""

from modules.fusion.attention_fusion import (
    MultiModalAttentionFusion,
    AttentionFusion,       # backward-compat alias
    FusionInput,
    FusionOutput,
)
from modules.fusion.risk_stratifier import RiskStratifier

__all__ = [
    "MultiModalAttentionFusion",
    "AttentionFusion",
    "FusionInput",
    "FusionOutput",
    "RiskStratifier",
]
