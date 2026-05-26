"""Pydantic request and response schemas for the screening API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ModalityData(BaseModel):
    """Raw multi-modal input data for one screening session."""

    questionnaire_responses: dict[str, Any] | None = Field(
        default=None, description="Mapping of item_id to raw response value"
    )
    narrative_text: str | None = Field(
        default=None, description="Free-text clinical narrative in Bengali or English"
    )
    drawing_image_b64: str | None = Field(
        default=None, description="Base64-encoded drawing image"
    )
    video_clip_b64: str | None = Field(
        default=None, description="Base64-encoded short facial video clip"
    )

    @field_validator("narrative_text")
    @classmethod
    def validate_text_not_empty(cls, v: str | None) -> str | None:
        if v is not None and len(v.strip()) == 0:
            raise ValueError("narrative_text must not be empty if provided")
        return v


class ScreeningRequest(BaseModel):
    """Request body for a new screening session."""

    case_id: str = Field(..., description="Pseudonymised case identifier")
    age: int = Field(..., ge=6, le=18, description="Child's age in years")
    gender: str = Field(..., description="Gender identity as self-reported")
    modalities: ModalityData
    clinician_id: str = Field(..., description="Authenticated clinician identifier")
    session_notes: str = ""


class ModalityContribution(BaseModel):
    """Per-modality attention weight in the fusion output."""

    modality: str
    weight: float
    available: bool


class ScreeningResponse(BaseModel):
    """Response for a completed screening session."""

    case_id: str
    session_id: str
    risk_level: RiskLevel
    risk_score: float
    confidence: float
    urgency_flag: bool
    modality_contributions: list[ModalityContribution]
    recommendations: list[str]
    explanation_url: str | None = None
    screened_at: datetime


class CaseRecord(BaseModel):
    """Full case record stored in the database."""

    case_id: str
    created_at: datetime
    screening_history: list[ScreeningResponse] = []
    clinician_id: str
    status: str = "active"


class ReportRequest(BaseModel):
    """Request to generate an XAI clinical report for a session."""

    session_id: str
    format: str = Field(default="pdf", pattern="^(pdf|json)$")
    include_shap: bool = True
    include_gradcam: bool = True
    include_attention: bool = True
    language: str = Field(default="bn", pattern="^(bn|en)$")


class ReportResponse(BaseModel):
    """Response containing a generated XAI report."""

    session_id: str
    report_url: str
    generated_at: datetime
    format: str
    size_bytes: int
