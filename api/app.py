"""Minimal FastAPI entry point for XAI-MPSCAP-BD.

Endpoints
---------
POST /screen   — run multi-modal fusion + risk stratification
GET  /cases    — return all cases from data/cases.json
GET  /health   — liveness probe
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# Resolve project root so imports work regardless of CWD
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.fusion.attention_fusion import FusionInput, MultiModalAttentionFusion
from modules.fusion.risk_stratifier import RiskStratifier

# ── Singletons ────────────────────────────────────────────────────────────────
_fusion = MultiModalAttentionFusion()
_stratifier = RiskStratifier()
_CASES_PATH = _ROOT / "data" / "cases.json"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="XAI-MPSCAP-BD API",
    description=(
        "Explainable AI Multi-modal Psychiatric Screening "
        "for Children and Adolescents — Bangladesh"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth stub ─────────────────────────────────────────────────────────────────
_bearer = HTTPBearer(auto_error=False)


def _jwt_stub(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    # TODO: verify token with python-jose once keys are provisioned
    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
        )
    return {"sub": "anonymous", "token": creds.credentials}


# ── Schemas ───────────────────────────────────────────────────────────────────
class ScreenRequest(BaseModel):
    questionnaire_scores: dict[str, float] = Field(
        default_factory=dict,
        example={"sdq_total": 22, "risk_score": 0.55},
    )
    text: str = Field(default="", example="শিশুটি বিষণ্ণ অনুভব করছে।")
    drawing_features: list[float] = Field(
        default_factory=list,
        description="1300-dim CNN + HTP marker vector (zeros if unavailable)",
    )
    facial_features: list[float] = Field(
        default_factory=list,
        description="32-dim action-unit feature vector (zeros if unavailable)",
    )
    existing_case_id: str | None = Field(
        default=None,
        description="Case ID to load previous screenings for longitudinal tracking",
    )


class ScreenResponse(BaseModel):
    risk_level: str
    risk_level_bn: str
    risk_score: float
    confidence: float
    referral: str
    referral_bn: str
    urgency: str
    recommended_actions: list[str]
    per_modality_scores: dict[str, float]
    modality_weights: dict[str, float]
    override_notes: list[str]
    trend: str | None = None
    previous_score: float | None = None
    score_change: float | None = None
    screening_count: int | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/screen", response_model=ScreenResponse, tags=["screening"])
def screen(
    body: ScreenRequest,
    _user: dict = Depends(_jwt_stub),
) -> ScreenResponse:
    drawing = np.array(body.drawing_features, dtype=np.float32) if body.drawing_features else np.zeros(1300, dtype=np.float32)
    facial  = np.array(body.facial_features,  dtype=np.float32) if body.facial_features  else np.zeros(32,   dtype=np.float32)

    # Text features: simple character-level hash projection to 768-dim
    # (BanglaBERT inference is deferred to the full pipeline)
    rng = np.random.RandomState(seed=abs(hash(body.text)) % (2 ** 31))
    text_vec = rng.randn(768).astype(np.float32) * 0.01

    fusion_input = FusionInput(
        questionnaire_features=body.questionnaire_scores,
        text_features=text_vec,
        drawing_features=drawing,
        facial_features=facial,
    )

    fusion_output = _fusion.fuse(fusion_input)
    result = _stratifier.stratify(fusion_output)

    # Longitudinal tracking
    trend = previous_score = score_change = screening_count = None
    if body.existing_case_id and _CASES_PATH.exists():
        with _CASES_PATH.open(encoding="utf-8") as fh:
            cases: list[dict] = json.load(fh)
        history = [
            e for c in cases
            if c.get("id") == body.existing_case_id
            for e in c.get("screeningHistory", [])
        ]
        if history:
            previous_score = history[-1].get("score")
            if previous_score is not None:
                new_score = result["risk_score"]
                score_change = round(new_score - previous_score, 4)
                if score_change > 0.10:
                    trend = "worsening"
                elif score_change < -0.10:
                    trend = "improving"
                else:
                    trend = "stable"
            screening_count = len(history) + 1

    return ScreenResponse(
        **result,
        trend=trend,
        previous_score=previous_score,
        score_change=score_change,
        screening_count=screening_count,
    )


@app.get("/cases", tags=["cases"])
def get_cases(_user: dict = Depends(_jwt_stub)) -> list[dict]:
    if not _CASES_PATH.exists():
        return []
    with _CASES_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
