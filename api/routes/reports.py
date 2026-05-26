"""Report API routes — generate and download XAI clinical reports."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse

from api.schemas.models import ReportRequest, ReportResponse
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/",
    response_model=ReportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request generation of an XAI clinical report",
)
async def request_report(
    report_req: ReportRequest,
    background_tasks: BackgroundTasks,
) -> ReportResponse:
    """Trigger asynchronous XAI report generation for a screening session.

    - Compiles SHAP, LIME, Grad-CAM, and attention visualisations.
    - Renders a Bengali-localised PDF or JSON clinical report.
    - Stores the report and returns a download URL.
    """
    ...


@router.get(
    "/{session_id}",
    response_model=ReportResponse,
    summary="Get report status and download link",
)
async def get_report_status(session_id: str) -> ReportResponse:
    """Check the status of an ongoing report generation request."""
    ...


@router.get(
    "/{session_id}/download",
    summary="Download a generated XAI report file",
)
async def download_report(session_id: str) -> FileResponse:
    """Stream the generated PDF or JSON report file to the client."""
    ...
