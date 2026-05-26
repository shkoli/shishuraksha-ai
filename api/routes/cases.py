"""Case management API routes — CRUD for child psychiatric case records."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.schemas.models import CaseRecord
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/",
    response_model=CaseRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new case record",
)
async def create_case(
    case_id: str,
    clinician_id: str,
) -> CaseRecord:
    """Create a new pseudonymised case record for a child."""
    ...


@router.get(
    "/{case_id}",
    response_model=CaseRecord,
    summary="Retrieve a case record",
)
async def get_case(case_id: str) -> CaseRecord:
    """Retrieve the full case record including screening history."""
    ...


@router.get(
    "/",
    response_model=list[CaseRecord],
    summary="List cases for the authenticated clinician",
)
async def list_cases(
    clinician_id: str,
    status: str | None = Query(default=None),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0),
) -> list[CaseRecord]:
    """Return a paginated list of cases assigned to the clinician."""
    ...


@router.patch(
    "/{case_id}/status",
    response_model=CaseRecord,
    summary="Update case status",
)
async def update_case_status(case_id: str, new_status: str) -> CaseRecord:
    """Update the status field of a case (e.g., active → closed)."""
    ...


@router.delete(
    "/{case_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Pseudonymise and archive a case",
)
async def archive_case(case_id: str) -> None:
    """Pseudonymise and soft-delete a case record per data retention policy."""
    ...
