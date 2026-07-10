"""API request/response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class FieldSource(str, Enum):
    VISUAL = "visual"
    MRZ = "mrz"
    LOOKUP = "lookup"
    NONE = "none"


class FieldValue(BaseModel):
    value: Optional[str] = None
    confidence: float = 0.0
    source: FieldSource = FieldSource.NONE
    confirmed: bool = False


class WarningCode(str, Enum):
    FIELD_NOT_FOUND = "FIELD_NOT_FOUND"
    PARTIAL_DOCUMENT = "PARTIAL_DOCUMENT"
    LOW_RESOLUTION = "LOW_RESOLUTION"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    ORIENTATION_CORRECTED = "ORIENTATION_CORRECTED"
    IMAGE_DOWNSCALED = "IMAGE_DOWNSCALED"


class Warning(BaseModel):
    code: WarningCode
    field: Optional[str] = None
    message: str


class MrzChecksums(BaseModel):
    number: bool = False
    birth_date: bool = False
    optional: bool = False


class MrzBlock(BaseModel):
    line1: str = ""
    line2: str = ""
    valid: bool = False
    checksums: MrzChecksums = Field(default_factory=MrzChecksums)


class DocumentBlock(BaseModel):
    type: Literal["passport_rf_internal"] = "passport_rf_internal"
    confidence: float = 0.0
    fields: dict[str, FieldValue] = Field(default_factory=dict)
    mrz: MrzBlock = Field(default_factory=MrzBlock)
    overall_confidence: float = 0.0
    warnings: list[Warning] = Field(default_factory=list)


class RecognizeSuccess(BaseModel):
    success: Literal[True] = True
    request_id: str
    processing_time_ms: int
    document: DocumentBlock


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[dict[str, Any]] = None


class RecognizeError(BaseModel):
    success: Literal[False] = False
    error: ErrorDetail
    request_id: str


class AsyncTaskAccepted(BaseModel):
    success: Literal[True] = True
    task_id: str
    request_id: str
    status: Literal["pending"] = "pending"
    poll_url: str


class TaskStatus(BaseModel):
    task_id: str
    request_id: str
    status: Literal["pending", "completed", "failed"]
    result: Optional[RecognizeSuccess | RecognizeError] = None


class LimitsResponse(BaseModel):
    client_id: str
    rate_limit_rps: float
    rate_limit_burst: int
    rate_limit_daily: int
    queue_max_depth: int


class TaskPayload(BaseModel):
    """Internal task metadata stored in Redis."""

    task_id: str
    request_id: str
    client_id: str
    api_key_id: str
    mode: Literal["sync", "async"] = "sync"
    webhook_url: Optional[str] = None
    min_quality: str = "normal"
    include_raw: bool = False
    enqueued_at: float = 0.0
    # Препроцессинг на API: downscale до max_image_side_px
    image_scaled: bool = False
    image_original_width: Optional[int] = None
    image_original_height: Optional[int] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
