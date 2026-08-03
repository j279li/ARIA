from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Point(BaseModel):
    x: int
    y: int


class TextRegion(BaseModel):
    id: str
    polygon: list[Point]
    bbox: tuple[int, int, int, int]
    render_bbox: tuple[int, int, int, int] | None = None
    source_text: str
    translated_text: str = ""
    font_size: int | None = Field(default=None, ge=8, le=200)
    confidence: float = Field(ge=0, le=100)
    detector_confidence: float | None = Field(default=None, ge=0, le=100)
    recognition_confidence: float | None = Field(default=None, ge=0, le=100)
    segmentation: list[list[Point]] = Field(default_factory=list)
    detector_metadata: dict[str, object] = Field(default_factory=dict)
    orientation: Literal["horizontal", "vertical"] = "horizontal"
    reading_order: int = 0


class ManualInpaintRegion(BaseModel):
    bbox: tuple[int, int, int, int]


class PageStatus(BaseModel):
    id: str
    filename: str
    status: Literal["queued", "processing", "complete", "failed"] = "queued"
    original_url: str
    cleaned_url: str | None = None
    output_url: str | None = None
    regions: list[TextRegion] = Field(default_factory=list)
    manual_inpaint_regions: list[ManualInpaintRegion] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "processing", "complete", "failed"] = "queued"
    created_at: datetime = Field(default_factory=utc_now)
    detector_provider: Literal["tesseract", "paddleocr"] = "tesseract"
    recognizer_provider: Literal["tesseract", "manga-ocr"] = "tesseract"
    translation_provider: Literal["deepl", "argos", "helsinki", "identity"] = "deepl"
    pages: list[PageStatus] = Field(default_factory=list)
    error: str | None = None


class PipelineOptions(BaseModel):
    ocr_lang: str = "jpn"
    min_confidence: float = Field(default=25, ge=-1, le=100)
    detector_provider: Literal["tesseract", "paddleocr"] = "tesseract"
    recognizer_provider: Literal["tesseract", "manga-ocr"] = "tesseract"
    translation_provider: Literal["deepl", "argos", "helsinki", "identity"] = "deepl"
    font_path: str | None = None
    mask_dilation: int = Field(default=5, ge=0, le=32)


class RenderRegionUpdate(BaseModel):
    id: str
    translated_text: str | None = None
    render_bbox: tuple[int, int, int, int] | None = None
    font_size: int | None = Field(default=None, ge=8, le=200)


class PageRenderRequest(BaseModel):
    regions: list[RenderRegionUpdate] = Field(default_factory=list)
    manual_inpaint_regions: list[ManualInpaintRegion] | None = None
