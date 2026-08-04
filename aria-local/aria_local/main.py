from __future__ import annotations

import importlib.util
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .models import JobStatus, PageRenderRequest, PageStatus, PipelineOptions
from .pipeline import (
    PipelineError,
    clean_page,
    process_page,
    render_page,
    resolve_tesseract_command,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aria-local")

DATA_ROOT = Path(os.getenv("ARIA_DATA_DIR", ".aria-data"))
MAX_FILES = int(os.getenv("ARIA_MAX_FILES", "20"))
MAX_FILE_BYTES = int(os.getenv("ARIA_MAX_FILE_BYTES", str(20 * 1024 * 1024)))


class JobManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, JobStatus] = {}
        self.lock = threading.RLock()
        self.page_locks: dict[tuple[str, str], threading.Lock] = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aria-job")

    def _job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _status_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def _page_lock(self, job_id: str, page_id: str) -> threading.Lock:
        key = (job_id, page_id)
        with self.lock:
            return self.page_locks.setdefault(key, threading.Lock())

    def _persist(self, job: JobStatus) -> None:
        path = self._status_path(job.id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)

    def create_job(
        self,
        uploads: list[tuple[str, bytes]],
        *,
        detector_provider: Literal["tesseract", "paddleocr"],
        recognizer_provider: Literal["tesseract", "manga-ocr"],
        translation_provider: Literal["deepl", "argos", "helsinki", "identity"],
    ) -> JobStatus:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self._job_dir(job_id)
        pages: list[PageStatus] = []
        for index, (filename, contents) in enumerate(uploads, start=1):
            page_id = f"page-{index:03d}"
            page_dir = job_dir / page_id
            page_dir.mkdir(parents=True, exist_ok=True)
            suffix = Path(filename).suffix.lower() or ".bin"
            source_path = page_dir / f"original{suffix}"
            source_path.write_bytes(contents)
            pages.append(
                PageStatus(
                    id=page_id,
                    filename=filename,
                    original_url=f"/api/jobs/{job_id}/pages/{page_id}/artifacts/original",
                )
            )

        job = JobStatus(
            id=job_id,
            detector_provider=detector_provider,
            recognizer_provider=recognizer_provider,
            translation_provider=translation_provider,
            pages=pages,
        )
        with self.lock:
            self.jobs[job_id] = job
            self._persist(job)
        self.executor.submit(self.process_job, job_id)
        return job

    def get(self, job_id: str) -> JobStatus | None:
        with self.lock:
            if job_id in self.jobs:
                return self.jobs[job_id]
            path = self._status_path(job_id)
            if not path.exists():
                return None
            job = JobStatus.model_validate_json(path.read_text(encoding="utf-8"))
            self.jobs[job_id] = job
            return job

    def process_job(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            return
        with self.lock:
            job.status = "processing"
            self._persist(job)

        failed_pages = 0
        for page in job.pages:
            with self.lock:
                page.status = "processing"
                self._persist(job)
            try:
                self._process_page(job, page)
            except Exception as exc:  # Keep one broken page from killing the worker.
                failed_pages += 1
                logger.exception("Failed to process %s/%s", job.id, page.id)
                with self.lock:
                    page.status = "failed"
                    page.error = str(exc)
                    self._persist(job)

        with self.lock:
            job.status = "failed" if failed_pages == len(job.pages) else "complete"
            if failed_pages:
                job.error = f"{failed_pages} page(s) failed to process."
            self._persist(job)

    def _process_page(self, job: JobStatus, page: PageStatus) -> None:
        page_dir = self._job_dir(job.id) / page.id
        source_candidates = [candidate for candidate in page_dir.glob("original.*")]
        if not source_candidates:
            raise PipelineError("Original image is missing")
        source_path = source_candidates[0]
        cleaned_path = page_dir / "cleaned.png"
        output_path = page_dir / "translated.png"
        options = PipelineOptions(
            ocr_lang=os.getenv("ARIA_OCR_LANG", "jpn"),
            min_confidence=float(os.getenv("ARIA_OCR_MIN_CONFIDENCE", "25")),
            detector_provider=job.detector_provider,
            recognizer_provider=job.recognizer_provider,
            translation_provider=job.translation_provider,
            font_path=os.getenv("ARIA_FONT_PATH") or None,
            mask_dilation=int(os.getenv("ARIA_MASK_DILATION", "5")),
        )
        regions, warnings = process_page(
            source_path, cleaned_path, output_path, options
        )
        with self.lock:
            page.status = "complete"
            page.cleaned_url = f"/api/jobs/{job.id}/pages/{page.id}/artifacts/cleaned"
            page.output_url = f"/api/jobs/{job.id}/pages/{page.id}/artifacts/translated"
            page.regions = regions
            page.warnings = warnings
            self._persist(job)

    def rerender_page(
        self, job_id: str, page_id: str, request: PageRenderRequest
    ) -> JobStatus:
        with self._page_lock(job_id, page_id):
            return self._rerender_page(job_id, page_id, request)

    def _rerender_page(
        self, job_id: str, page_id: str, request: PageRenderRequest
    ) -> JobStatus:
        job = self.get(job_id)
        if job is None:
            raise LookupError("Job not found")
        page = next(
            (candidate for candidate in job.pages if candidate.id == page_id), None
        )
        if page is None:
            raise LookupError("Page not found")
        if page.status != "complete":
            raise ValueError("Only completed pages can be rerendered")

        updated_page = page.model_copy(deep=True)
        page_dir = self._job_dir(job.id) / updated_page.id
        cleaned_path = page_dir / "cleaned.png"
        output_path = page_dir / "translated.png"
        next_cleaned_path = page_dir / "cleaned.next.png"
        next_output_path = page_dir / "translated.next.png"
        if not cleaned_path.exists():
            raise ValueError("Cleaned image is missing")

        regions = {region.id: region for region in updated_page.regions}
        for update in request.regions:
            region = regions.get(update.id)
            if region is None:
                raise ValueError(f"Unknown text region: {update.id}")
            if "translated_text" in update.model_fields_set:
                region.translated_text = update.translated_text or ""
            if "render_bbox" in update.model_fields_set:
                region.render_bbox = update.render_bbox
                region.detector_metadata["manual_render_bbox"] = (
                    update.render_bbox is not None
                )
            if "font_size" in update.model_fields_set:
                region.font_size = update.font_size

        manual_regions_changed = (
            request.manual_inpaint_regions is not None
            and request.manual_inpaint_regions != page.manual_inpaint_regions
        )
        if request.manual_inpaint_regions is not None:
            updated_page.manual_inpaint_regions = request.manual_inpaint_regions

        try:
            render_source = cleaned_path
            if manual_regions_changed:
                source_candidates = list(page_dir.glob("original.*"))
                if not source_candidates:
                    raise ValueError("Original image is missing")
                clean_page(
                    source_candidates[0],
                    next_cleaned_path,
                    updated_page.regions,
                    int(os.getenv("ARIA_MASK_DILATION", "5")),
                    updated_page.manual_inpaint_regions,
                )
                render_source = next_cleaned_path
            render_page(
                render_source,
                next_output_path,
                updated_page.regions,
                os.getenv("ARIA_FONT_PATH") or None,
            )
            if manual_regions_changed:
                next_cleaned_path.replace(cleaned_path)
            next_output_path.replace(output_path)
        finally:
            next_cleaned_path.unlink(missing_ok=True)
            next_output_path.unlink(missing_ok=True)

        cache_version = uuid.uuid4().hex[:8]
        if manual_regions_changed:
            updated_page.cleaned_url = (
                f"/api/jobs/{job.id}/pages/{page.id}/artifacts/cleaned"
                f"?v={cache_version}"
            )
        updated_page.output_url = (
            f"/api/jobs/{job.id}/pages/{page.id}/artifacts/translated?v={cache_version}"
        )
        with self.lock:
            page_index = job.pages.index(page)
            job.pages[page_index] = updated_page
            self._persist(job)
        return job


manager = JobManager(DATA_ROOT)


app = FastAPI(title="ARIA Local", version="0.1.0")


allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ARIA_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/providers")
def providers() -> dict[str, list[dict[str, object]]]:
    tesseract_path = resolve_tesseract_command()
    return {
        "detectors": [
            {
                "id": "tesseract",
                "label": "Tesseract",
                "available": bool(tesseract_path),
            },
            {
                "id": "paddleocr",
                "label": "PaddleOCR",
                "available": importlib.util.find_spec("paddleocr") is not None
                and importlib.util.find_spec("paddle") is not None,
            },
        ],
        "recognizers": [
            {
                "id": "tesseract",
                "label": "Tesseract",
                "available": bool(tesseract_path),
            },
            {
                "id": "manga-ocr",
                "label": "Manga OCR",
                "available": importlib.util.find_spec("manga_ocr") is not None,
            },
        ],
        "translators": [
            {
                "id": "deepl",
                "label": "DeepL",
                "available": bool(os.getenv("DEEPL_API_KEY")),
            },
            {
                "id": "argos",
                "label": "Argos Translate",
                "available": importlib.util.find_spec("argostranslate") is not None,
            },
            {
                "id": "helsinki",
                "label": "Helsinki OPUS-MT",
                "available": importlib.util.find_spec("transformers") is not None
                and importlib.util.find_spec("sentencepiece") is not None,
            },
            {"id": "identity", "label": "Source text", "available": True},
        ],
    }


@app.post("/api/jobs", response_model=JobStatus, status_code=202)
async def create_job(
    files: Annotated[list[UploadFile], File()],
    detector_provider: Annotated[
        Literal["tesseract", "paddleocr"] | None, Form()
    ] = None,
    recognizer_provider: Annotated[
        Literal["tesseract", "manga-ocr"] | None, Form()
    ] = None,
    translation_provider: Annotated[
        Literal["deepl", "argos", "helsinki", "identity"] | None, Form()
    ] = None,
) -> JobStatus:
    if not files or len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400, detail=f"Upload between 1 and {MAX_FILES} images."
        )

    selected_detector = detector_provider or os.getenv(
        "ARIA_TEXT_DETECTOR", "tesseract"
    )
    selected_recognizer = recognizer_provider or os.getenv(
        "ARIA_TEXT_RECOGNIZER", "tesseract"
    )
    selected_translator = translation_provider or os.getenv(
        "ARIA_TRANSLATION_PROVIDER", "deepl"
    )
    if selected_detector == "paddleocr" and selected_recognizer != "manga-ocr":
        raise HTTPException(
            status_code=400,
            detail="PaddleOCR must be paired with Manga OCR.",
        )

    uploads: list[tuple[str, bytes]] = []
    for upload in files:
        if upload.content_type and not upload.content_type.startswith("image/"):
            raise HTTPException(
                status_code=415, detail=f"Unsupported file type: {upload.content_type}"
            )
        contents = await upload.read(MAX_FILE_BYTES + 1)
        if not contents:
            raise HTTPException(
                status_code=400, detail="Uploaded images cannot be empty."
            )
        if len(contents) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Each image must be smaller than {MAX_FILE_BYTES // (1024 * 1024)} MiB.",
            )
        uploads.append((upload.filename or "upload", contents))

    return manager.create_job(
        uploads,
        detector_provider=selected_detector,
        recognizer_provider=selected_recognizer,
        translation_provider=selected_translator,
    )


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/jobs/{job_id}/pages/{page_id}/render", response_model=JobStatus)
def rerender_page(job_id: str, page_id: str, request: PageRenderRequest) -> JobStatus:
    try:
        return manager.rerender_page(job_id, page_id, request)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/pages/{page_id}/artifacts/{artifact}")
def get_artifact(job_id: str, page_id: str, artifact: str) -> FileResponse:
    job = manager.get(job_id)
    if job is None or not any(page.id == page_id for page in job.pages):
        raise HTTPException(status_code=404, detail="Page not found")
    if artifact not in {"original", "cleaned", "translated"}:
        raise HTTPException(status_code=404, detail="Artifact not found")

    page_dir = manager._job_dir(job_id) / page_id
    if artifact == "original":
        matches = list(page_dir.glob("original.*"))
        path = matches[0] if matches else None
    else:
        path = page_dir / f"{artifact}.png"
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not ready")
    return FileResponse(path)
