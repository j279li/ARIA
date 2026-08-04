from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import aria_local.main as main_module
from aria_local.main import app
from aria_local.models import JobStatus, PageStatus, Point, TextRegion


def test_health() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_provider_catalog() -> None:
    response = TestClient(app).get("/api/providers")

    assert response.status_code == 200
    payload = response.json()
    assert {provider["id"] for provider in payload["detectors"]} == {
        "tesseract",
        "paddleocr",
    }
    assert {provider["id"] for provider in payload["recognizers"]} == {
        "tesseract",
        "manga-ocr",
    }
    assert {provider["id"] for provider in payload["translators"]} == {
        "deepl",
        "argos",
        "helsinki",
        "identity",
    }


def test_rejects_non_image_upload() -> None:
    response = TestClient(app).post(
        "/api/jobs",
        files={"files": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415


def test_rejects_unknown_provider() -> None:
    response = TestClient(app).post(
        "/api/jobs",
        files={"files": ("page.png", b"not really an image", "image/png")},
        data={"detector_provider": "unknown"},
    )

    assert response.status_code == 422


def test_rejects_paddleocr_without_manga_ocr() -> None:
    response = TestClient(app).post(
        "/api/jobs",
        files={"files": ("page.png", b"not really an image", "image/png")},
        data={
            "detector_provider": "paddleocr",
            "recognizer_provider": "tesseract",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "PaddleOCR must be paired with Manga OCR."


def test_rerender_updates_translation_without_processing(
    tmp_path: Path, monkeypatch
) -> None:
    job_id = "render-test"
    page_id = "page-001"
    page_dir = tmp_path / job_id / page_id
    page_dir.mkdir(parents=True)
    Image.new("RGB", (160, 100), "white").save(page_dir / "original.png")
    Image.new("RGB", (160, 100), "white").save(page_dir / "cleaned.png")
    region = TextRegion(
        id="region-001",
        polygon=[
            Point(x=30, y=25),
            Point(x=80, y=25),
            Point(x=80, y=45),
            Point(x=30, y=45),
        ],
        bbox=(30, 25, 50, 20),
        render_bbox=(15, 10, 80, 60),
        source_text="日本語",
        translated_text="old",
        confidence=90,
    )
    job = JobStatus(
        id=job_id,
        status="complete",
        pages=[
            PageStatus(
                id=page_id,
                filename="page.png",
                status="complete",
                original_url="/original",
                cleaned_url="/cleaned",
                output_url="/translated",
                regions=[region],
            )
        ],
    )
    manager = main_module.JobManager(tmp_path)
    manager.jobs[job_id] = job
    monkeypatch.setattr(main_module, "manager", manager)
    clean_calls: list[object] = []
    monkeypatch.setattr(
        main_module, "clean_page", lambda *args, **kwargs: clean_calls.append(args)
    )

    response = TestClient(app).post(
        f"/api/jobs/{job_id}/pages/{page_id}/render",
        json={
            "regions": [
                {
                    "id": "region-001",
                    "translated_text": "new",
                    "render_bbox": [10, 10, 100, 70],
                    "font_size": 18,
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["pages"][0]["regions"][0]["translated_text"] == "new"
    assert response.json()["pages"][0]["regions"][0]["font_size"] == 18
    assert response.json()["pages"][0]["regions"][0]["detector_metadata"][
        "manual_render_bbox"
    ]
    assert (
        manager.jobs[job_id]
        .pages[0]
        .output_url.startswith(
            f"/api/jobs/{job_id}/pages/{page_id}/artifacts/translated?v="
        )
    )
    assert not clean_calls
    assert (page_dir / "translated.png").exists()

    def fail_render(*args: object, **kwargs: object) -> None:
        raise RuntimeError("render failed")

    monkeypatch.setattr(main_module, "render_page", fail_render)
    failed_response = TestClient(app, raise_server_exceptions=False).post(
        f"/api/jobs/{job_id}/pages/{page_id}/render",
        json={"regions": [{"id": "region-001", "translated_text": "broken"}]},
    )

    assert failed_response.status_code == 500
    assert manager.jobs[job_id].pages[0].regions[0].translated_text == "new"
    assert not (page_dir / "translated.next.png").exists()


def test_rerender_applies_manual_inpaint_regions_and_refreshes_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    job_id = "manual-render-test"
    page_id = "page-001"
    page_dir = tmp_path / job_id / page_id
    page_dir.mkdir(parents=True)
    original = Image.new("RGB", (160, 100), "white")
    draw = ImageDraw.Draw(original)
    draw.rectangle((20, 20, 80, 50), outline="black")
    draw.rectangle((35, 30, 65, 40), fill="black")
    original.save(page_dir / "original.png")
    original.save(page_dir / "cleaned.png")
    region = TextRegion(
        id="region-001",
        polygon=[],
        bbox=(35, 30, 30, 10),
        render_bbox=(20, 20, 80, 40),
        source_text="日本語",
        translated_text="translation",
        confidence=90,
    )
    job = JobStatus(
        id=job_id,
        status="complete",
        pages=[
            PageStatus(
                id=page_id,
                filename="page.png",
                status="complete",
                original_url="/original",
                cleaned_url="/cleaned",
                output_url="/translated",
                regions=[region],
            )
        ],
    )
    manager = main_module.JobManager(tmp_path)
    manager.jobs[job_id] = job
    monkeypatch.setattr(main_module, "manager", manager)

    response = TestClient(app).post(
        f"/api/jobs/{job_id}/pages/{page_id}/render",
        json={"manual_inpaint_regions": [{"bbox": [35, 30, 30, 10]}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pages"][0]["manual_inpaint_regions"] == [{"bbox": [35, 30, 30, 10]}]
    assert "?v=" in payload["pages"][0]["cleaned_url"]
    assert "?v=" in payload["pages"][0]["output_url"]
    with Image.open(page_dir / "cleaned.png") as cleaned:
        assert cleaned.getpixel((45, 35))[0] > 200
