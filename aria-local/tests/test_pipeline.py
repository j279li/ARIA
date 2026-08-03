from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from aria_local import pipeline
from aria_local.models import ManualInpaintRegion, PipelineOptions, Point, TextRegion
from aria_local.providers import (
    ArgosTranslationProvider,
    DetectedRegion,
    OCRResult,
    PaddleOCRDetector,
    group_detected_regions,
)


def test_argos_translation_reuses_duplicate_source_text() -> None:
    provider = object.__new__(ArgosTranslationProvider)

    class Translator:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def translate(self, text: str, source: str, target: str) -> str:
            assert source == "ja"
            assert target == "en"
            self.calls.append(text)
            return f"translated:{text}"

    translator = Translator()
    provider._translate = translator

    result = provider.translate(["同じ", "違う", "同じ"], PipelineOptions())

    assert result == ["translated:同じ", "translated:違う", "translated:同じ"]
    assert translator.calls == ["同じ", "違う"]


def test_process_page_writes_cleaned_and_translated_images(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "original.png"
    cleaned_path = tmp_path / "cleaned.png"
    output_path = tmp_path / "translated.png"
    Image.new("RGB", (160, 100), "white").save(source_path)

    monkeypatch.setattr(
        pipeline.pytesseract,
        "image_to_data",
        lambda *args, **kwargs: {
            "text": ["", "日本語"],
            "conf": ["-1", "95"],
            "block_num": [0, 1],
            "par_num": [0, 1],
            "line_num": [0, 1],
            "left": [0, 30],
            "top": [0, 25],
            "width": [0, 50],
            "height": [0, 20],
        },
    )

    regions, warnings = pipeline.process_page(
        source_path,
        cleaned_path,
        output_path,
        PipelineOptions(translation_provider="identity"),
    )

    assert not warnings
    assert len(regions) == 1
    assert regions[0].translated_text == "日本語"
    assert regions[0].detector_confidence is None
    assert regions[0].recognition_confidence == 95
    assert cleaned_path.exists()
    assert output_path.exists()


def test_process_page_accepts_detector_and_recognizer_providers(tmp_path: Path) -> None:
    source_path = tmp_path / "original.png"
    cleaned_path = tmp_path / "cleaned.png"
    output_path = tmp_path / "translated.png"
    Image.new("RGB", (160, 100), "white").save(source_path)

    class Detector:
        def detect(self, image_path: str, options: PipelineOptions) -> list[DetectedRegion]:
            assert image_path == str(source_path)
            return [
                DetectedRegion(
                    polygon=[
                        Point(x=30, y=25),
                        Point(x=80, y=25),
                        Point(x=80, y=45),
                        Point(x=30, y=45),
                    ],
                    bbox=(30, 25, 50, 20),
                    detector_confidence=88,
                )
            ]

    class Recognizer:
        def recognize(
            self,
            image_path: str,
            regions: list[DetectedRegion],
            options: PipelineOptions,
        ) -> list[OCRResult]:
            assert len(regions) == 1
            return [OCRResult(source_text="検出された文字")]

    regions, warnings = pipeline.process_page(
        source_path,
        cleaned_path,
        output_path,
        PipelineOptions(translation_provider="identity"),
        detector=Detector(),
        recognizer=Recognizer(),
    )

    assert not warnings
    assert regions[0].source_text == "検出された文字"
    assert regions[0].confidence == 88
    assert regions[0].detector_confidence == 88
    assert regions[0].recognition_confidence is None


def test_inpainting_mask_tracks_dark_text_inside_an_enclosed_bubble() -> None:
    image = np.full((160, 220, 3), 150, dtype=np.uint8)
    cv2.ellipse(image, (110, 80), (75, 55), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(image, (110, 80), (75, 55), 0, 0, 360, (20, 20, 20), 3)
    cv2.putText(image, "TEXT", (73, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    region = TextRegion(
        id="region-001",
        polygon=[Point(x=65, y=50), Point(x=155, y=50), Point(x=155, y=105), Point(x=65, y=105)],
        bbox=(65, 50, 90, 55),
        source_text="テキスト",
        confidence=90,
    )

    mask = pipeline._build_inpainting_mask(image, [region], dilation=1)

    assert cv2.countNonZero(mask) > 0
    assert mask[80, 80] > 0
    assert mask[15, 15] == 0


def test_inpainting_mask_finds_a_closed_bubble_on_a_white_page() -> None:
    image = np.full((160, 220, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (110, 80), (75, 55), 0, 0, 360, (20, 20, 20), 3)
    cv2.putText(image, "TEXT", (73, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    region = TextRegion(
        id="region-001",
        polygon=[Point(x=65, y=50), Point(x=155, y=50), Point(x=155, y=105), Point(x=65, y=105)],
        bbox=(65, 50, 90, 55),
        source_text="テキスト",
        confidence=90,
    )

    mask = pipeline._build_inpainting_mask(image, [region], dilation=1)

    assert cv2.countNonZero(mask) > 0


def test_inpainting_mask_rejects_text_on_unenclosed_art_background() -> None:
    image = np.full((160, 220, 3), 150, dtype=np.uint8)
    cv2.putText(image, "TITLE", (65, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    region = TextRegion(
        id="region-001",
        polygon=[Point(x=55, y=45), Point(x=165, y=45), Point(x=165, y=105), Point(x=55, y=105)],
        bbox=(55, 45, 110, 60),
        source_text="タイトル",
        confidence=90,
    )

    mask = pipeline._build_inpainting_mask(image, [region], dilation=5)

    assert cv2.countNonZero(mask) == 0


def test_inpainting_mask_handles_two_bubbles_in_one_ocr_region() -> None:
    image = np.full((140, 240, 3), 150, dtype=np.uint8)
    for center_x in (60, 180):
        cv2.ellipse(image, (center_x, 70), (45, 50), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (center_x, 70), (45, 50), 0, 0, 360, (20, 20, 20), 3)
    cv2.putText(image, "A", (52, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(image, "B", (172, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    region = TextRegion(
        id="region-001",
        polygon=[Point(x=25, y=25), Point(x=215, y=25), Point(x=215, y=115), Point(x=25, y=115)],
        bbox=(25, 25, 190, 90),
        segmentation=[
            [Point(x=40, y=45), Point(x=85, y=45), Point(x=85, y=100), Point(x=40, y=100)],
            [Point(x=160, y=45), Point(x=205, y=45), Point(x=205, y=100), Point(x=160, y=100)],
        ],
        source_text="A\nB",
        confidence=90,
    )

    mask = pipeline._build_inpainting_mask(image, [region], dilation=1)

    dark = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 100)
    assert cv2.countNonZero(cv2.bitwise_and(mask[45:100, 40:85], dark[45:100, 40:85])) > 0
    assert cv2.countNonZero(cv2.bitwise_and(mask[45:100, 160:205], dark[45:100, 160:205])) > 0


def test_inpainting_mask_rejects_a_bubble_open_to_the_page_edge() -> None:
    image = np.full((140, 240, 3), 150, dtype=np.uint8)
    image[25:115, 40:240] = 255
    cv2.line(image, (40, 25), (239, 25), (20, 20, 20), 3)
    cv2.line(image, (40, 25), (40, 115), (20, 20, 20), 3)
    cv2.line(image, (40, 115), (239, 115), (20, 20, 20), 3)
    cv2.putText(image, "SIDE", (145, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    region = TextRegion(
        id="region-001",
        polygon=[Point(x=130, y=45), Point(x=220, y=45), Point(x=220, y=100), Point(x=130, y=100)],
        bbox=(130, 45, 90, 55),
        source_text="サイド",
        confidence=90,
    )

    mask = pipeline._build_inpainting_mask(image, [region], dilation=5)

    assert cv2.countNonZero(mask) == 0


def test_rejected_detector_region_is_not_cleaned(tmp_path: Path) -> None:
    source_path = tmp_path / "original.png"
    cleaned_path = tmp_path / "cleaned.png"
    output_path = tmp_path / "translated.png"
    source_array = np.full((100, 160, 3), 255, dtype=np.uint8)
    source_array[30:40, 40:70] = 0
    image = Image.fromarray(source_array)
    image.save(source_path)

    class Detector:
        def detect(self, image_path: str, options: PipelineOptions) -> list[DetectedRegion]:
            return [
                DetectedRegion(
                    polygon=[
                        Point(x=30, y=25),
                        Point(x=80, y=25),
                        Point(x=80, y=45),
                        Point(x=30, y=45),
                    ],
                    bbox=(30, 25, 50, 20),
                )
            ]

    class Recognizer:
        def recognize(
            self,
            image_path: str,
            regions: list[DetectedRegion],
            options: PipelineOptions,
        ) -> list[OCRResult]:
            return [OCRResult(source_text="...")]

    pipeline.process_page(
        source_path,
        cleaned_path,
        output_path,
        PipelineOptions(translation_provider="identity"),
        detector=Detector(),
        recognizer=Recognizer(),
    )

    with Image.open(cleaned_path) as cleaned:
        assert np.array_equal(np.asarray(cleaned), np.asarray(image))


def test_clean_image_applies_manual_inpaint_regions() -> None:
    image = np.full((100, 160, 3), 255, dtype=np.uint8)
    cv2.putText(image, "MANUAL", (35, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cleaned = pipeline.clean_image(
        image,
        [],
        dilation=5,
        manual_inpaint_regions=[ManualInpaintRegion(bbox=(25, 25, 110, 45))],
    )

    assert int(cleaned[45, 55].mean()) > 200
    assert np.array_equal(cleaned[10, 10], image[10, 10])


def test_paddle_modern_results_preserve_polygon_and_metadata() -> None:
    detector = object.__new__(PaddleOCRDetector)

    regions = detector._regions_from_modern_results(
        [
            {
                "dt_polys": [[[10, 20], [70, 20], [70, 50], [10, 50]]],
                "rec_texts": ["日本語"],
                "rec_scores": [0.91],
            }
        ],
        PipelineOptions(min_confidence=25),
    )

    assert len(regions) == 1
    assert regions[0].bbox == (10, 20, 60, 30)
    assert regions[0].detector_confidence is None
    assert regions[0].recognition_confidence == 91
    assert regions[0].segmentation == [regions[0].polygon]
    assert regions[0].metadata == {"provider": "paddleocr", "api": "modern"}


def test_paddle_detection_only_results_preserve_detector_confidence() -> None:
    detector = object.__new__(PaddleOCRDetector)

    regions = detector._regions_from_modern_results(
        [
            {
                "dt_polys": [[[10, 20], [70, 20], [70, 50], [10, 50]]],
                "dt_scores": [0.91],
            }
        ],
        PipelineOptions(min_confidence=25),
    )

    assert len(regions) == 1
    assert regions[0].source_text is None
    assert regions[0].detector_confidence == 91
    assert regions[0].recognition_confidence is None


def test_grouped_regions_preserve_detector_confidence() -> None:
    regions = group_detected_regions(
        [
            DetectedRegion(
                polygon=[Point(x=10, y=10), Point(x=30, y=10), Point(x=30, y=50), Point(x=10, y=50)],
                bbox=(10, 10, 20, 40),
                orientation="vertical",
                detector_confidence=80,
            ),
            DetectedRegion(
                polygon=[Point(x=30, y=12), Point(x=50, y=12), Point(x=50, y=45), Point(x=30, y=45)],
                bbox=(30, 12, 20, 33),
                orientation="vertical",
                detector_confidence=90,
            ),
        ]
    )

    assert len(regions) == 1
    assert regions[0].detector_confidence == 85


def test_group_detected_regions_merges_bubble_lines_and_drops_non_japanese_noise() -> None:
    regions = group_detected_regions(
        [
            DetectedRegion(
                polygon=[Point(x=100, y=100), Point(x=120, y=100), Point(x=120, y=180), Point(x=100, y=180)],
                bbox=(100, 100, 20, 80),
                orientation="vertical",
                source_text="面白い",
                metadata={"provider": "paddleocr"},
            ),
            DetectedRegion(
                polygon=[Point(x=120, y=104), Point(x=140, y=104), Point(x=140, y=150), Point(x=120, y=150)],
                bbox=(120, 104, 20, 46),
                orientation="vertical",
                source_text="もの",
                metadata={"provider": "paddleocr"},
            ),
            DetectedRegion(
                polygon=[Point(x=500, y=200), Point(x=650, y=200), Point(x=650, y=220), Point(x=500, y=220)],
                bbox=(500, 200, 150, 20),
                orientation="horizontal",
                source_text="WE ACCEPT",
                metadata={"provider": "paddleocr"},
            ),
        ]
    )

    assert len(regions) == 1
    assert regions[0].bbox == (100, 100, 40, 80)
    assert regions[0].source_text == "面白い\nもの"
    assert len(regions[0].segmentation) == 2
    assert regions[0].metadata["group_size"] == 2


def test_render_text_fits_requested_font_and_clamps_render_box(tmp_path: Path) -> None:
    cleaned_path = tmp_path / "cleaned.png"
    output_path = tmp_path / "translated.png"
    Image.new("RGB", (160, 100), "white").save(cleaned_path)
    region = TextRegion(
        id="region-001",
        polygon=[],
        bbox=(20, 20, 20, 20),
        render_bbox=(-10, -10, 500, 500),
        source_text="日本語",
        translated_text="A short sentence",
        font_size=96,
        confidence=90,
    )

    pipeline.render_page(cleaned_path, output_path, [region])

    assert region.render_bbox == (0, 0, 160, 100)
    assert output_path.exists()
