from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import aria_local.providers as providers_module
from aria_local import pipeline
from aria_local.bubbles import (
    WhiteBubbleLocator,
    _largest_centered_rectangle,
    polygon_geometry_mask,
)
from aria_local.models import ManualInpaintRegion, PipelineOptions, Point, TextRegion
from aria_local.providers import (
    ArgosTranslationProvider,
    DetectedRegion,
    HelsinkiTranslationProvider,
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


def test_helsinki_translation_deduplicates_model_inputs(monkeypatch) -> None:
    provider = HelsinkiTranslationProvider()

    class Tokenizer:
        def __init__(self) -> None:
            self.inputs: list[str] = []

        def __call__(self, texts: list[str], **kwargs) -> dict[str, object]:
            self.inputs = texts
            return {}

        def batch_decode(self, outputs: object, **kwargs) -> list[str]:
            return [f"translated:{text}" for text in self.inputs]

    class Model:
        def generate(self, **kwargs) -> object:
            return object()

    tokenizer = Tokenizer()
    monkeypatch.setattr(
        providers_module, "_get_helsinki_model", lambda: (Model(), tokenizer)
    )

    result = provider.translate(["同じ", "違う", "同じ"], PipelineOptions())

    assert tokenizer.inputs == ["同じ", "違う"]
    assert result == ["translated:同じ", "translated:違う", "translated:同じ"]


def test_process_page_writes_cleaned_and_translated_images(
    tmp_path: Path, monkeypatch
) -> None:
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
        def detect(
            self, image_path: str, options: PipelineOptions
        ) -> list[DetectedRegion]:
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
        polygon=[
            Point(x=65, y=50),
            Point(x=155, y=50),
            Point(x=155, y=105),
            Point(x=65, y=105),
        ],
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
        polygon=[
            Point(x=65, y=50),
            Point(x=155, y=50),
            Point(x=155, y=105),
            Point(x=65, y=105),
        ],
        bbox=(65, 50, 90, 55),
        source_text="テキスト",
        confidence=90,
    )

    mask = pipeline._build_inpainting_mask(image, [region], dilation=1)

    assert cv2.countNonZero(mask) > 0


def test_inpainting_mask_reuses_persisted_bubble_contour(monkeypatch) -> None:
    image = np.full((160, 220, 3), 150, dtype=np.uint8)
    cv2.ellipse(image, (110, 80), (75, 55), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(image, (110, 80), (75, 55), 0, 0, 360, (20, 20, 20), 3)
    cv2.putText(image, "TEXT", (73, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    contour = [
        (int(point[0]), int(point[1]))
        for point in cv2.ellipse2Poly((110, 80), (72, 52), 0, 0, 360, 10)
    ]
    polygon = [
        Point(x=65, y=50),
        Point(x=155, y=50),
        Point(x=155, y=105),
        Point(x=65, y=105),
    ]
    region = TextRegion(
        id="region-001",
        polygon=polygon,
        bbox=(65, 50, 90, 55),
        segmentation=[polygon],
        source_text="テキスト",
        confidence=90,
        detector_metadata={
            "bubble_checked": True,
            "bubble_contour": contour,
        },
    )

    def unexpected_locator(image: np.ndarray) -> None:
        raise AssertionError("bubble locator should not be rebuilt")

    monkeypatch.setattr(pipeline, "WhiteBubbleLocator", unexpected_locator)

    mask = pipeline._build_inpainting_mask(image, [region], dilation=1)

    assert cv2.countNonZero(mask) > 0


def test_adaptive_bubble_detection_cleans_off_white_bubble() -> None:
    image = np.full((180, 240, 3), 175, dtype=np.uint8)
    cv2.ellipse(image, (120, 90), (80, 60), 0, 0, 360, (205, 205, 205), -1)
    cv2.ellipse(image, (120, 90), (80, 60), 0, 0, 360, (20, 20, 20), 3)
    cv2.putText(image, "TEXT", (82, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    polygon = [
        Point(x=75, y=60),
        Point(x=165, y=60),
        Point(x=165, y=120),
        Point(x=75, y=120),
    ]
    region = TextRegion(
        id="region-001",
        polygon=polygon,
        bbox=(75, 60, 90, 60),
        source_text="テキスト",
        confidence=90,
    )

    mask = pipeline._build_inpainting_mask(image, [region], dilation=1)
    cleaned = pipeline.clean_image(image, [region], dilation=1)

    assert cv2.countNonZero(mask) > 0
    assert (
        cv2.countNonZero(mask)
        < cv2.countNonZero(polygon_geometry_mask(image.shape, [polygon])) * 0.5
    )
    assert 190 <= float(cleaned[mask > 0].mean()) <= 220


def test_adaptive_bubble_detection_rejects_text_without_outline() -> None:
    image = np.full((180, 240, 3), 175, dtype=np.uint8)
    cv2.ellipse(image, (120, 90), (80, 60), 0, 0, 360, (205, 205, 205), -1)
    cv2.putText(image, "TEXT", (82, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    polygon = [
        Point(x=75, y=60),
        Point(x=165, y=60),
        Point(x=165, y=120),
        Point(x=75, y=120),
    ]
    region = TextRegion(
        id="region-001",
        polygon=polygon,
        bbox=(75, 60, 90, 60),
        source_text="テキスト",
        confidence=90,
    )

    mask = pipeline._build_inpainting_mask(image, [region], dilation=1)

    assert cv2.countNonZero(mask) == 0


def test_adaptive_bubble_detection_rejects_borderless_patch_on_dark_art() -> None:
    image = np.full((180, 240, 3), 50, dtype=np.uint8)
    cv2.ellipse(image, (120, 90), (80, 60), 0, 0, 360, (215, 215, 215), -1)
    cv2.putText(image, "TEXT", (82, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    polygon = [
        Point(x=75, y=60),
        Point(x=165, y=60),
        Point(x=165, y=120),
        Point(x=75, y=120),
    ]
    region = TextRegion(
        id="region-001",
        polygon=polygon,
        bbox=(75, 60, 90, 60),
        source_text="テキスト",
        confidence=90,
    )

    mask = pipeline._build_inpainting_mask(image, [region], dilation=1)

    assert cv2.countNonZero(mask) == 0


def test_adaptive_bubble_detection_supports_two_light_tones() -> None:
    image = np.full((180, 320, 3), 175, dtype=np.uint8)
    for center_x, tone in ((80, 205), (240, 220)):
        cv2.ellipse(
            image,
            (center_x, 90),
            (60, 55),
            0,
            0,
            360,
            (tone, tone, tone),
            -1,
        )
        cv2.ellipse(image, (center_x, 90), (60, 55), 0, 0, 360, (20, 20, 20), 3)

    left_polygon = [
        Point(x=55, y=65),
        Point(x=105, y=65),
        Point(x=105, y=115),
        Point(x=55, y=115),
    ]
    right_polygon = [
        Point(x=215, y=65),
        Point(x=265, y=65),
        Point(x=265, y=115),
        Point(x=215, y=115),
    ]
    locator = WhiteBubbleLocator(image)

    left = locator.find(polygon_geometry_mask(image.shape, [left_polygon]))
    right = locator.find(polygon_geometry_mask(image.shape, [right_polygon]))

    assert left is not None
    assert right is not None
    assert 195 <= left.fill_tone <= 215
    assert 210 <= right.fill_tone <= 225


def test_persisted_contour_relocalizes_uncovered_segmentation() -> None:
    image = np.full((150, 260, 3), 150, dtype=np.uint8)
    for center_x in (65, 195):
        cv2.ellipse(image, (center_x, 75), (50, 55), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (center_x, 75), (50, 55), 0, 0, 360, (20, 20, 20), 3)
    cv2.putText(image, "A", (55, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(image, "B", (185, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    left_polygon = [
        Point(x=45, y=50),
        Point(x=85, y=50),
        Point(x=85, y=100),
        Point(x=45, y=100),
    ]
    right_polygon = [
        Point(x=175, y=50),
        Point(x=215, y=50),
        Point(x=215, y=100),
        Point(x=175, y=100),
    ]
    left_contour = [
        (int(point[0]), int(point[1]))
        for point in cv2.ellipse2Poly((65, 75), (47, 52), 0, 0, 360, 10)
    ]
    region = TextRegion(
        id="region-001",
        polygon=left_polygon,
        bbox=(45, 50, 170, 50),
        segmentation=[left_polygon, right_polygon],
        source_text="A B",
        confidence=90,
        detector_metadata={
            "bubble_checked": True,
            "bubble_contour": left_contour,
        },
    )

    mask = pipeline._build_inpainting_mask(image, [region], dilation=1)

    assert cv2.countNonZero(mask[50:100, 45:85]) > 0
    assert cv2.countNonZero(mask[50:100, 175:215]) > 0


def test_automatic_cleanup_does_not_mask_bubble_outline() -> None:
    image = np.full((160, 220, 3), 150, dtype=np.uint8)
    cv2.ellipse(image, (110, 80), (75, 55), 0, 0, 360, (255, 255, 255), -1)
    outline = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(outline, (110, 80), (75, 55), 0, 0, 360, 255, 3)
    image[outline > 0] = (20, 20, 20)
    cv2.putText(image, "TEXT", (73, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    polygon = [
        Point(x=34, y=45),
        Point(x=155, y=45),
        Point(x=155, y=110),
        Point(x=34, y=110),
    ]
    region = TextRegion(
        id="region-001",
        polygon=polygon,
        bbox=(34, 45, 121, 65),
        source_text="テキスト",
        confidence=90,
    )

    mask = pipeline._build_inpainting_mask(image, [region], dilation=3)

    assert cv2.countNonZero(mask) > 0
    assert cv2.countNonZero(cv2.bitwise_and(mask, outline)) == 0


def test_inpainting_mask_rejects_text_on_unenclosed_art_background() -> None:
    image = np.full((160, 220, 3), 150, dtype=np.uint8)
    cv2.putText(image, "TITLE", (65, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    region = TextRegion(
        id="region-001",
        polygon=[
            Point(x=55, y=45),
            Point(x=165, y=45),
            Point(x=165, y=105),
            Point(x=55, y=105),
        ],
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
        polygon=[
            Point(x=25, y=25),
            Point(x=215, y=25),
            Point(x=215, y=115),
            Point(x=25, y=115),
        ],
        bbox=(25, 25, 190, 90),
        segmentation=[
            [
                Point(x=40, y=45),
                Point(x=85, y=45),
                Point(x=85, y=100),
                Point(x=40, y=100),
            ],
            [
                Point(x=160, y=45),
                Point(x=205, y=45),
                Point(x=205, y=100),
                Point(x=160, y=100),
            ],
        ],
        source_text="A\nB",
        confidence=90,
    )

    mask = pipeline._build_inpainting_mask(image, [region], dilation=1)

    dark = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 100)
    assert (
        cv2.countNonZero(cv2.bitwise_and(mask[45:100, 40:85], dark[45:100, 40:85])) > 0
    )
    assert (
        cv2.countNonZero(cv2.bitwise_and(mask[45:100, 160:205], dark[45:100, 160:205]))
        > 0
    )


def test_inpainting_mask_rejects_a_bubble_open_to_the_page_edge() -> None:
    image = np.full((140, 240, 3), 150, dtype=np.uint8)
    image[25:115, 40:240] = 255
    cv2.line(image, (40, 25), (239, 25), (20, 20, 20), 3)
    cv2.line(image, (40, 25), (40, 115), (20, 20, 20), 3)
    cv2.line(image, (40, 115), (239, 115), (20, 20, 20), 3)
    cv2.putText(image, "SIDE", (145, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    region = TextRegion(
        id="region-001",
        polygon=[
            Point(x=130, y=45),
            Point(x=220, y=45),
            Point(x=220, y=100),
            Point(x=130, y=100),
        ],
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
        def detect(
            self, image_path: str, options: PipelineOptions
        ) -> list[DetectedRegion]:
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
                polygon=[
                    Point(x=10, y=10),
                    Point(x=30, y=10),
                    Point(x=30, y=50),
                    Point(x=10, y=50),
                ],
                bbox=(10, 10, 20, 40),
                orientation="vertical",
                detector_confidence=80,
            ),
            DetectedRegion(
                polygon=[
                    Point(x=30, y=12),
                    Point(x=50, y=12),
                    Point(x=50, y=45),
                    Point(x=30, y=45),
                ],
                bbox=(30, 12, 20, 33),
                orientation="vertical",
                detector_confidence=90,
            ),
        ]
    )

    assert len(regions) == 1
    assert regions[0].detector_confidence == 85


def test_group_detected_regions_merges_bubble_lines_and_drops_non_japanese_noise() -> (
    None
):
    regions = group_detected_regions(
        [
            DetectedRegion(
                polygon=[
                    Point(x=100, y=100),
                    Point(x=120, y=100),
                    Point(x=120, y=180),
                    Point(x=100, y=180),
                ],
                bbox=(100, 100, 20, 80),
                orientation="vertical",
                source_text="面白い",
                metadata={"provider": "paddleocr"},
            ),
            DetectedRegion(
                polygon=[
                    Point(x=120, y=104),
                    Point(x=140, y=104),
                    Point(x=140, y=150),
                    Point(x=120, y=150),
                ],
                bbox=(120, 104, 20, 46),
                orientation="vertical",
                source_text="もの",
                metadata={"provider": "paddleocr"},
            ),
            DetectedRegion(
                polygon=[
                    Point(x=500, y=200),
                    Point(x=650, y=200),
                    Point(x=650, y=220),
                    Point(x=500, y=220),
                ],
                bbox=(500, 200, 150, 20),
                orientation="horizontal",
                source_text="WE ACCEPT",
                metadata={"provider": "paddleocr"},
            ),
        ]
    )

    assert len(regions) == 1
    assert regions[0].bbox == (100, 100, 40, 80)
    assert regions[0].source_text == "もの\n面白い"
    assert len(regions[0].segmentation) == 2
    assert regions[0].metadata["group_size"] == 2


def test_group_detected_regions_does_not_merge_adjacent_bubbles() -> None:
    image = np.full((140, 240, 3), 150, dtype=np.uint8)
    for center_x in (60, 180):
        cv2.ellipse(image, (center_x, 70), (45, 50), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (center_x, 70), (45, 50), 0, 0, 360, (20, 20, 20), 3)

    first_polygon = [
        Point(x=60, y=45),
        Point(x=95, y=45),
        Point(x=95, y=95),
        Point(x=60, y=95),
    ]
    second_polygon = [
        Point(x=145, y=45),
        Point(x=180, y=45),
        Point(x=180, y=95),
        Point(x=145, y=95),
    ]
    regions = group_detected_regions(
        [
            DetectedRegion(
                polygon=first_polygon,
                bbox=(60, 45, 35, 50),
                orientation="vertical",
                source_text="左",
                segmentation=[first_polygon],
                metadata={"provider": "paddleocr"},
            ),
            DetectedRegion(
                polygon=second_polygon,
                bbox=(145, 45, 35, 50),
                orientation="vertical",
                source_text="右",
                segmentation=[second_polygon],
                metadata={"provider": "paddleocr"},
            ),
        ],
        image=image,
    )

    assert len(regions) == 2
    assert {region.source_text for region in regions} == {"左", "右"}
    assert all("bubble_bbox" in region.metadata for region in regions)
    first_bubble, second_bubble = sorted(
        (region.metadata["bubble_bbox"] for region in regions),
        key=lambda bbox: bbox[0],
    )
    assert isinstance(first_bubble, tuple)
    assert isinstance(second_bubble, tuple)
    assert first_bubble[0] + first_bubble[2] < second_bubble[0]


def test_group_detected_regions_separates_bubbles_joined_by_scan_gap() -> None:
    image = np.full((220, 240, 3), 150, dtype=np.uint8)
    for center_x in (60, 180):
        cv2.ellipse(image, (center_x, 110), (45, 50), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (center_x, 110), (45, 50), 0, 0, 360, (20, 20, 20), 3)
    cv2.rectangle(image, (103, 108), (137, 112), (255, 255, 255), -1)

    light_mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 225, 255)
    _, labels = cv2.connectedComponents(light_mask)
    assert labels[110, 60] == labels[110, 180]

    first_polygon = [
        Point(x=60, y=85),
        Point(x=95, y=85),
        Point(x=95, y=135),
        Point(x=60, y=135),
    ]
    second_polygon = [
        Point(x=145, y=85),
        Point(x=180, y=85),
        Point(x=180, y=135),
        Point(x=145, y=135),
    ]
    regions = group_detected_regions(
        [
            DetectedRegion(
                polygon=first_polygon,
                bbox=(60, 85, 35, 50),
                orientation="vertical",
                source_text="左",
                segmentation=[first_polygon],
                metadata={"provider": "paddleocr"},
            ),
            DetectedRegion(
                polygon=second_polygon,
                bbox=(145, 85, 35, 50),
                orientation="vertical",
                source_text="右",
                segmentation=[second_polygon],
                metadata={"provider": "paddleocr"},
            ),
        ],
        image=image,
    )

    assert len(regions) == 2
    assert all("bubble_bbox" in region.metadata for region in regions)


def test_nearby_text_outside_bubble_is_not_attached() -> None:
    image = np.full((180, 260, 3), 150, dtype=np.uint8)
    cv2.ellipse(image, (110, 90), (75, 60), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(image, (110, 90), (75, 60), 0, 0, 360, (20, 20, 20), 3)
    inside_polygon = [
        Point(x=155, y=60),
        Point(x=175, y=60),
        Point(x=175, y=120),
        Point(x=155, y=120),
    ]
    outside_polygon = [
        Point(x=185, y=60),
        Point(x=205, y=60),
        Point(x=205, y=120),
        Point(x=185, y=120),
    ]

    regions = group_detected_regions(
        [
            DetectedRegion(
                polygon=inside_polygon,
                bbox=(155, 60, 20, 60),
                orientation="vertical",
                source_text="内",
                segmentation=[inside_polygon],
                metadata={"provider": "paddleocr"},
            ),
            DetectedRegion(
                polygon=outside_polygon,
                bbox=(185, 60, 20, 60),
                orientation="vertical",
                source_text="外",
                segmentation=[outside_polygon],
                metadata={"provider": "paddleocr"},
            ),
        ],
        image=image,
    )

    assert len(regions) == 2
    assert sum("bubble_bbox" in region.metadata for region in regions) == 1


def test_geometry_fallback_does_not_merge_transitive_chain() -> None:
    candidates = [
        DetectedRegion(
            polygon=[],
            bbox=(10, 10, 20, 60),
            orientation="vertical",
            source_text="一",
        ),
        DetectedRegion(
            polygon=[],
            bbox=(45, 10, 20, 60),
            orientation="vertical",
            source_text="二",
        ),
        DetectedRegion(
            polygon=[],
            bbox=(80, 10, 20, 60),
            orientation="vertical",
            source_text="三",
        ),
    ]

    regions = group_detected_regions(candidates)
    reversed_regions = group_detected_regions(list(reversed(candidates)))

    def memberships(grouped: list[DetectedRegion]) -> list[tuple[str, ...]]:
        return sorted(
            tuple(sorted((region.source_text or "").splitlines())) for region in grouped
        )

    assert len(regions) == 2
    assert max(int(region.metadata.get("group_size", 1)) for region in regions) == 2
    assert memberships(regions) == memberships(reversed_regions)


def test_group_detected_regions_uses_shared_bubble_over_distance() -> None:
    image = np.full((240, 240, 3), 150, dtype=np.uint8)
    cv2.ellipse(image, (120, 120), (85, 70), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(image, (120, 120), (85, 70), 0, 0, 360, (20, 20, 20), 3)
    first_polygon = [
        Point(x=70, y=85),
        Point(x=90, y=85),
        Point(x=90, y=150),
        Point(x=70, y=150),
    ]
    second_polygon = [
        Point(x=150, y=85),
        Point(x=170, y=85),
        Point(x=170, y=150),
        Point(x=150, y=150),
    ]

    regions = group_detected_regions(
        [
            DetectedRegion(
                polygon=first_polygon,
                bbox=(70, 85, 20, 65),
                orientation="vertical",
                source_text="後",
                segmentation=[first_polygon],
                metadata={"provider": "paddleocr"},
            ),
            DetectedRegion(
                polygon=second_polygon,
                bbox=(150, 85, 20, 65),
                orientation="vertical",
                source_text="先",
                segmentation=[second_polygon],
                metadata={"provider": "paddleocr"},
            ),
        ],
        image=image,
    )

    assert len(regions) == 1
    assert regions[0].source_text == "先\n後"
    assert regions[0].metadata["group_size"] == 2
    assert "bubble_bbox" in regions[0].metadata
    assert "bubble_text_bbox" in regions[0].metadata
    assert "bubble_contour" in regions[0].metadata


def test_bubble_text_bbox_stays_in_body_and_ignores_tail() -> None:
    image = np.full((180, 240, 3), 150, dtype=np.uint8)
    bubble_shape = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(bubble_shape, (100, 85), (65, 55), 0, 0, 360, 255, -1)
    cv2.fillConvexPoly(
        bubble_shape,
        np.array([[135, 120], [210, 155], [145, 105]], dtype=np.int32),
        255,
    )
    image[bubble_shape > 0] = (255, 255, 255)
    contours, _ = cv2.findContours(
        bubble_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, (20, 20, 20), 3)

    text_polygon = [
        Point(x=75, y=55),
        Point(x=125, y=55),
        Point(x=125, y=110),
        Point(x=75, y=110),
    ]
    locator = WhiteBubbleLocator(image)
    match = locator.find(polygon_geometry_mask(image.shape, [text_polygon]))

    assert match is not None
    x, y, width, height = match.text_bbox
    assert x + width < 175
    assert 80 <= x + width / 2 <= 120
    assert 65 <= y + height / 2 <= 105

    component = locator.mask(match)
    component_contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(component)
    cv2.drawContours(filled, component_contours, -1, 255, cv2.FILLED)
    assert np.all(filled[y : y + height, x : x + width] > 0)


def test_large_safe_rectangle_remains_inside_mask_after_downsampling() -> None:
    mask = np.zeros((700, 900), dtype=np.uint8)
    cv2.ellipse(mask, (390, 350), (330, 280), 0, 0, 360, 255, -1)
    cv2.fillConvexPoly(
        mask,
        np.array([[620, 470], [870, 650], [650, 420]], dtype=np.int32),
        255,
    )

    rectangle = _largest_centered_rectangle(mask)

    assert rectangle is not None
    x, y, width, height = rectangle
    assert np.all(mask[y : y + height, x : x + width] > 0)


def test_default_render_bbox_uses_detected_bubble_bounds() -> None:
    region = TextRegion(
        id="region-001",
        polygon=[],
        bbox=(75, 40, 20, 60),
        source_text="日本語",
        confidence=90,
        detector_metadata={"bubble_bbox": (20, 10, 100, 80)},
    )

    assert pipeline._default_render_bbox(region, (200, 120)) == (23, 13, 94, 74)


def test_default_render_bbox_prefers_safe_bubble_text_bounds() -> None:
    region = TextRegion(
        id="region-001",
        polygon=[],
        bbox=(75, 40, 20, 60),
        source_text="日本語",
        confidence=90,
        detector_metadata={
            "bubble_bbox": (20, 10, 140, 100),
            "bubble_text_bbox": (42, 25, 96, 68),
        },
    )

    assert pipeline._default_render_bbox(region, (200, 120)) == (42, 25, 96, 68)


def test_text_layout_uses_available_space_and_wraps_long_words() -> None:
    draw = ImageDraw.Draw(Image.new("RGB", (300, 200), "white"))

    short_layout = pipeline._fit_text_layout(draw, "Hi", None, 100, 70)
    long_layout = pipeline._fit_text_layout(
        draw,
        "SUPERCALIFRAGILISTICEXPIALIDOCIOUS",
        None,
        60,
        60,
    )

    assert short_layout is not None
    short_font, _, short_bounds, _ = short_layout
    assert short_font.size > 15
    assert short_bounds[2] - short_bounds[0] <= 100
    assert short_bounds[3] - short_bounds[1] <= 70

    assert long_layout is not None
    _, long_text, long_bounds, _ = long_layout
    assert "\n" in long_text
    assert long_bounds[2] - long_bounds[0] <= 60
    assert long_bounds[3] - long_bounds[1] <= 60


def test_text_layout_honors_requested_size_when_it_fits() -> None:
    draw = ImageDraw.Draw(Image.new("RGB", (300, 200), "white"))

    layout = pipeline._fit_text_layout(
        draw, "A short line", None, 220, 80, requested_size=24
    )

    assert layout is not None
    font, _, _, _ = layout
    assert font.size == 24


def test_text_layout_scales_above_legacy_cap_on_large_pages() -> None:
    draw = ImageDraw.Draw(Image.new("RGB", (800, 600), "white"))

    layout = pipeline._fit_text_layout(draw, "Hi", None, 500, 400)

    assert layout is not None
    font, _, _, _ = layout
    assert font.size > 200

    larger_font = pipeline._load_font(font.size + 1, None)
    larger_text = pipeline._wrap_text(draw, "Hi", larger_font, 500, stroke_width=1)
    larger_bounds = draw.multiline_textbbox(
        (0, 0),
        larger_text,
        font=larger_font,
        spacing=max(1, round((font.size + 1) * 0.12)),
        align="center",
        stroke_width=1,
    )
    assert (
        larger_bounds[2] - larger_bounds[0] > 500
        or larger_bounds[3] - larger_bounds[1] > 400
    )


def test_rendered_text_stays_inside_safe_bubble_bounds() -> None:
    image = Image.new("RGB", (200, 120), "white")
    region = TextRegion(
        id="region-001",
        polygon=[],
        bbox=(80, 45, 20, 30),
        source_text="日本語",
        translated_text="A centered translation",
        confidence=90,
        detector_metadata={"bubble_text_bbox": (40, 20, 120, 80)},
    )

    rendered = np.asarray(pipeline._render_text(image, [region], None))
    dark_y, dark_x = np.where(np.min(rendered, axis=2) < 128)

    assert dark_x.size > 0
    assert dark_x.min() >= 40
    assert dark_x.max() < 160
    assert dark_y.min() >= 20
    assert dark_y.max() < 100


def test_manga_reading_order_reorders_regions_right_to_left() -> None:
    left = TextRegion(
        id="left",
        polygon=[],
        bbox=(20, 10, 30, 30),
        source_text="左",
        confidence=90,
    )
    right = TextRegion(
        id="right",
        polygon=[],
        bbox=(120, 12, 30, 30),
        source_text="右",
        confidence=90,
    )
    lower = TextRegion(
        id="lower",
        polygon=[],
        bbox=(100, 80, 30, 30),
        source_text="下",
        confidence=90,
    )
    regions = [left, lower, right]

    pipeline._assign_manga_reading_order(regions)

    assert [region.id for region in regions] == ["right", "left", "lower"]
    assert [region.reading_order for region in regions] == [1, 2, 3]


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
