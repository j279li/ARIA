from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont

from .bubbles import WhiteBubbleLocator
from .bubbles import polygon_geometry_mask as _polygon_geometry_mask
from .models import ManualInpaintRegion, PipelineOptions, Point, TextRegion
from .providers import (
    ArgosTranslationProvider,
    DetectedRegion,
    HelsinkiTranslationProvider,
    MangaOCRRecognizer,
    OCRResult,
    PaddleOCRDetector,
    ProviderUnavailableError,
    TextDetector,
    TextRecognizer,
    TranslationProvider,
    group_detected_regions,
)


class PipelineError(RuntimeError):
    """Raised when a page cannot be processed."""


def resolve_tesseract_command() -> str | None:
    candidates = [
        os.getenv("TESSERACT_CMD"),
        shutil.which("tesseract"),
        "C:/Program Files/Tesseract-OCR/tesseract.exe",
        "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


@lru_cache(maxsize=2)
def create_detector(provider: str) -> TextDetector:
    if provider == "tesseract":
        return TesseractDetector()
    if provider == "paddleocr":
        try:
            return PaddleOCRDetector()
        except ProviderUnavailableError as exc:
            raise PipelineError(str(exc)) from exc
    raise PipelineError(f"Unsupported text detector provider: {provider}")


@lru_cache(maxsize=2)
def create_recognizer(provider: str) -> TextRecognizer:
    if provider == "tesseract":
        return TesseractRecognizer()
    if provider == "manga-ocr":
        try:
            return MangaOCRRecognizer()
        except ProviderUnavailableError as exc:
            raise PipelineError(str(exc)) from exc
    raise PipelineError(f"Unsupported text recognizer provider: {provider}")


@lru_cache(maxsize=1)
def create_translation_provider(provider: str) -> TranslationProvider:
    if provider == "argos":
        try:
            return ArgosTranslationProvider()
        except ProviderUnavailableError as exc:
            raise PipelineError(str(exc)) from exc
    if provider == "helsinki":
        try:
            return HelsinkiTranslationProvider()
        except ProviderUnavailableError as exc:
            raise PipelineError(str(exc)) from exc
    raise PipelineError(f"Unsupported translation provider: {provider}")


def _extract_regions(image_path: Path, options: PipelineOptions) -> list[TextRegion]:
    config = "--oem 1 --psm 11"
    configured_tesseract = resolve_tesseract_command()
    if configured_tesseract:
        pytesseract.pytesseract.tesseract_cmd = configured_tesseract

    try:
        data = pytesseract.image_to_data(
            str(image_path),
            lang=options.ocr_lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise PipelineError(
            "Tesseract is not installed or is not available on PATH. "
            "Install Tesseract and Japanese language data before starting a job."
        ) from exc
    except pytesseract.TesseractError as exc:
        raise PipelineError(
            f"Tesseract could not use language data '{options.ocr_lang}'. "
            "Install the Japanese traineddata or set ARIA_OCR_LANG=jpn."
        ) from exc

    lines: dict[tuple[int, int, int], dict[str, object]] = {}
    for index, raw_text in enumerate(data.get("text", [])):
        text = raw_text.strip()
        if not text:
            continue

        try:
            confidence = float(data["conf"][index])
        except (KeyError, TypeError, ValueError):
            confidence = -1
        if confidence < options.min_confidence:
            continue

        key = (
            int(data.get("block_num", [0])[index]),
            int(data.get("par_num", [0])[index]),
            int(data.get("line_num", [0])[index]),
        )
        x = int(data.get("left", [0])[index])
        y = int(data.get("top", [0])[index])
        width = int(data.get("width", [0])[index])
        height = int(data.get("height", [0])[index])
        line = lines.setdefault(
            key,
            {
                "texts": [],
                "x1": x,
                "y1": y,
                "x2": x + width,
                "y2": y + height,
                "confidences": [],
            },
        )
        line["texts"].append(text)  # type: ignore[union-attr]
        line["x1"] = min(int(line["x1"]), x)
        line["y1"] = min(int(line["y1"]), y)
        line["x2"] = max(int(line["x2"]), x + width)
        line["y2"] = max(int(line["y2"]), y + height)
        line["confidences"].append(confidence)  # type: ignore[union-attr]

    raw_regions: list[dict[str, object]] = []
    for line in lines.values():
        x1, y1 = int(line["x1"]), int(line["y1"])
        x2, y2 = int(line["x2"]), int(line["y2"])
        source_text = "".join(line["texts"])  # Japanese text should not gain spaces.
        if not source_text or x2 <= x1 or y2 <= y1:
            continue
        raw_regions.append(
            {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "source_text": source_text,
                "confidence": sum(line["confidences"]) / len(line["confidences"]),
            }
        )

    if not raw_regions:
        return []

    average_height = max(
        1,
        sum(int(region["height"]) for region in raw_regions) // len(raw_regions),
    )
    row_height = max(1, average_height * 2)
    raw_regions.sort(
        key=lambda region: (
            int(region["y"]) // row_height,
            -int(region["x"]),
        )
    )

    regions: list[TextRegion] = []
    for order, region in enumerate(raw_regions, start=1):
        x = int(region["x"])
        y = int(region["y"])
        width = int(region["width"])
        height = int(region["height"])
        regions.append(
            TextRegion(
                id=f"region-{order:03d}",
                polygon=[
                    Point(x=x, y=y),
                    Point(x=x + width, y=y),
                    Point(x=x + width, y=y + height),
                    Point(x=x, y=y + height),
                ],
                bbox=(x, y, width, height),
                source_text=str(region["source_text"]),
                confidence=float(region["confidence"]),
                recognition_confidence=float(region["confidence"]),
                orientation="vertical" if height > width else "horizontal",
                reading_order=order,
            )
        )
    return regions


class TesseractDetector:
    """Compatibility detector backed by Tesseract's line-level output."""

    def detect(self, image_path: str, options: PipelineOptions) -> list[DetectedRegion]:
        regions = _extract_regions(Path(image_path), options)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        return group_detected_regions(
            [
                DetectedRegion(
                    polygon=region.polygon,
                    bbox=region.bbox,
                    detector_confidence=region.detector_confidence,
                    orientation=region.orientation,
                    source_text=region.source_text,
                    recognition_confidence=region.recognition_confidence,
                    segmentation=[region.polygon],
                )
                for region in regions
            ],
            image=image,
        )


class TesseractRecognizer:
    """Pass through text already returned by the compatibility detector."""

    def recognize(
        self,
        image_path: str,
        regions: Sequence[DetectedRegion],
        options: PipelineOptions,
    ) -> list[OCRResult]:
        del image_path, options
        return [
            OCRResult(
                source_text=region.source_text or "",
                confidence=region.recognition_confidence,
            )
            for region in regions
        ]


def _regions_from_provider_output(
    detected: list[DetectedRegion], recognized: list[OCRResult]
) -> list[TextRegion]:
    if len(detected) != len(recognized):
        raise PipelineError(
            "The text recognizer returned a different number of results than the detector."
        )

    regions: list[TextRegion] = []
    for order, (detected_region, ocr) in enumerate(zip(detected, recognized), start=1):
        source_text = ocr.source_text.strip()
        if not source_text:
            continue
        if not _recognized_text_is_usable(detected_region, source_text):
            continue
        confidence = ocr.confidence
        if confidence is None:
            confidence = detected_region.detector_confidence
        if confidence is None:
            confidence = 0
        regions.append(
            TextRegion(
                id=f"region-{order:03d}",
                polygon=detected_region.polygon,
                bbox=detected_region.bbox,
                source_text=source_text,
                confidence=confidence,
                detector_confidence=detected_region.detector_confidence,
                recognition_confidence=ocr.confidence,
                segmentation=detected_region.segmentation or [detected_region.polygon],
                detector_metadata=detected_region.metadata,
                orientation=detected_region.orientation
                if detected_region.orientation in {"horizontal", "vertical"}
                else "horizontal",
                reading_order=order,
            )
        )
    return regions


def _recognized_text_is_usable(region: DetectedRegion, source_text: str) -> bool:
    if not any(character.isalnum() or character.isalpha() for character in source_text):
        return False
    if region.metadata.get("provider") == "paddleocr":
        return any(_is_japanese_character(character) for character in source_text)
    return True


def _is_japanese_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3040 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _assign_manga_reading_order(regions: list[TextRegion]) -> None:
    if not regions:
        return
    regions.sort(
        key=lambda region: (
            region.bbox[1],
            -region.bbox[0],
            -region.bbox[3],
            -region.bbox[2],
            region.id,
        )
    )
    rows: list[list[TextRegion]] = []
    for region in regions:
        placed = False
        for row in rows:
            first = row[0]
            region_center = region.bbox[1] + region.bbox[3] / 2
            row_center = first.bbox[1] + first.bbox[3] / 2
            if (
                abs(region_center - row_center)
                < max(first.bbox[3], region.bbox[3]) * 0.6
            ):
                row.append(region)
                placed = True
                break
        if not placed:
            rows.append([region])

    ordered: list[TextRegion] = []
    for row in rows:
        row.sort(
            key=lambda region: (
                -region.bbox[0],
                region.bbox[1],
                -region.bbox[3],
                -region.bbox[2],
                region.id,
            )
        )
        ordered.extend(row)
    regions[:] = ordered
    for reading_order, region in enumerate(regions, start=1):
        region.reading_order = reading_order


def _normalize_ocr_text(text: str) -> str:
    text = "".join(text.split())
    # Strip trailing full-width dots that follow kana (common OCR artifact).
    while len(text) >= 2 and text[-1] == "\uff0e" and "\u3040" <= text[-2] <= "\u30ff":
        text = text[:-1]
    while len(text) >= 2 and text[-1] == "." and "\u3040" <= text[-2] <= "\u30ff":
        text = text[:-1]
    return text


def _translate_regions(
    regions: list[TextRegion], options: PipelineOptions
) -> list[str]:
    if not regions:
        return []
    if options.translation_provider == "identity":
        return [region.source_text for region in regions]

    if options.translation_provider == "argos":
        return create_translation_provider("argos").translate(
            [region.source_text for region in regions], options
        )

    if options.translation_provider == "helsinki":
        return create_translation_provider("helsinki").translate(
            [region.source_text for region in regions], options
        )

    api_key = os.getenv("DEEPL_API_KEY")
    if not api_key:
        return [region.source_text for region in regions]

    translations: list[str] = []
    max_request_chars = 100_000
    start = 0
    while start < len(regions):
        end = start
        size = 0
        while end < len(regions):
            candidate_size = len(regions[end].source_text)
            if end > start and size + candidate_size > max_request_chars:
                break
            size += candidate_size
            end += 1

        batch = regions[start:end]
        context = "\n".join(region.source_text for region in batch)[:10_000]
        response = httpx.post(
            "https://api-free.deepl.com/v2/translate",
            headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
            json={
                "text": [region.source_text for region in batch],
                "source_lang": "JA",
                "target_lang": "EN-US",
                "context": context,
                "preserve_formatting": True,
            },
            timeout=60,
        )
        if not response.is_success:
            raise PipelineError(
                f"DeepL request failed with status {response.status_code}"
            )

        payload = response.json()
        raw_translations = payload.get("translations")
        if not isinstance(raw_translations, list) or len(raw_translations) != len(
            batch
        ):
            raise PipelineError("DeepL returned an unexpected translation count")
        for item in raw_translations:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise PipelineError("DeepL returned an invalid translation")
            translations.append(item["text"].strip())
        start = end
    return translations


def _font_candidates(explicit_path: str | None) -> list[str]:
    candidates = [
        explicit_path,
        os.getenv("ARIA_FONT_PATH"),
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


@lru_cache(maxsize=256)
def _load_font(
    size: int, explicit_path: str | None
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _font_candidates(explicit_path):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    width: int,
    stroke_width: int = 0,
) -> str:
    def text_width(value: str) -> int:
        bounds = draw.textbbox((0, 0), value, font=font, stroke_width=stroke_width)
        return bounds[2] - bounds[0]

    def wrap_characters(value: str) -> list[str]:
        lines: list[str] = []
        current = ""
        for character in value:
            candidate = current + character
            if current and text_width(candidate) > width:
                lines.append(current)
                current = character
            else:
                current = candidate
        if current or not lines:
            lines.append(current)
        return lines

    def balance(lines: list[str]) -> list[str]:
        for index in range(len(lines) - 1):
            words = lines[index].split()
            while len(words) > 1:
                next_line = f"{words[-1]} {lines[index + 1]}".strip()
                shortened = " ".join(words[:-1])
                if text_width(next_line) > width or max(
                    text_width(shortened), text_width(next_line)
                ) >= max(text_width(lines[index]), text_width(lines[index + 1])):
                    break
                lines[index] = shortened
                lines[index + 1] = next_line
                words.pop()
        return lines

    wrapped_lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        if text_width(paragraph) <= width:
            wrapped_lines.append(paragraph)
            continue
        if " " not in paragraph.strip():
            wrapped_lines.extend(wrap_characters(paragraph))
            continue

        paragraph_lines: list[str] = []
        current = ""
        for word in paragraph.split():
            if text_width(word) > width:
                if current:
                    paragraph_lines.append(current)
                    current = ""
                chunks = wrap_characters(word)
                paragraph_lines.extend(chunks[:-1])
                current = chunks[-1]
                continue
            candidate = f"{current} {word}".strip()
            if current and text_width(candidate) > width:
                paragraph_lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            paragraph_lines.append(current)
        wrapped_lines.extend(balance(paragraph_lines))
    return "\n".join(wrapped_lines)


def _default_render_bbox(
    region: TextRegion, image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x, y, width, height = region.bbox

    bubble_text_bbox = _metadata_bbox(region.detector_metadata.get("bubble_text_bbox"))
    if bubble_text_bbox is not None:
        text_x, text_y, text_width, text_height = bubble_text_bbox
        left = max(0, text_x)
        top = max(0, text_y)
        right = min(image_width, text_x + text_width)
        bottom = min(image_height, text_y + text_height)
        if right - left >= 8 and bottom - top >= 8:
            return left, top, right - left, bottom - top

    bubble_bbox = _metadata_bbox(region.detector_metadata.get("bubble_bbox"))
    if bubble_bbox is not None:
        bubble_x, bubble_y, bubble_width, bubble_height = bubble_bbox
        left = max(0, bubble_x)
        top = max(0, bubble_y)
        right = min(image_width, bubble_x + bubble_width)
        bottom = min(image_height, bubble_y + bubble_height)
        margin = max(2, round(min(right - left, bottom - top) * 0.04))
        if right - left > margin * 2 + 8 and bottom - top > margin * 2 + 8:
            return (
                left + margin,
                top + margin,
                right - left - margin * 2,
                bottom - top - margin * 2,
            )

    # Use segmentation polygons (the individual line boxes from the detector)
    # to get a tighter estimate of where text sits within the bubble.
    seg_box = _segmentation_union_bbox(region.segmentation)
    if seg_box is not None:
        seg_x, seg_y, seg_w, seg_h = seg_box
    else:
        seg_x, seg_y, seg_w, seg_h = x, y, width, height

    vertical = region.orientation == "vertical" or height > width * 1.4
    if vertical:
        # English translations need horizontal space; expand width
        # substantially beyond the vertical OCR column.
        target_width = max(seg_w + 32, int(seg_h * 0.9))
        target_height = max(seg_h + 32, int(seg_h * 1.2))
    else:
        target_width = max(seg_w + 32, int(seg_w * 1.3))
        target_height = max(seg_h + 24, int(target_width * 0.55))

    target_width = min(target_width, max(seg_w, int(image_width * 0.35)))
    target_height = min(target_height, max(seg_h, int(image_height * 0.25)))
    target_width = min(target_width, image_width)
    target_height = min(target_height, image_height)
    center_x = seg_x + seg_w / 2
    center_y = seg_y + seg_h / 2
    left = max(0, min(round(center_x - target_width / 2), image_width - target_width))
    top = max(0, min(round(center_y - target_height / 2), image_height - target_height))
    return left, top, target_width, target_height


def _fit_text_layout(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str | None,
    width: int,
    height: int,
    requested_size: int | None = None,
) -> tuple[ImageFont.ImageFont, str, tuple[int, int, int, int], int] | None:
    minimum_size = 6
    selected: tuple[ImageFont.ImageFont, str, tuple[int, int, int, int], int] | None = (
        None
    )
    measured: dict[
        int, tuple[ImageFont.ImageFont, str, tuple[int, int, int, int], int]
    ] = {}

    def measure(
        size: int,
    ) -> tuple[ImageFont.ImageFont, str, tuple[int, int, int, int], int]:
        if size not in measured:
            font = _load_font(size, font_path)
            wrapped = _wrap_text(draw, text, font, width, stroke_width=1)
            spacing = max(1, round(size * 0.12))
            bounds = draw.multiline_textbbox(
                (0, 0),
                wrapped,
                font=font,
                spacing=spacing,
                align="center",
                stroke_width=1,
            )
            measured[size] = font, wrapped, bounds, spacing
        return measured[size]

    def fits(
        layout: tuple[ImageFont.ImageFont, str, tuple[int, int, int, int], int],
    ) -> bool:
        bounds = layout[2]
        return bounds[2] - bounds[0] <= width and bounds[3] - bounds[1] <= height

    maximum_size = requested_size or max(width, height)
    if requested_size is None:
        while maximum_size < 8192 and fits(measure(maximum_size)):
            maximum_size *= 2

    low, high = minimum_size, max(minimum_size, maximum_size)
    while low <= high:
        size = (low + high) // 2
        candidate = measure(size)
        if fits(candidate):
            selected = candidate
            low = size + 1
        else:
            high = size - 1
    return selected


def _metadata_bbox(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(item, (int, float)) for item in value):
        return None
    x, y, width, height = (round(item) for item in value)
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def _metadata_contour(value: object) -> tuple[tuple[int, int], ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    points: list[tuple[int, int]] = []
    for point in value:
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or not all(isinstance(coordinate, (int, float)) for coordinate in point)
        ):
            return None
        points.append((round(point[0]), round(point[1])))
    return tuple(points)


def _metadata_tone(value: object) -> int | None:
    if not isinstance(value, (int, float)):
        return None
    return max(0, min(255, round(value)))


def _segmentation_union_bbox(
    segmentations: list[list[Point]],
) -> tuple[int, int, int, int] | None:
    if not segmentations:
        return None
    xs: list[int] = []
    ys: list[int] = []
    for polygon in segmentations:
        for point in polygon:
            xs.append(point.x)
            ys.append(point.y)
    if not xs:
        return None
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _ensure_render_bboxes(
    regions: Sequence[TextRegion], image_size: tuple[int, int]
) -> None:
    image_width, image_height = image_size
    for region in regions:
        if region.render_bbox is None:
            region.render_bbox = _default_render_bbox(region, image_size)
        x, y, width, height = region.render_bbox
        width = max(8, min(width, image_width))
        height = max(8, min(height, image_height))
        x = max(0, min(x, image_width - width))
        y = max(0, min(y, image_height - height))
        region.render_bbox = (x, y, width, height)


def _render_text(
    image: Image.Image, regions: list[TextRegion], font_path: str | None
) -> Image.Image:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    _ensure_render_bboxes(regions, output.size)
    for region in regions:
        x, y, width, height = region.render_bbox or region.bbox
        if width <= 4 or height <= 4 or not region.translated_text:
            continue

        has_safe_bounds = (
            _metadata_bbox(region.detector_metadata.get("bubble_text_bbox")) is not None
            and region.detector_metadata.get("manual_render_bbox") is not True
        )
        padding_ratio = 0.025 if has_safe_bounds else 0.10
        padding = max(2, round(min(width, height) * padding_ratio))
        target_width = max(4, width - padding * 2)
        target_height = max(4, height - padding * 2)
        layout = _fit_text_layout(
            draw,
            region.translated_text,
            font_path,
            target_width,
            target_height,
            region.font_size,
        )
        if layout is None:
            continue
        selected_font, selected_text, text_bounds, spacing = layout

        text_width = text_bounds[2] - text_bounds[0]
        text_height = text_bounds[3] - text_bounds[1]
        position = (
            x + (width - text_width) / 2 - text_bounds[0],
            y + (height - text_height) / 2 - text_bounds[1],
        )
        draw.multiline_text(
            position,
            selected_text,
            font=selected_font,
            fill="black",
            align="center",
            spacing=spacing,
            stroke_width=1,
            stroke_fill="white",
        )
    return output


def render_page(
    cleaned_path: Path,
    output_path: Path,
    regions: list[TextRegion],
    font_path: str | None = None,
) -> None:
    with Image.open(cleaned_path) as image:
        rendered = _render_text(image, regions, font_path)
        rendered.save(output_path, format="PNG")


def _build_inpainting_mask(
    image: np.ndarray, regions: Sequence[TextRegion], dilation: int
) -> np.ndarray:
    """Build a conservative text-pixel mask for white speech bubbles."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if not regions:
        return mask

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_masks: dict[int, np.ndarray] = {}
    locator: WhiteBubbleLocator | None = None

    for region in regions:
        contour = _metadata_contour(region.detector_metadata.get("bubble_contour"))
        region_tone = _metadata_tone(region.detector_metadata.get("bubble_fill_tone"))
        if contour is None and region.detector_metadata.get("bubble_checked") is True:
            continue

        contour_support: np.ndarray | None = None
        if contour is not None:
            contour_support = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(
                contour_support,
                [np.array(contour, dtype=np.int32)],
                255,
            )

        bubble_supports: dict[tuple[int, int], np.ndarray] = {}
        for points in region.segmentation or [region.polygon]:
            geometry = _polygon_geometry_mask(image.shape, [points])
            bubble_support = contour_support
            fill_tone = region_tone or 255
            if bubble_support is not None:
                geometry_area = cv2.countNonZero(geometry)
                supported_area = cv2.countNonZero(
                    cv2.bitwise_and(geometry, bubble_support)
                )
                if not geometry_area or supported_area / geometry_area < 0.80:
                    bubble_support = None
            if bubble_support is None:
                locator = locator or WhiteBubbleLocator(image)
                bubble_match = locator.find(geometry)
                if bubble_match is None:
                    continue
                fill_tone = bubble_match.fill_tone
                bubble_key = (bubble_match.component_set, bubble_match.label)
                bubble_support = bubble_supports.get(bubble_key)
                if bubble_support is None:
                    bubble = locator.mask(bubble_match)
                    contours, _ = cv2.findContours(
                        bubble, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    bubble_support = np.zeros_like(bubble)
                    if contours:
                        cv2.drawContours(bubble_support, contours, -1, 255, cv2.FILLED)
                    bubble_supports[bubble_key] = bubble_support
            text_threshold = max(0, min(210, fill_tone - 30))
            dark_mask = dark_masks.get(text_threshold)
            if dark_mask is None:
                dark_mask = cv2.inRange(gray, 0, text_threshold)
                dark_masks[text_threshold] = dark_mask
            text_mask = cv2.bitwise_and(dark_mask, geometry)
            text_mask = cv2.bitwise_and(text_mask, bubble_support)

            component_count, text_labels, text_stats, _ = (
                cv2.connectedComponentsWithStats(text_mask, connectivity=8)
            )
            filtered = np.zeros_like(text_mask)
            for label in range(1, component_count):
                if int(text_stats[label, cv2.CC_STAT_AREA]) >= 2:
                    filtered[text_labels == label] = 255
            text_mask = filtered

            if dilation:
                radius = max(0, int(dilation))
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
                )
                text_mask = cv2.dilate(text_mask, kernel)
                # Keep the safety margin inside the bubble, especially when a
                # detector polygon lies close to the balloon outline.
                text_mask = cv2.bitwise_and(text_mask, bubble_support)
            mask = cv2.bitwise_or(mask, text_mask)
    return mask


def _add_manual_inpaint_regions(
    mask: np.ndarray, regions: Sequence[ManualInpaintRegion]
) -> np.ndarray:
    """Add explicit user-selected rectangles to an existing cleanup mask."""
    for region in regions:
        x, y, width, height = region.bbox
        x1 = max(0, min(mask.shape[1], int(x)))
        y1 = max(0, min(mask.shape[0], int(y)))
        x2 = max(x1, min(mask.shape[1], int(x + width)))
        y2 = max(y1, min(mask.shape[0], int(y + height)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def clean_image(
    image: np.ndarray,
    regions: Sequence[TextRegion],
    dilation: int,
    manual_inpaint_regions: Sequence[ManualInpaintRegion] = (),
) -> np.ndarray:
    """Apply automatic and explicit cleanup masks to a decoded BGR image."""
    automatic_mask = _build_inpainting_mask(image, regions, dilation)
    manual_mask = np.zeros_like(automatic_mask)
    _add_manual_inpaint_regions(manual_mask, manual_inpaint_regions)
    has_automatic_mask = bool(cv2.countNonZero(automatic_mask))
    has_manual_mask = bool(cv2.countNonZero(manual_mask))
    if not has_automatic_mask and not has_manual_mask:
        return image

    cleaned = (
        cv2.inpaint(image, automatic_mask, 3, cv2.INPAINT_NS)
        if has_automatic_mask
        else image.copy()
    )
    if has_manual_mask:
        cleaned = cv2.inpaint(cleaned, manual_mask, 1, cv2.INPAINT_NS)
    return cleaned


def clean_page(
    source_path: Path,
    cleaned_path: Path,
    regions: Sequence[TextRegion],
    dilation: int,
    manual_inpaint_regions: Sequence[ManualInpaintRegion] = (),
) -> None:
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise PipelineError(f"Could not decode image: {source_path.name}")
    cleaned = clean_image(image, regions, dilation, manual_inpaint_regions)
    if not cv2.imwrite(str(cleaned_path), cleaned):
        raise PipelineError(f"Could not write cleaned image: {cleaned_path.name}")


def process_page(
    source_path: Path,
    cleaned_path: Path,
    output_path: Path,
    options: PipelineOptions,
    *,
    detector: TextDetector | None = None,
    recognizer: TextRecognizer | None = None,
) -> tuple[list[TextRegion], list[str]]:
    if detector is None and recognizer is None:
        # These models do not depend on one another; load them together so the
        # first page does not pay both model initialization costs sequentially.
        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="aria-provider-init"
        ) as executor:
            detector_future = executor.submit(
                create_detector, options.detector_provider
            )
            recognizer_future = executor.submit(
                create_recognizer, options.recognizer_provider
            )
            detector = detector_future.result()
            recognizer = recognizer_future.result()
    else:
        detector = detector or create_detector(options.detector_provider)
        recognizer = recognizer or create_recognizer(options.recognizer_provider)

    detected = detector.detect(str(source_path), options)
    recognized = recognizer.recognize(str(source_path), detected, options)
    regions = _regions_from_provider_output(detected, recognized)
    _assign_manga_reading_order(regions)
    for region in regions:
        region.source_text = _normalize_ocr_text(region.source_text)
    warnings: list[str] = []
    if not regions:
        warnings.append("No text regions were detected.")

    if options.translation_provider == "deepl" and not os.getenv("DEEPL_API_KEY"):
        warnings.append(
            "DEEPL_API_KEY is not configured; source text was used as the translation."
        )

    translated = _translate_regions(regions, options)
    for region, translation in zip(regions, translated):
        region.translated_text = translation

    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise PipelineError(f"Could not decode image: {source_path.name}")

    _ensure_render_bboxes(regions, (image.shape[1], image.shape[0]))

    # Only accepted OCR regions are eligible for automatic cleanup. Explicit
    # manual regions are empty during initial processing.
    cleaned = clean_image(image, regions, options.mask_dilation)
    if not cv2.imwrite(str(cleaned_path), cleaned):
        raise PipelineError(f"Could not write cleaned image: {cleaned_path.name}")

    cleaned_pil = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
    rendered = _render_text(cleaned_pil, regions, options.font_path)
    rendered.save(output_path, format="PNG")
    return regions, warnings
