from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import shutil
from typing import Sequence

import cv2
import httpx
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont

from .models import ManualInpaintRegion, PipelineOptions, Point, TextRegion
from .providers import (
    DetectedRegion,
    ArgosTranslationProvider,
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
        return group_detected_regions([
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
        ])


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
    regions.sort(key=lambda region: (region.bbox[0], region.bbox[1]))
    rows: list[list[TextRegion]] = []
    for region in regions:
        placed = False
        for row in rows:
            first = row[0]
            first_bottom = first.bbox[1] + first.bbox[3]
            if (
                abs(region.bbox[1] - first.bbox[1])
                < max(first.bbox[3], region.bbox[3]) * 0.6
            ):
                row.append(region)
                placed = True
                break
        if not placed:
            rows.append([region])

    reading_order = 1
    for row in rows:
        row.sort(key=lambda region: -region.bbox[0])
        for region in row:
            region.reading_order = reading_order
            reading_order += 1


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
            raise PipelineError(f"DeepL request failed with status {response.status_code}")

        payload = response.json()
        raw_translations = payload.get("translations")
        if not isinstance(raw_translations, list) or len(raw_translations) != len(batch):
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
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    return [candidate for candidate in candidates if candidate]


@lru_cache(maxsize=256)
def _load_font(size: int, explicit_path: str | None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _font_candidates(explicit_path):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> str:
    def text_width(value: str) -> int:
        return draw.textbbox((0, 0), value, font=font)[2]

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

    wrapped_lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        if " " not in paragraph.strip():
            if paragraph and all(character.isascii() for character in paragraph):
                # Keep short English words intact so auto-fit reduces the font
                # instead of producing awkward splits such as "Fun" / "ny".
                wrapped_lines.append(paragraph)
            else:
                wrapped_lines.extend(wrap_characters(paragraph))
            continue

        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if current and text_width(candidate) > width:
                wrapped_lines.append(current)
                current = word
            elif not current and text_width(word) > width:
                wrapped_lines.extend(wrap_characters(word))
                current = ""
            else:
                current = candidate
        if current:
            wrapped_lines.append(current)
    return "\n".join(wrapped_lines)


def _default_render_bbox(
    region: TextRegion, image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x, y, width, height = region.bbox

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
    left = max(0, min(int(round(center_x - target_width / 2)), image_width - target_width))
    top = max(0, min(int(round(center_y - target_height / 2)), image_height - target_height))
    return left, top, target_width, target_height


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


def _ensure_render_bboxes(regions: Sequence[TextRegion], image_size: tuple[int, int]) -> None:
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


def _render_text(image: Image.Image, regions: list[TextRegion], font_path: str | None) -> Image.Image:
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    _ensure_render_bboxes(regions, output.size)
    for region in regions:
        x, y, width, height = region.render_bbox or region.bbox
        if width <= 4 or height <= 4 or not region.translated_text:
            continue

        padding = max(8, min(width, height) // 16)
        max_width = max(4, width - padding * 2)
        max_height = max(4, height - padding * 2)

        # Target about 65 % of the available space so text has breathing room.
        fill_ratio = 0.65
        target_width = max(4, int(max_width * fill_ratio))
        target_height = max(4, int(max_height * fill_ratio))

        minimum_size = 8
        maximum_size = max(minimum_size, min(64, min(target_width, target_height) // 3))

        def layout(size: int) -> tuple[ImageFont.ImageFont, str, tuple[int, int, int, int]]:
            font = _load_font(size, font_path)
            candidate = _wrap_text(draw, region.translated_text, font, target_width)
            spacing = max(2, size // 5)
            bounds = draw.multiline_textbbox((0, 0), candidate, font=font, spacing=spacing)
            return font, candidate, bounds

        requested_size = (
            maximum_size
            if region.font_size is None
            else max(minimum_size, min(maximum_size, region.font_size))
        )
        low, high = minimum_size, requested_size
        selected_font, selected_text, text_bounds = layout(minimum_size)
        while low <= high:
            size = (low + high) // 2
            candidate_font, candidate_text, candidate_bounds = layout(size)
            fits = (
                candidate_bounds[2] - candidate_bounds[0] <= target_width
                and candidate_bounds[3] - candidate_bounds[1] <= target_height
            )
            if fits:
                selected_font, selected_text, text_bounds = (
                    candidate_font,
                    candidate_text,
                    candidate_bounds,
                )
                low = size + 1
            else:
                high = size - 1

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
            spacing=max(2, selected_font.size // 5) if hasattr(selected_font, "size") else 3,
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


def _polygon_geometry_mask(
    image_shape: tuple[int, ...], polygons: Sequence[Sequence[Point]]
) -> np.ndarray:
    """Rasterize detector geometry without expanding it to a bounding box."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for points in polygons:
        polygon = np.array([[point.x, point.y] for point in points], dtype=np.int32)
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [polygon], 255)
    return mask


def _region_geometry_mask(
    image_shape: tuple[int, ...], region: TextRegion
) -> np.ndarray:
    """Rasterize all detector text geometry for one accepted region."""
    return _polygon_geometry_mask(
        image_shape, region.segmentation or [region.polygon]
    )


def _find_white_bubble_component(
    image: np.ndarray,
    geometry: np.ndarray,
    light_mask: np.ndarray,
    light_labels: np.ndarray,
    light_stats: np.ndarray,
) -> np.ndarray | None:
    """Find a local, enclosed light component around an OCR region.

    A connected component is used instead of a rectangle so irregular bubbles
    and their tails do not turn into large rectangular cleanup areas.
    """
    geometry_area = cv2.countNonZero(geometry)
    if geometry_area == 0:
        return None

    x, y, width, height = cv2.boundingRect(geometry)
    context_radius = max(6, min(24, int(round(min(width, height) * 0.25))))
    context_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (context_radius * 2 + 1, context_radius * 2 + 1),
    )
    context = cv2.dilate(geometry, context_kernel)

    light_inside = cv2.bitwise_and(light_mask, geometry)
    light_inside_ratio = cv2.countNonZero(light_inside) / geometry_area
    ring = context.copy()
    ring[geometry > 0] = 0
    ring_area = cv2.countNonZero(ring)
    if light_inside_ratio < 0.35 or ring_area == 0:
        return None
    light_ring = cv2.bitwise_and(light_mask, ring)
    if cv2.countNonZero(light_ring) / ring_area < 0.50:
        return None

    # Prefer the light component actually under the detector polygon. On a
    # white page, using the whole context first selects the page background
    # instead of a closed bubble because it overlaps more of the ring.
    candidate_labels = light_labels[geometry > 0]
    candidate_labels = candidate_labels[candidate_labels > 0]
    if candidate_labels.size == 0:
        candidate_labels = light_labels[context > 0]
        candidate_labels = candidate_labels[candidate_labels > 0]
    if candidate_labels.size == 0:
        return None
    labels, counts = np.unique(candidate_labels, return_counts=True)
    label = int(labels[int(np.argmax(counts))])

    image_height, image_width = image.shape[:2]
    component_x, component_y, component_width, component_height, component_area = (
        int(value) for value in light_stats[label]
    )
    if (
        component_x <= 1
        or component_y <= 1
        or component_x + component_width >= image_width - 1
        or component_y + component_height >= image_height - 1
    ):
        # A page or panel background usually reaches an image edge. Requiring
        # an enclosed component avoids treating titles and sound effects on
        # the page background as speech text.
        return None
    if component_width * component_height > image_width * image_height * 0.45:
        return None
    if component_area < max(256, int(geometry_area * 0.35)):
        return None

    geometry_center_x = x + width / 2
    geometry_center_y = y + height / 2
    if not (
        component_x - context_radius
        <= geometry_center_x
        <= component_x + component_width + context_radius
        and component_y - context_radius
        <= geometry_center_y
        <= component_y + component_height + context_radius
    ):
        return None

    component = np.where(light_labels == label, 255, 0).astype(np.uint8)
    # The separated component is eroded to break scan gaps, so inspect a few
    # pixels farther out when looking for the bubble's dark outline.
    boundary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    boundary = cv2.subtract(cv2.dilate(component, boundary_kernel), component)
    dark_boundary = cv2.bitwise_and(
        boundary, cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 160)
    )
    boundary_area = cv2.countNonZero(boundary)
    if boundary_area == 0 or cv2.countNonZero(dark_boundary) / boundary_area < 0.03:
        return None
    return component


def _build_inpainting_mask(
    image: np.ndarray, regions: Sequence[TextRegion], dilation: int
) -> np.ndarray:
    """Build a conservative text-pixel mask for white speech bubbles."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if not regions:
        return mask

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # A light component is deliberately stricter than a generic grayscale
    # threshold: screen tones and artwork should not qualify as a bubble.
    light_mask = cv2.inRange(gray, 225, 255)
    _, light_labels, light_stats, _ = cv2.connectedComponentsWithStats(
        light_mask, connectivity=8
    )
    # Small gaps in scanned bubble outlines can connect their white interior
    # to the page background. A lightly eroded copy separates those bridges
    # without changing the source pixels used for inpainting.
    separated_light_mask = cv2.erode(
        light_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    _, separated_labels, separated_stats, _ = cv2.connectedComponentsWithStats(
        separated_light_mask, connectivity=8
    )
    dark_mask = cv2.inRange(gray, 0, 210)

    for region in regions:
        # A recognizer can group lines from adjacent bubbles into one region.
        # Evaluate each detector polygon separately so each line can select its
        # own enclosing light component.
        for points in region.segmentation or [region.polygon]:
            geometry = _polygon_geometry_mask(image.shape, [points])
            bubble = _find_white_bubble_component(
                image, geometry, light_mask, light_labels, light_stats
            )
            if bubble is None:
                bubble = _find_white_bubble_component(
                    image,
                    geometry,
                    separated_light_mask,
                    separated_labels,
                    separated_stats,
                )
            if bubble is None:
                continue

            # Detector polygons can cover whitespace around a line. Recover
            # the actual dark glyph pixels before applying the safety margin.
            bubble_support = np.zeros_like(bubble)
            contours, _ = cv2.findContours(
                bubble, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                cv2.drawContours(bubble_support, contours, -1, 255, cv2.FILLED)
            bubble_support = cv2.dilate(
                bubble_support,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            )
            text_mask = cv2.bitwise_and(dark_mask, geometry)
            text_mask = cv2.bitwise_and(text_mask, bubble_support)

            component_count, text_labels, text_stats, _ = cv2.connectedComponentsWithStats(
                text_mask, connectivity=8
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
    mask = automatic_mask.copy()
    manual_mask = np.zeros_like(automatic_mask)
    _add_manual_inpaint_regions(manual_mask, manual_inpaint_regions)
    mask = cv2.bitwise_or(mask, manual_mask)
    if not cv2.countNonZero(mask):
        return image

    cleaned = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
    if cv2.countNonZero(manual_mask):
        # User rectangles usually cover a small glyph; a tight radius avoids
        # pulling in nearby outlines or artwork when the selection is narrow.
        cleaned = cv2.inpaint(cleaned, manual_mask, 1, cv2.INPAINT_NS)
    if cv2.countNonZero(automatic_mask):
        # Automatic mask pixels are restricted to white bubbles. Manual
        # rectangles are intentionally left to inpainting because they may
        # cover artwork or a non-white background.
        cleaned[automatic_mask > 0] = (255, 255, 255)
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
    cv2.imwrite(str(cleaned_path), cleaned)


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
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="aria-provider-init") as executor:
            detector_future = executor.submit(create_detector, options.detector_provider)
            recognizer_future = executor.submit(create_recognizer, options.recognizer_provider)
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
        warnings.append("DEEPL_API_KEY is not configured; source text was used as the translation.")

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
    cv2.imwrite(str(cleaned_path), cleaned)

    cleaned_pil = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
    rendered = _render_text(cleaned_pil, regions, options.font_path)
    rendered.save(output_path, format="PNG")
    return regions, warnings
