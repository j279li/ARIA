from __future__ import annotations

import os
import unicodedata
from threading import Lock
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import cv2
import numpy as np

from .models import PipelineOptions, Point


@dataclass(frozen=True)
class DetectedRegion:
    """Geometry produced by a detector, optionally with detector-side OCR."""

    polygon: list[Point]
    bbox: tuple[int, int, int, int]
    detector_confidence: float | None = None
    orientation: str = "horizontal"
    source_text: str | None = None
    recognition_confidence: float | None = None
    segmentation: list[list[Point]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OCRResult:
    source_text: str
    confidence: float | None = None


class TextDetector(Protocol):
    def detect(self, image_path: str, options: PipelineOptions) -> list[DetectedRegion]:
        """Return text geometry in page coordinates."""


class TextRecognizer(Protocol):
    def recognize(
        self,
        image_path: str,
        regions: Sequence[DetectedRegion],
        options: PipelineOptions,
    ) -> list[OCRResult]:
        """Return one OCR result for each detected region, in the same order."""


class MaskProvider(Protocol):
    def create_mask(
        self,
        image_shape: tuple[int, ...],
        regions: Sequence[DetectedRegion],
        dilation: int,
    ) -> np.ndarray:
        """Create an 8-bit inpainting mask for the detected regions."""


class TranslationProvider(Protocol):
    def translate(
        self, texts: Sequence[str], options: PipelineOptions
    ) -> list[str]:
        """Translate source texts in order."""


class PolygonMaskProvider:
    """Rasterize detector polygons and optionally expand them for inpainting."""

    def create_mask(
        self,
        image_shape: tuple[int, ...],
        regions: Sequence[DetectedRegion],
        dilation: int,
    ) -> np.ndarray:
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        for region in regions:
            polygons = region.segmentation or [region.polygon]
            for points in polygons:
                polygon = np.array(
                    [[point.x, point.y] for point in points], dtype=np.int32
                )
                if len(polygon) >= 3:
                    cv2.fillPoly(mask, [polygon], 255)

        if dilation:
            kernel_size = dilation * 2 + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel)
        return mask


class ProviderUnavailableError(RuntimeError):
    """Raised when an optional provider is unavailable or not configured."""


_ARGOS_LOCK = Lock()


class ArgosTranslationProvider:
    """Translate Japanese to English with an offline Argos language package."""

    def __init__(self) -> None:
        try:
            import argostranslate.package as package
            import argostranslate.translate as translate
        except ImportError as exc:
            raise ProviderUnavailableError(
                "The Argos Translate provider is not installed. "
                "Install it with 'pip install -e .[argos]'."
            ) from exc

        self._package = package
        self._translate = translate
        with _ARGOS_LOCK:
            self._ensure_language_pair()

    def _ensure_language_pair(self) -> None:
        installed_languages = self._translate.get_installed_languages()
        if _find_translation(installed_languages):
            return

        if os.getenv("ARIA_ARGOS_AUTO_INSTALL", "1").lower() not in {
            "1",
            "true",
            "yes",
        }:
            raise ProviderUnavailableError(
                "Argos Translate has no Japanese-to-English package installed. "
                "Enable ARIA_ARGOS_AUTO_INSTALL or install the ja-en Argos model."
            )

        try:
            self._package.update_package_index()
            available = self._package.get_available_packages()
            model = next(
                package
                for package in available
                if package.from_code == "ja" and package.to_code == "en"
            )
            self._package.install_from_path(model.download())
        except (OSError, StopIteration, RuntimeError) as exc:
            raise ProviderUnavailableError(
                "Argos could not install its Japanese-to-English model package. "
                "Check network access or install the Argos ja-en package manually."
            ) from exc

        if not _find_translation(self._translate.get_installed_languages()):
            raise ProviderUnavailableError(
                "Argos Japanese-to-English model installation did not complete."
            )

    def translate(
        self, texts: Sequence[str], options: PipelineOptions
    ) -> list[str]:
        del options
        translated: dict[str, str] = {}
        for text in texts:
            if text not in translated:
                result = self._translate.translate(text, "ja", "en").strip()
                if not result:
                    normalized = unicodedata.normalize("NFKC", text)
                    if normalized != text:
                        result = self._translate.translate(normalized, "ja", "en").strip()
                    if not result:
                        simplified = normalized.rstrip(" .!?。！？…")
                        if simplified != normalized:
                            result = self._translate.translate(simplified, "ja", "en").strip()
                translated[text] = result or text
        return [translated[text] for text in texts]


def _find_translation(languages: Sequence[Any]) -> Any | None:
    source = next((language for language in languages if language.code == "ja"), None)
    if source is None:
        return None
    return next(
        (
            translation
            for translation in source.translations_from
            if translation.to_lang.code == "en"
        ),
        None,
    )


_HELSINKI_MODEL_LOCK = Lock()
_HELSINKI_MODEL_NAME = "Helsinki-NLP/opus-mt-ja-en"


def _load_helsinki_model() -> tuple[Any, Any]:
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError as exc:
        raise ProviderUnavailableError(
            "The Helsinki OPUS-MT provider requires the transformers package. "
            "Install it with 'pip install -e .[helsinki]'."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(_HELSINKI_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(_HELSINKI_MODEL_NAME)
    return model, tokenizer


_HELSINKI_CACHE: tuple[Any, Any] | None = None


def _get_helsinki_model() -> tuple[Any, Any]:
    global _HELSINKI_CACHE
    if _HELSINKI_CACHE is not None:
        return _HELSINKI_CACHE
    with _HELSINKI_MODEL_LOCK:
        if _HELSINKI_CACHE is not None:
            return _HELSINKI_CACHE
        _HELSINKI_CACHE = _load_helsinki_model()
        return _HELSINKI_CACHE


class HelsinkiTranslationProvider:
    """Translate Japanese to English with Helsinki OPUS-MT."""

    def translate(
        self, texts: Sequence[str], options: PipelineOptions
    ) -> list[str]:
        del options
        if not texts:
            return []
        model, tokenizer = _get_helsinki_model()
        deduped: dict[str, str] = {}
        pending = [text for text in texts if text not in deduped]
        if pending:
            inputs = tokenizer(pending, return_tensors="pt", padding=True)
            outputs = model.generate(**inputs, max_new_tokens=80)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for original, translation in zip(pending, decoded):
                deduped[original] = translation.strip()
        return [deduped[text] or text for text in texts]


class MangaOCRRecognizer:
    """Recognize detector crops with the manga-specific Manga OCR model."""

    def __init__(self) -> None:
        try:
            from manga_ocr import MangaOcr
        except ImportError as exc:
            raise ProviderUnavailableError(
                "The manga-ocr provider is not installed. "
                "Install it with 'pip install -e .[manga-ocr]'."
            ) from exc

        self._ocr = MangaOcr()
        self._padding = max(0, int(os.getenv("ARIA_MANGA_OCR_PADDING", "8")))

    def recognize(
        self,
        image_path: str,
        regions: Sequence[DetectedRegion],
        options: PipelineOptions,
    ) -> list[OCRResult]:
        del options
        from PIL import Image

        with Image.open(image_path) as image:
            page_width, page_height = image.size
            results: list[OCRResult] = []
            for region in regions:
                x, y, width, height = region.bbox
                left = max(0, x - self._padding)
                top = max(0, y - self._padding)
                right = min(page_width, x + width + self._padding)
                bottom = min(page_height, y + height + self._padding)
                if right <= left or bottom <= top:
                    results.append(OCRResult(source_text=""))
                    continue
                text = str(self._ocr(image.crop((left, top, right, bottom)))).strip()
                results.append(OCRResult(source_text=text))
            return results


class PaddleOCRDetector:
    """Detect text polygons with PaddleOCR while supporting its 2.x and 3.x APIs."""

    def __init__(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ProviderUnavailableError(
                "The PaddleOCR provider is not installed. "
                "Install PaddlePaddle for your platform, then run "
                "'pip install -e .[paddleocr]'."
            ) from exc

        language = os.getenv("ARIA_PADDLEOCR_LANG", "japan")
        device = os.getenv("ARIA_DEVICE", "cpu")
        enable_mkldnn = os.getenv("ARIA_PADDLEOCR_ENABLE_MKLDNN", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        try:
            from paddleocr._models.text_detection import TextDetection
        except ImportError:
            TextDetection = None

        if TextDetection is not None:
            try:
                self._ocr = TextDetection(
                    model_name=os.getenv(
                        "ARIA_PADDLEOCR_DET_MODEL", "PP-OCRv6_medium_det"
                    ),
                    device=device,
                    enable_mkldnn=enable_mkldnn,
                    limit_side_len=max(
                        32, int(os.getenv("ARIA_PADDLEOCR_LIMIT_SIDE_LEN", "960"))
                    ),
                )
                self._modern_api = True
                self._detection_only = True
                return
            except (TypeError, ValueError, RuntimeError):
                # Fall back to PaddleOCR's public pipeline for older/partial installs.
                pass

        try:
            self._ocr = PaddleOCR(
                lang=language,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=device,
                enable_mkldnn=enable_mkldnn,
            )
            self._modern_api = True
            self._detection_only = False
        except TypeError:
            self._ocr = PaddleOCR(
                lang=language,
                use_angle_cls=True,
                use_gpu=False,
                enable_mkldnn=enable_mkldnn,
            )
            self._modern_api = False
            self._detection_only = False

    def detect(self, image_path: str, options: PipelineOptions) -> list[DetectedRegion]:
        if self._modern_api:
            results = self._ocr.predict(image_path)
            return self._regions_from_modern_results(results, options)

        results = self._ocr.ocr(image_path, cls=True)
        return self._regions_from_legacy_results(results, options)

    def _regions_from_modern_results(
        self, results: Any, options: PipelineOptions
    ) -> list[DetectedRegion]:
        regions: list[DetectedRegion] = []
        for result in results or []:
            texts = _result_values(result, "rec_texts")
            recognition_scores = _result_values(result, "rec_scores")
            detector_scores = _result_values(result, "dt_scores")
            polygons = _result_values(result, "dt_polys") or _result_values(
                result, "rec_polys"
            )
            for index, points in enumerate(polygons):
                polygon = _points_from_paddle(points)
                if len(polygon) < 3:
                    continue
                recognition_score = _float_or_none(
                    recognition_scores[index]
                    if index < len(recognition_scores)
                    else None
                )
                detector_score = _float_or_none(
                    detector_scores[index] if index < len(detector_scores) else None
                )
                score = recognition_score if recognition_score is not None else detector_score
                if (
                    score is not None and score * 100 < options.min_confidence
                ):
                    continue
                text = str(texts[index]).strip() if index < len(texts) else None
                regions.append(
                    _detected_region_from_polygon(
                        polygon,
                        recognition_score,
                        text,
                        detector_confidence=detector_score,
                        metadata={"provider": "paddleocr", "api": "modern"},
                    )
                )
        return group_detected_regions(regions)

    def _regions_from_legacy_results(
        self, results: Any, options: PipelineOptions
    ) -> list[DetectedRegion]:
        regions: list[DetectedRegion] = []
        for page in results or []:
            for line in page or []:
                if len(line) < 2:
                    continue
                polygon = _points_from_paddle(line[0])
                text_score = line[1]
                text = str(text_score[0]).strip() if text_score else None
                recognition_score = _float_or_none(
                    text_score[1] if text_score else None
                )
                if len(polygon) < 3 or (
                    recognition_score is not None
                    and recognition_score * 100 < options.min_confidence
                ):
                    continue
                regions.append(
                    _detected_region_from_polygon(
                        polygon,
                        recognition_score,
                        text,
                        metadata={"provider": "paddleocr", "api": "legacy"},
                    )
                )
        return group_detected_regions(regions)


def group_detected_regions(regions: Sequence[DetectedRegion]) -> list[DetectedRegion]:
    """Combine nearby OCR lines so recognizers receive one speech bubble at a time."""
    candidates = [region for region in regions if _keep_detected_region(region)]
    if len(candidates) < 2:
        return list(candidates)

    groups: list[list[DetectedRegion]] = []
    for region in candidates:
        matching_groups = [
            group
            for group in groups
            if any(_regions_should_merge(region, member) for member in group)
        ]
        if not matching_groups:
            groups.append([region])
            continue

        target = matching_groups[0]
        target.append(region)
        for other in matching_groups[1:]:
            target.extend(other)
            groups.remove(other)

    max_group_width = max(1, int(os.getenv("ARIA_GROUP_MAX_WIDTH", "280")))
    split: list[list[DetectedRegion]] = []
    for group in groups:
        split.extend(_split_wide_group(group, max_group_width))

    grouped: list[DetectedRegion] = []
    for group in split:
        if len(group) == 1:
            grouped.append(group[0])
            continue
        grouped.append(_merge_detected_regions(group))
    return grouped


def _group_bbox_x_span(group: Sequence[DetectedRegion]) -> int:
    if not group:
        return 0
    x1 = min(region.bbox[0] for region in group)
    x2 = max(region.bbox[0] + region.bbox[2] for region in group)
    return x2 - x1


def _split_wide_group(
    group: list[DetectedRegion], max_width: int
) -> list[list[DetectedRegion]]:
    if _group_bbox_x_span(group) <= max_width:
        return [group]

    group.sort(key=lambda region: region.bbox[0])
    split_points: list[int] = []
    for i in range(len(group) - 1):
        current_right = group[i].bbox[0] + group[i].bbox[2]
        next_left = group[i + 1].bbox[0]
        gap = next_left - current_right
        if gap > max(12, int(max_width * 0.08)):
            split_points.append(i + 1)

    if not split_points:
        return [group]

    result: list[list[DetectedRegion]] = []
    start = 0
    for point in split_points:
        result.append(group[start:point])
        start = point
    result.append(group[start:])
    return result


def _contains_text_character(text: str) -> bool:
    return any(
        character.isalnum() or unicodedata.category(character).startswith("L")
        for character in text
    )


def _keep_detected_region(region: DetectedRegion) -> bool:
    if not region.source_text:
        return True
    if not _contains_text_character(region.source_text):
        return False
    if region.metadata.get("provider") == "paddleocr":
        return any(_is_japanese_character(character) for character in region.source_text)
    return True


def _is_japanese_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3040 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _regions_should_merge(first: DetectedRegion, second: DetectedRegion) -> bool:
    first_x, first_y, first_width, first_height = first.bbox
    second_x, second_y, second_width, second_height = second.bbox
    first_right = first_x + first_width
    second_right = second_x + second_width
    first_bottom = first_y + first_height
    second_bottom = second_y + second_height
    horizontal_overlap = max(0, min(first_right, second_right) - max(first_x, second_x))
    vertical_overlap = max(0, min(first_bottom, second_bottom) - max(first_y, second_y))
    horizontal_gap = max(first_x - second_right, second_x - first_right, 0)
    vertical_gap = max(first_y - second_bottom, second_y - first_bottom, 0)

    minimum_width = max(1, min(first_width, second_width))
    minimum_height = max(1, min(first_height, second_height))
    vertical_lines = first.orientation == second.orientation == "vertical"
    horizontal_lines = first.orientation == second.orientation == "horizontal"
    if vertical_lines:
        return (
            vertical_overlap / minimum_height >= 0.08
            and horizontal_gap <= max(8, minimum_width * 2)
        )
    if horizontal_lines:
        return (
            horizontal_overlap / minimum_width >= 0.08
            and vertical_gap <= max(8, minimum_height * 2)
        )
    return (
        horizontal_overlap / minimum_width >= 0.08
        and vertical_overlap / minimum_height >= 0.08
    )


def _merge_detected_regions(group: Sequence[DetectedRegion]) -> DetectedRegion:
    x1 = min(region.bbox[0] for region in group)
    y1 = min(region.bbox[1] for region in group)
    x2 = max(region.bbox[0] + region.bbox[2] for region in group)
    y2 = max(region.bbox[1] + region.bbox[3] for region in group)
    polygon = [
        Point(x=x1, y=y1),
        Point(x=x2, y=y1),
        Point(x=x2, y=y2),
        Point(x=x1, y=y2),
    ]
    segmentations = [
        segmentation
        for region in group
        for segmentation in (region.segmentation or [region.polygon])
    ]
    confidences = [
        region.recognition_confidence
        for region in group
        if region.recognition_confidence is not None
    ]
    detector_confidences = [
        region.detector_confidence
        for region in group
        if region.detector_confidence is not None
    ]
    metadata = dict(group[0].metadata)
    metadata["group_size"] = len(group)
    return DetectedRegion(
        polygon=polygon,
        bbox=(x1, y1, x2 - x1, y2 - y1),
        detector_confidence=(
            sum(detector_confidences) / len(detector_confidences)
            if detector_confidences
            else None
        ),
        orientation="vertical" if (y2 - y1) > (x2 - x1) else "horizontal",
        source_text="\n".join(
            region.source_text.strip()
            for region in group
            if region.source_text and region.source_text.strip()
        ),
        recognition_confidence=(sum(confidences) / len(confidences) if confidences else None),
        segmentation=segmentations,
        metadata=metadata,
    )


def _result_values(result: Any, key: str) -> list[Any]:
    """Read PaddleOCR result fields across result object versions."""
    value: Any = None
    if isinstance(result, dict):
        value = result.get(key)
    else:
        try:
            value = result[key]
        except (AttributeError, KeyError, IndexError, TypeError):
            value = getattr(result, key, None)

    if value is None and hasattr(result, "res"):
        nested = result.res
        if isinstance(nested, dict):
            value = nested.get(key)
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _points_from_paddle(points: Any) -> list[Point]:
    if hasattr(points, "tolist"):
        points = points.tolist()
    result: list[Point] = []
    for point in points or []:
        if hasattr(point, "tolist"):
            point = point.tolist()
        if len(point) >= 2:
            result.append(Point(x=int(round(point[0])), y=int(round(point[1]))))
    return result


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _detected_region_from_polygon(
    polygon: list[Point],
    recognition_score: float | None,
    source_text: str | None,
    *,
    detector_confidence: float | None = None,
    metadata: dict[str, object],
) -> DetectedRegion:
    x_values = [point.x for point in polygon]
    y_values = [point.y for point in polygon]
    x = min(x_values)
    y = min(y_values)
    width = max(x_values) - x
    height = max(y_values) - y
    return DetectedRegion(
        polygon=polygon,
        bbox=(x, y, width, height),
        detector_confidence=(
            detector_confidence * 100 if detector_confidence is not None else None
        ),
        orientation="vertical" if height > width else "horizontal",
        source_text=source_text,
        recognition_confidence=(
            recognition_score * 100 if recognition_score is not None else None
        ),
        segmentation=[polygon],
        metadata=metadata,
    )
