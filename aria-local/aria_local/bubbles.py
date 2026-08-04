from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from .models import Point


@dataclass(frozen=True)
class BubbleMatch:
    id: int
    bbox: tuple[int, int, int, int]
    text_bbox: tuple[int, int, int, int]
    contour: tuple[tuple[int, int], ...]
    fill_tone: int
    component_set: int
    label: int


def polygon_geometry_mask(
    image_shape: tuple[int, ...], polygons: Sequence[Sequence[Point]]
) -> np.ndarray:
    """Rasterize detector geometry without expanding it to a bounding box."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for points in polygons:
        polygon = np.array([[point.x, point.y] for point in points], dtype=np.int32)
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [polygon], 255)
    return mask


class WhiteBubbleLocator:
    """Match text geometry to enclosed light speech-bubble components."""

    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image_scale = max(1.0, min(image.shape[:2]) / 1000)
        light_mask = cv2.inRange(self.gray, 225, 255)
        self.separation_radius = min(12, round(3 * self.image_scale))
        separated_light_mask = cv2.erode(
            light_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    self.separation_radius * 2 + 1,
                    self.separation_radius * 2 + 1,
                ),
            ),
        )
        self.component_sets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        # Prefer components with narrow scan gaps removed so two neighboring
        # bubbles cannot become one group through a broken outline.
        for mask in (separated_light_mask, light_mask):
            _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            self.component_sets.append((mask, labels, stats))

        self._matches: dict[tuple[int, int], BubbleMatch] = {}
        self._representatives: list[BubbleMatch] = []
        self._valid_boundaries: dict[tuple[int, int], bool] = {}
        self._adaptive_component_sets: dict[int, int] = {}
        self._adaptive_raw_masks: dict[int, np.ndarray] = {}

    def find(self, geometry: np.ndarray) -> BubbleMatch | None:
        for component_set, (light_mask, labels, stats) in enumerate(
            self.component_sets[:2]
        ):
            label = self._find_component_label(
                component_set, geometry, light_mask, labels, stats
            )
            if label is None:
                continue
            if component_set == 1 and self._has_multiple_separated_parts(label, stats):
                continue
            return self._match(component_set, label, stats)

        adaptive_set = self._adaptive_component_set(geometry)
        if adaptive_set is None:
            return None
        component_set, created, tone_bin = adaptive_set
        light_mask, labels, stats = self.component_sets[component_set]
        label = self._find_component_label(
            component_set, geometry, light_mask, labels, stats
        )
        if label is None:
            if created:
                self.component_sets.pop()
                self._adaptive_component_sets.pop(tone_bin, None)
                self._adaptive_raw_masks.pop(component_set, None)
                self._valid_boundaries = {
                    key: valid
                    for key, valid in self._valid_boundaries.items()
                    if key[0] != component_set
                }
            return None
        return self._match(component_set, label, stats)

    def _match(self, component_set: int, label: int, stats: np.ndarray) -> BubbleMatch:
        key = (component_set, label)
        existing = self._matches.get(key)
        if existing is not None:
            return existing

        bbox = tuple(int(value) for value in stats[label, :4])
        text_bbox, contour = self._bubble_geometry(component_set, label, bbox)
        x, y, width, height = bbox
        labels = self.component_sets[component_set][1]
        component_pixels = self.gray[y : y + height, x : x + width][
            labels[y : y + height, x : x + width] == label
        ]
        match = BubbleMatch(
            id=self._canonical_id(component_set, label, bbox),
            bbox=bbox,
            text_bbox=text_bbox,
            contour=contour,
            fill_tone=(
                round(float(np.median(component_pixels)))
                if component_pixels.size
                else 255
            ),
            component_set=component_set,
            label=label,
        )
        self._matches[key] = match
        if not any(candidate.id == match.id for candidate in self._representatives):
            self._representatives.append(match)
        return match

    def _adaptive_component_set(
        self, geometry: np.ndarray
    ) -> tuple[int, bool, int] | None:
        _, _, width, height = cv2.boundingRect(geometry)
        if width <= 0 or height <= 0:
            return None
        context_radius = self._context_radius(width, height)
        context = cv2.dilate(
            geometry,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (context_radius * 2 + 1, context_radius * 2 + 1),
            ),
        )
        samples = self.gray[(context > 0) & (self.gray > 160)]
        if samples.size == 0:
            return None
        bins = np.bincount(samples // 16, minlength=16)
        tone_bin = int(np.argmax(bins))
        tone = tone_bin * 16 + 8
        if tone < 200:
            return None

        existing = self._adaptive_component_sets.get(tone_bin)
        if existing is not None:
            return existing, False, tone_bin
        if len(self._adaptive_component_sets) >= 2:
            return None

        raw_tone_mask = cv2.inRange(
            self.gray,
            max(161, tone - 20),
            min(255, tone + 20),
        )
        tone_mask = cv2.erode(
            raw_tone_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    self.separation_radius * 2 + 1,
                    self.separation_radius * 2 + 1,
                ),
            ),
        )
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            tone_mask, connectivity=4
        )
        component_set = len(self.component_sets)
        self.component_sets.append((tone_mask, labels, stats))
        self._adaptive_component_sets[tone_bin] = component_set
        self._adaptive_raw_masks[component_set] = raw_tone_mask
        return component_set, True, tone_bin

    def _context_radius(self, width: int, height: int) -> int:
        return max(
            round(6 * self.image_scale),
            min(
                round(24 * self.image_scale),
                round(min(width, height) * 0.25),
            ),
        )

    def mask(self, match: BubbleMatch) -> np.ndarray:
        labels = self.component_sets[match.component_set][1]
        return np.where(labels == match.label, 255, 0).astype(np.uint8)

    def geometry_overlap(self, match: BubbleMatch, geometry: np.ndarray) -> float:
        geometry_area = cv2.countNonZero(geometry)
        if geometry_area == 0:
            return 0
        labels = self.component_sets[match.component_set][1]
        return np.count_nonzero(labels[geometry > 0] == match.label) / geometry_area

    def _bubble_geometry(
        self,
        component_set: int,
        label: int,
        bbox: tuple[int, int, int, int],
    ) -> tuple[tuple[int, int, int, int], tuple[tuple[int, int], ...]]:
        x, y, width, height = bbox
        labels = self.component_sets[component_set][1]
        component = np.where(
            labels[y : y + height, x : x + width] == label, 255, 0
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filled = np.zeros_like(component)
        if not contours:
            return (x, y, 1, 1), ()
        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
        page_contour = tuple(
            (x + int(px), y + int(py)) for px, py in contour.reshape(-1, 2)
        )

        margin = max(2, round(min(width, height) * 0.03))
        safe = cv2.erode(
            filled,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1)
            ),
        )
        rectangle = _largest_centered_rectangle(
            safe if cv2.countNonZero(safe) else filled
        )
        if rectangle is None:
            return (x, y, 1, 1), page_contour
        local_x, local_y, rect_width, rect_height = rectangle
        return (
            (x + local_x, y + local_y, rect_width, rect_height),
            page_contour,
        )

    def _canonical_id(
        self,
        component_set: int,
        label: int,
        bbox: tuple[int, int, int, int],
    ) -> int:
        matching_ids = {
            representative.id
            for representative in self._representatives
            if self._components_overlap(component_set, label, bbox, representative)
        }
        if len(matching_ids) == 1:
            return matching_ids.pop()
        return len(self._representatives) + 1

    def _has_multiple_separated_parts(self, label: int, stats: np.ndarray) -> bool:
        full_labels = self.component_sets[1][1]
        separated_labels = self.component_sets[0][1]
        labels, counts = np.unique(
            separated_labels[full_labels == label], return_counts=True
        )
        full_area = int(stats[label, cv2.CC_STAT_AREA])
        minimum_area = max(round(256 * self.image_scale**2), int(full_area * 0.10))
        significant_parts = sum(
            int(part_label) > 0 and int(count) >= minimum_area
            for part_label, count in zip(labels, counts)
        )
        return significant_parts > 1

    def _components_overlap(
        self,
        component_set: int,
        label: int,
        bbox: tuple[int, int, int, int],
        other: BubbleMatch,
    ) -> bool:
        x1 = max(bbox[0], other.bbox[0])
        y1 = max(bbox[1], other.bbox[1])
        x2 = min(bbox[0] + bbox[2], other.bbox[0] + other.bbox[2])
        y2 = min(bbox[1] + bbox[3], other.bbox[1] + other.bbox[3])
        if x2 <= x1 or y2 <= y1:
            return False

        labels = self.component_sets[component_set][1][y1:y2, x1:x2]
        other_labels = self.component_sets[other.component_set][1][y1:y2, x1:x2]
        overlap = np.count_nonzero((labels == label) & (other_labels == other.label))
        first_area = int(self.component_sets[component_set][2][label, cv2.CC_STAT_AREA])
        second_area = int(
            self.component_sets[other.component_set][2][other.label, cv2.CC_STAT_AREA]
        )
        return overlap / max(1, min(first_area, second_area)) >= 0.65

    def _find_component_label(
        self,
        component_set: int,
        geometry: np.ndarray,
        light_mask: np.ndarray,
        light_labels: np.ndarray,
        light_stats: np.ndarray,
    ) -> int | None:
        geometry_area = cv2.countNonZero(geometry)
        if geometry_area == 0:
            return None

        x, y, width, height = cv2.boundingRect(geometry)
        context_radius = self._context_radius(width, height)
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

        candidate_labels = light_labels[geometry > 0]
        candidate_labels = candidate_labels[candidate_labels > 0]
        if candidate_labels.size == 0:
            candidate_labels = light_labels[context > 0]
            candidate_labels = candidate_labels[candidate_labels > 0]
        if candidate_labels.size == 0:
            return None
        labels, counts = np.unique(candidate_labels, return_counts=True)
        ordered_labels = labels[np.argsort(-counts)]

        for raw_label in ordered_labels:
            label = int(raw_label)
            if self._component_is_valid(
                component_set,
                label,
                geometry_area,
                x,
                y,
                width,
                height,
                light_labels,
                light_stats,
            ):
                return label
        return None

    def _component_is_valid(
        self,
        component_set: int,
        label: int,
        geometry_area: int,
        geometry_x: int,
        geometry_y: int,
        geometry_width: int,
        geometry_height: int,
        light_labels: np.ndarray,
        light_stats: np.ndarray,
    ) -> bool:
        component_x, component_y, component_width, component_height, component_area = (
            int(value) for value in light_stats[label]
        )

        image_height, image_width = self.image.shape[:2]
        if (
            component_x <= 1
            or component_y <= 1
            or component_x + component_width >= image_width - 1
            or component_y + component_height >= image_height - 1
        ):
            return False
        if component_width * component_height > image_width * image_height * 0.45:
            return False
        minimum_component_area = round(256 * self.image_scale**2)
        if component_area < max(minimum_component_area, int(geometry_area * 0.35)):
            return False

        geometry_center_x = geometry_x + geometry_width / 2
        geometry_center_y = geometry_y + geometry_height / 2
        if not (
            component_x <= geometry_center_x <= component_x + component_width
            and component_y <= geometry_center_y <= component_y + component_height
        ):
            return False

        key = (component_set, label)
        if key not in self._valid_boundaries:
            component = np.where(light_labels == label, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            filled = np.zeros_like(component)
            if contours:
                cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
            if component_set >= 2:
                raw_mask = self._adaptive_raw_masks.get(component_set)
                if raw_mask is not None:
                    raw_contours, _ = cv2.findContours(
                        raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    component_center = (
                        component_x + component_width / 2,
                        component_y + component_height / 2,
                    )
                    enclosing = [
                        contour
                        for contour in raw_contours
                        if cv2.pointPolygonTest(contour, component_center, False) >= 0
                    ]
                    if enclosing:
                        filled = np.zeros_like(component)
                        cv2.drawContours(
                            filled,
                            [min(enclosing, key=cv2.contourArea)],
                            -1,
                            255,
                            cv2.FILLED,
                        )
            boundary_radius = min(16, round(4 * self.image_scale))
            boundary_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (boundary_radius * 2 + 1, boundary_radius * 2 + 1),
            )
            boundary = cv2.subtract(cv2.dilate(filled, boundary_kernel), filled)
            boundary_dark_threshold = 80 if component_set >= 2 else 160
            dark_boundary = cv2.bitwise_and(
                boundary,
                cv2.inRange(self.gray, 0, boundary_dark_threshold),
            )
            boundary_area = cv2.countNonZero(boundary)
            valid = (
                boundary_area > 0
                and cv2.countNonZero(dark_boundary) / boundary_area >= 0.03
            )
            if valid and component_set >= 2:
                outer_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (boundary_radius * 4 + 1, boundary_radius * 4 + 1),
                )
                outer_ring = cv2.subtract(
                    cv2.dilate(filled, outer_kernel),
                    cv2.dilate(filled, boundary_kernel),
                )
                outer_area = cv2.countNonZero(outer_ring)
                outer_dark = cv2.bitwise_and(
                    outer_ring,
                    cv2.inRange(self.gray, 0, boundary_dark_threshold),
                )
                valid = (
                    outer_area > 0 and cv2.countNonZero(outer_dark) / outer_area <= 0.50
                )

            if valid and component_set >= 2:
                center_x = component_x + component_width // 2
                center_y = component_y + component_height // 2
                outer_left = max(0, component_x - boundary_radius)
                outer_top = max(0, component_y - boundary_radius)
                outer_right = min(
                    self.image.shape[1],
                    component_x + component_width + boundary_radius,
                )
                outer_bottom = min(
                    self.image.shape[0],
                    component_y + component_height + boundary_radius,
                )
                quadrants = (
                    (outer_left, outer_top, center_x, center_y),
                    (center_x, outer_top, outer_right, center_y),
                    (outer_left, center_y, center_x, outer_bottom),
                    (center_x, center_y, outer_right, outer_bottom),
                )
                supported_quadrants = 0
                for left, top, right, bottom in quadrants:
                    quadrant_boundary = boundary[top:bottom, left:right]
                    quadrant_area = cv2.countNonZero(quadrant_boundary)
                    if quadrant_area and (
                        cv2.countNonZero(dark_boundary[top:bottom, left:right])
                        / quadrant_area
                        >= 0.015
                    ):
                        supported_quadrants += 1
                valid = supported_quadrants >= 3
            self._valid_boundaries[key] = valid
        return self._valid_boundaries[key]


def _largest_centered_rectangle(
    mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Find the largest foreground rectangle containing the bubble's visual center."""
    if not cv2.countNonZero(mask):
        return None

    source_height, source_width = mask.shape
    scale = min(1.0, 512 / max(mask.shape))
    if scale < 1:
        analysis_mask = cv2.resize(
            mask,
            (max(1, round(source_width * scale)), max(1, round(source_height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        analysis_mask = np.where(analysis_mask == 255, 255, 0).astype(np.uint8)
    else:
        analysis_mask = mask

    distance = cv2.distanceTransform(analysis_mask, cv2.DIST_L2, 5)
    weights = distance**2
    total_weight = float(weights.sum())
    if total_weight:
        center_x = round(
            float(np.dot(weights.sum(axis=0), np.arange(analysis_mask.shape[1])))
            / total_weight
        )
        center_y = round(
            float(np.dot(weights.sum(axis=1), np.arange(analysis_mask.shape[0])))
            / total_weight
        )
    else:
        center_y, center_x = np.unravel_index(int(distance.argmax()), distance.shape)

    center_x = max(0, min(center_x, analysis_mask.shape[1] - 1))
    center_y = max(0, min(center_y, analysis_mask.shape[0] - 1))
    if analysis_mask[center_y, center_x] == 0:
        center_y, center_x = np.unravel_index(int(distance.argmax()), distance.shape)

    spans: list[tuple[int, int] | None] = []
    foreground = analysis_mask > 0
    for row in foreground:
        if not row[center_x]:
            spans.append(None)
            continue
        left_background = np.flatnonzero(~row[:center_x])
        right_background = np.flatnonzero(~row[center_x + 1 :])
        left = int(left_background[-1] + 1) if left_background.size else 0
        right = (
            int(center_x + 1 + right_background[0])
            if right_background.size
            else analysis_mask.shape[1]
        )
        spans.append((left, right))

    best: tuple[int, int, int, int] | None = None
    best_area = 0
    upper_left, upper_right = 0, analysis_mask.shape[1]
    for top in range(center_y, -1, -1):
        span = spans[top]
        if span is None:
            break
        upper_left = max(upper_left, span[0])
        upper_right = min(upper_right, span[1])
        lower_left, lower_right = upper_left, upper_right
        for bottom in range(center_y, analysis_mask.shape[0]):
            span = spans[bottom]
            if span is None:
                break
            lower_left = max(lower_left, span[0])
            lower_right = min(lower_right, span[1])
            area = max(0, lower_right - lower_left) * (bottom - top + 1)
            if area > best_area:
                best_area = area
                best = (lower_left, top, lower_right - lower_left, bottom - top + 1)

    if best is None or scale == 1:
        return best
    x, y, width, height = best
    scale_x = analysis_mask.shape[1] / source_width
    scale_y = analysis_mask.shape[0] / source_height
    left = math.ceil(x / scale_x)
    top = math.ceil(y / scale_y)
    right = math.floor((x + width) / scale_x)
    bottom = math.floor((y + height) / scale_y)
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top
