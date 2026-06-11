from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


TRAFFIC_LIGHT_COLOR_RANGES = {
    "red": (((0, 80, 110), (10, 255, 255)), ((170, 80, 110), (179, 255, 255))),
    "yellow": (((16, 80, 115), (42, 255, 255)),),
    "green": (((42, 70, 90), (96, 255, 255)),),
}


@dataclass
class TrafficLightCameraConfig:
    roi_top_ratio: float = 0.02
    roi_bottom_ratio: float = 0.72
    min_area_px: int = 18
    max_area_ratio: float = 0.20
    min_score: float = 35.0
    min_fill_ratio: float = 0.42
    max_component_width_ratio: float = 0.14
    max_component_height_ratio: float = 0.16
    green_max_area_ratio: float = 0.02
    green_max_center_y_ratio: float = 0.68


@dataclass
class TrafficLightCameraDetection:
    state: str
    score: float
    area_px: int
    center_x: int
    center_y: int
    left_x: int
    top_y: int
    right_x: int
    bottom_y: int
    width_px: int
    height_px: int
    fill_ratio: float
    frame_width: int
    frame_height: int


class TrafficLightCameraDetector:
    """Camera-only red/yellow/green traffic-light detector.

    The detector deliberately scans all colors every frame. It does not accept
    or inspect the simulated traffic-light state, so it cannot use simulation
    ground truth to choose the color to detect.
    """

    def __init__(self, config: TrafficLightCameraConfig | None = None) -> None:
        self.config = config or TrafficLightCameraConfig()

    def detect(self, frame_bgr: np.ndarray) -> TrafficLightCameraDetection | None:
        height, width = frame_bgr.shape[:2]
        roi_top = int(height * self.config.roi_top_ratio)
        roi_bottom = int(height * self.config.roi_bottom_ratio)
        if roi_bottom <= roi_top:
            return None

        roi = frame_bgr[roi_top:roi_bottom, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_height = max(1, roi_bottom - roi_top)
        max_area = max(
            self.config.min_area_px,
            int(width * roi_height * self.config.max_area_ratio),
        )
        kernel = np.ones((3, 3), np.uint8)
        best_detection: TrafficLightCameraDetection | None = None

        for state, ranges in TRAFFIC_LIGHT_COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                lower_np = np.array(lower, dtype=np.uint8)
                upper_np = np.array(upper, dtype=np.uint8)
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_np, upper_np))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            candidate = self._best_component(state, hsv, mask, roi_top, height, max_area)
            if candidate is None:
                continue
            if best_detection is None or candidate.score > best_detection.score:
                best_detection = candidate

        if best_detection is None or best_detection.score < self.config.min_score:
            return None
        return best_detection

    def _best_component(
        self,
        state: str,
        hsv: np.ndarray,
        mask: np.ndarray,
        roi_top: int,
        frame_height: int,
        max_area: int,
    ) -> TrafficLightCameraDetection | None:
        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best: TrafficLightCameraDetection | None = None
        frame_width = hsv.shape[1]
        roi_height = hsv.shape[0]

        for label_index in range(1, component_count):
            width_px = int(stats[label_index, cv2.CC_STAT_WIDTH])
            height_px = int(stats[label_index, cv2.CC_STAT_HEIGHT])
            area_px = int(stats[label_index, cv2.CC_STAT_AREA])
            left_x = int(stats[label_index, cv2.CC_STAT_LEFT])
            top_y = int(stats[label_index, cv2.CC_STAT_TOP]) + roi_top
            right_x = left_x + width_px
            bottom_y = top_y + height_px
            if area_px < self.config.min_area_px or area_px > max_area:
                continue
            if width_px <= 0 or height_px <= 0:
                continue

            fill_ratio = area_px / float(max(1, width_px * height_px))
            center_x = int(round(float(centroids[label_index][0])))
            center_y = int(round(float(centroids[label_index][1]) + roi_top))
            if not self._shape_allows(
                state,
                width_px,
                height_px,
                area_px,
                fill_ratio,
                center_y,
                frame_width,
                roi_height,
                frame_height,
            ):
                continue

            if center_x < int(frame_width * 0.04) or center_x > int(frame_width * 0.96):
                continue

            component_mask = labels == label_index
            mean_s = float(np.mean(hsv[:, :, 1][component_mask]))
            mean_v = float(np.mean(hsv[:, :, 2][component_mask]))
            score = area_px * (mean_s / 255.0) * (mean_v / 255.0)
            detection = TrafficLightCameraDetection(
                state=state,
                score=score,
                area_px=area_px,
                center_x=center_x,
                center_y=center_y,
                left_x=left_x,
                top_y=top_y,
                right_x=right_x,
                bottom_y=bottom_y,
                width_px=width_px,
                height_px=height_px,
                fill_ratio=fill_ratio,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if best is None or detection.score > best.score:
                best = detection

        return best

    def _shape_allows(
        self,
        state: str,
        width_px: int,
        height_px: int,
        area_px: int,
        fill_ratio: float,
        center_y_px: int,
        frame_width: int,
        roi_height: int,
        frame_height: int,
    ) -> bool:
        aspect = max(width_px / float(max(1, height_px)), height_px / float(max(1, width_px)))
        max_aspect = {"red": 3.5, "yellow": 2.8, "green": 4.0}.get(state, 4.0)
        if aspect > max_aspect:
            return False

        min_fill = max(
            self.config.min_fill_ratio,
            {"red": 0.42, "yellow": 0.50, "green": 0.58}.get(state, self.config.min_fill_ratio),
        )
        if fill_ratio < min_fill:
            return False

        if state == "green":
            max_green_area = max(
                self.config.min_area_px,
                int(frame_width * roi_height * self.config.green_max_area_ratio),
            )
            if area_px > max_green_area:
                return False
            if center_y_px > int(frame_height * self.config.green_max_center_y_ratio):
                return False

        max_width = max(18, int(frame_width * self.config.max_component_width_ratio))
        max_height = max(18, int(roi_height * self.config.max_component_height_ratio))
        return width_px <= max_width and height_px <= max_height
