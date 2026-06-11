from __future__ import annotations

import json
import time
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from my_robot_vision.world import share_file


CAMERA_SIGN_TEXTURES = (
    (
        "camera_sign_go_straight",
        "go_straight_sign",
        "straight",
        ("models", "road_sign_go_straight", "materials", "textures", "go_straight.jpg"),
    ),
    (
        "camera_sign_turn_left",
        "turn_left_sign",
        "left",
        ("models", "road_sign_turn_left", "materials", "textures", "turn_left.png"),
    ),
    (
        "camera_sign_turn_right",
        "turn_right_sign",
        "right",
        ("models", "road_sign_turn_right", "materials", "textures", "turn_right.png"),
    ),
    (
        "camera_sign_parking",
        "parking_sign",
        "parking",
        ("models", "road_sign_parking", "materials", "textures", "parking.png"),
    ),
    (
        "camera_sign_speed_limit",
        "speed_limit_sign",
        None,
        ("models", "road_sign_speed_limit", "materials", "textures", "limit_speed.jpg"),
    ),
)

CAMERA_SIGN_FEATURE_SIZE = (96, 96)
CAMERA_SIGN_MANEUVER_KINDS = frozenset({"go_straight_sign", "turn_left_sign", "turn_right_sign"})
CAMERA_SIGN_MANEUVER_CENTER_MIN_RATIO = 0.20
CAMERA_SIGN_MANEUVER_CENTER_MAX_RATIO = 0.80
CAMERA_SIGN_TURN_AMBIGUITY_MARGIN = 0.035
CAMERA_SIGN_MANEUVER_AMBIGUITY_MARGIN = 0.030
CAMERA_SIGN_TURN_OPPOSITE_KINDS = {
    "turn_left_sign": "turn_right_sign",
    "turn_right_sign": "turn_left_sign",
}


@dataclass
class CameraSignTemplate:
    name: str
    kind: str
    maneuver: str | None
    feature: np.ndarray
    blue_ratio: float
    red_ratio: float
    white_ratio: float


@dataclass
class BlueGlyphMetrics:
    internal_area_px: int
    internal_ratio: float
    span_x_ratio: float
    span_y_ratio: float
    center_x_ratio: float | None
    top_center_x_ratio: float | None
    bottom_center_x_ratio: float | None


@dataclass
class RoadSignCameraDetection:
    name: str
    kind: str
    maneuver: str | None
    score: float
    area_px: int
    colored_area_px: int
    center_x: int
    center_y: int
    left_x: int
    top_y: int
    right_x: int
    bottom_y: int
    width_px: int
    height_px: int
    frame_width: int
    frame_height: int
    center_ratio: float
    blue_ratio: float
    red_ratio: float
    white_ratio: float
    scores_by_kind: dict[str, float]
    ambiguous: bool = False


@dataclass
class RoadSignCameraConfig:
    roi_top_ratio: float = 0.08
    roi_bottom_ratio: float = 0.84
    min_colored_area_px: int = 55
    min_bbox_height_px: int = 14
    min_score: float = 0.30
    action_min_area_px: int = 1500
    action_min_height_px: int = 34
    stable_frames: int = 2
    maneuver_stable_frames: int = 4


class RoadSignCameraDetector:
    """Camera-only road sign detector based on HSV candidate crops and templates."""

    def __init__(self, config: RoadSignCameraConfig | None = None) -> None:
        self.config = config or RoadSignCameraConfig()
        self.templates = self.load_templates()

    def load_templates(self) -> list[CameraSignTemplate]:
        templates: list[CameraSignTemplate] = []
        for name, kind, maneuver, path_parts in CAMERA_SIGN_TEXTURES:
            path = share_file(*path_parts)
            if path is None:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            feature, blue_ratio, red_ratio, white_ratio = self.sign_feature(image)
            templates.append(
                CameraSignTemplate(
                    name=name,
                    kind=kind,
                    maneuver=maneuver,
                    feature=feature,
                    blue_ratio=blue_ratio,
                    red_ratio=red_ratio,
                    white_ratio=white_ratio,
                )
            )
        return templates

    def canonical_sign_image(self, image: np.ndarray) -> np.ndarray:
        if image.size == 0:
            return image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array((85, 45, 40), dtype=np.uint8), np.array((135, 255, 255), dtype=np.uint8))
        red_low = cv2.inRange(hsv, np.array((0, 55, 55), dtype=np.uint8), np.array((12, 255, 255), dtype=np.uint8))
        red_high = cv2.inRange(hsv, np.array((168, 55, 55), dtype=np.uint8), np.array((179, 255, 255), dtype=np.uint8))
        color = cv2.bitwise_or(blue, cv2.bitwise_or(red_low, red_high))
        kernel = np.ones((3, 3), np.uint8)
        color = cv2.morphologyEx(color, cv2.MORPH_CLOSE, kernel)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(color, connectivity=8)
        if component_count <= 1:
            return image

        best_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[best_index, cv2.CC_STAT_AREA])
        if area < 12:
            return image
        x = int(stats[best_index, cv2.CC_STAT_LEFT])
        y = int(stats[best_index, cv2.CC_STAT_TOP])
        w = int(stats[best_index, cv2.CC_STAT_WIDTH])
        h = int(stats[best_index, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            return image

        # Keep the normalized crop tight to the colored sign face.  A wide pad
        # lets white sign boards or walls look like an arrow/P glyph.
        pad_x = max(1, int(round(w * 0.02)))
        pad_y = max(1, int(round(h * 0.02)))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(image.shape[1], x + w + pad_x)
        y1 = min(image.shape[0], y + h + pad_y)
        if x1 <= x0 or y1 <= y0:
            return image
        return image[y0:y1, x0:x1]

    def sign_feature(self, image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        image = self.canonical_sign_image(image)
        resized = cv2.resize(image, CAMERA_SIGN_FEATURE_SIZE, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array((85, 45, 45), dtype=np.uint8), np.array((135, 255, 255), dtype=np.uint8))
        red_low = cv2.inRange(hsv, np.array((0, 55, 55), dtype=np.uint8), np.array((12, 255, 255), dtype=np.uint8))
        red_high = cv2.inRange(hsv, np.array((168, 55, 55), dtype=np.uint8), np.array((179, 255, 255), dtype=np.uint8))
        red = cv2.bitwise_or(red_low, red_high)
        white = cv2.inRange(hsv, np.array((0, 0, 135), dtype=np.uint8), np.array((179, 92, 255), dtype=np.uint8))
        dark = cv2.inRange(hsv, np.array((0, 0, 0), dtype=np.uint8), np.array((179, 255, 75), dtype=np.uint8))
        feature = np.dstack([blue, red, white, dark]).astype(np.float32) / 255.0
        feature = cv2.GaussianBlur(feature, (3, 3), 0)
        pixel_count = float(CAMERA_SIGN_FEATURE_SIZE[0] * CAMERA_SIGN_FEATURE_SIZE[1])
        return (
            feature,
            float(np.count_nonzero(blue)) / pixel_count,
            float(np.count_nonzero(red)) / pixel_count,
            float(np.count_nonzero(white)) / pixel_count,
        )

    def blue_glyph_metrics(self, image: np.ndarray) -> BlueGlyphMetrics:
        image = self.canonical_sign_image(image)
        if image.size == 0:
            return BlueGlyphMetrics(0, 0.0, 0.0, 0.0, None, None, None)

        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array((0, 0, 135), dtype=np.uint8), np.array((179, 92, 255), dtype=np.uint8))
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)

        internal = np.zeros_like(white)
        border_margin = max(1, int(round(min(width, height) * 0.03)))
        min_component_area = max(6, int(round(width * height * 0.001)))
        for label_index in range(1, component_count):
            area = int(stats[label_index, cv2.CC_STAT_AREA])
            if area < min_component_area:
                continue
            x = int(stats[label_index, cv2.CC_STAT_LEFT])
            y = int(stats[label_index, cv2.CC_STAT_TOP])
            w = int(stats[label_index, cv2.CC_STAT_WIDTH])
            h = int(stats[label_index, cv2.CC_STAT_HEIGHT])
            touches_border = (
                x <= border_margin
                or y <= border_margin
                or x + w >= width - border_margin
                or y + h >= height - border_margin
            )
            if touches_border:
                continue
            internal[labels == label_index] = 255

        ys, xs = np.where(internal > 0)
        if xs.size == 0:
            return BlueGlyphMetrics(0, 0.0, 0.0, 0.0, None, None, None)

        top_mask = ys < height * 0.55
        bottom_mask = ys >= height * 0.45
        top_center = float(np.mean(xs[top_mask]) / max(1, width)) if np.any(top_mask) else None
        bottom_center = float(np.mean(xs[bottom_mask]) / max(1, width)) if np.any(bottom_mask) else None
        span_x = float((int(np.max(xs)) + 1 - int(np.min(xs))) / max(1, width))
        span_y = float((int(np.max(ys)) + 1 - int(np.min(ys))) / max(1, height))
        return BlueGlyphMetrics(
            internal_area_px=int(xs.size),
            internal_ratio=float(xs.size / max(1, width * height)),
            span_x_ratio=span_x,
            span_y_ratio=span_y,
            center_x_ratio=float(np.mean(xs) / max(1, width)),
            top_center_x_ratio=top_center,
            bottom_center_x_ratio=bottom_center,
        )

    def template_score(
        self,
        feature: np.ndarray,
        blue_ratio: float,
        red_ratio: float,
        white_ratio: float,
        template: CameraSignTemplate,
    ) -> float:
        weights = np.array((0.28, 0.30, 0.32, 0.10), dtype=np.float32)
        weighted_feature = feature * weights
        weighted_template = template.feature * weights
        numerator = float(np.sum(weighted_feature * weighted_template))
        denominator = float(np.linalg.norm(weighted_feature) * np.linalg.norm(weighted_template))
        feature_score = 0.0 if denominator <= 1e-6 else max(0.0, min(1.0, numerator / denominator))
        color_error = (
            abs(blue_ratio - template.blue_ratio)
            + abs(red_ratio - template.red_ratio)
            + 0.5 * abs(white_ratio - template.white_ratio)
        )
        color_score = max(0.0, 1.0 - 2.5 * color_error)
        return 0.82 * feature_score + 0.18 * color_score

    def template_sanity_allows(
        self,
        template: CameraSignTemplate,
        blue_ratio: float,
        red_ratio: float,
        white_ratio: float,
        crop_width: int,
        crop_height: int,
        blue_glyph: BlueGlyphMetrics | None = None,
    ) -> bool:
        aspect = crop_width / float(max(1, crop_height))
        if aspect < 0.35 or aspect > 3.0:
            return False
        if template.kind == "speed_limit_sign":
            return red_ratio >= 0.08 and white_ratio >= 0.12 and blue_ratio <= 0.20
        if template.kind in {"go_straight_sign", "turn_left_sign", "turn_right_sign", "parking_sign"}:
            if blue_ratio < 0.06 or red_ratio > 0.22:
                return False
            if blue_glyph is None or blue_glyph.internal_area_px <= 0:
                return False
            if blue_glyph.internal_ratio < 0.045:
                return False
            if template.kind == "parking_sign" and blue_glyph.internal_ratio < 0.075:
                return False
        return True

    def score_with_blue_glyph(
        self,
        template: CameraSignTemplate,
        score: float,
        glyph: BlueGlyphMetrics | None,
    ) -> float:
        if glyph is None or glyph.internal_area_px <= 0:
            return score
        if template.kind not in {"go_straight_sign", "turn_left_sign", "turn_right_sign", "parking_sign"}:
            return score

        top = glyph.top_center_x_ratio
        bottom = glyph.bottom_center_x_ratio
        if top is not None and bottom is not None:
            direction_reliable = glyph.internal_area_px >= 500
            turn_delta = top - bottom
            if template.kind == "turn_right_sign":
                if direction_reliable and top > 0.54 and turn_delta > 0.08:
                    score += 0.060
                elif direction_reliable and turn_delta < -0.05:
                    score -= 0.085
            elif template.kind == "turn_left_sign":
                if direction_reliable and top < 0.46 and turn_delta < -0.08:
                    score += 0.060
                elif direction_reliable and turn_delta > 0.05:
                    score -= 0.085
            elif template.kind == "go_straight_sign":
                if direction_reliable and abs(turn_delta) < 0.10 and glyph.span_x_ratio < 0.52:
                    score += 0.050
                elif direction_reliable and abs(turn_delta) > 0.20:
                    score -= 0.070

        if template.kind == "parking_sign":
            if glyph.internal_ratio >= 0.12 and glyph.span_y_ratio >= 0.45:
                score += 0.035
            elif glyph.internal_ratio < 0.09:
                score -= 0.060

        return max(0.0, min(1.0, score))

    def classify_crop(
        self,
        crop: np.ndarray,
    ) -> tuple[CameraSignTemplate, float, dict[str, float], float, float, float, bool] | None:
        if crop.size == 0:
            return None
        feature, blue_ratio, red_ratio, white_ratio = self.sign_feature(crop)
        blue_glyph = self.blue_glyph_metrics(crop) if blue_ratio >= 0.04 and red_ratio <= 0.22 else None
        best_template: CameraSignTemplate | None = None
        best_score = 0.0
        scores_by_kind: dict[str, float] = {}
        templates_by_kind: dict[str, CameraSignTemplate] = {}
        crop_height, crop_width = crop.shape[:2]
        for template in self.templates:
            score = self.template_score(feature, blue_ratio, red_ratio, white_ratio, template)
            if not self.template_sanity_allows(
                template,
                blue_ratio,
                red_ratio,
                white_ratio,
                crop_width,
                crop_height,
                blue_glyph,
            ):
                continue
            score = self.score_with_blue_glyph(template, score, blue_glyph)
            if score > scores_by_kind.get(template.kind, 0.0):
                scores_by_kind[template.kind] = score
                templates_by_kind[template.kind] = template
            if score > best_score:
                best_template = template
                best_score = score

        if best_template is None:
            return None

        ambiguous = False
        opposite_kind = CAMERA_SIGN_TURN_OPPOSITE_KINDS.get(best_template.kind)
        if opposite_kind is not None:
            opposite_score = scores_by_kind.get(opposite_kind, 0.0)
            ambiguous = opposite_score >= self.config.min_score and opposite_score >= best_score - CAMERA_SIGN_TURN_AMBIGUITY_MARGIN
            # Keep the best visual answer for overlay, but mark ambiguous so decision logic can wait.
        if best_template.kind in CAMERA_SIGN_MANEUVER_KINDS:
            for kind, score in scores_by_kind.items():
                if kind == best_template.kind or kind not in CAMERA_SIGN_MANEUVER_KINDS:
                    continue
                if score >= self.config.min_score and score >= best_score - CAMERA_SIGN_MANEUVER_AMBIGUITY_MARGIN:
                    ambiguous = True
                    break

        return best_template, best_score, scores_by_kind, blue_ratio, red_ratio, white_ratio, ambiguous

    def detect(
        self,
        frame_bgr: np.ndarray,
        suppress_bboxes: list[tuple[int, int, int, int]] | None = None,
    ) -> RoadSignCameraDetection | None:
        if not self.templates:
            return None
        frame_height, frame_width = frame_bgr.shape[:2]
        roi_top = int(frame_height * self.config.roi_top_ratio)
        roi_bottom = int(frame_height * self.config.roi_bottom_ratio)
        if roi_bottom <= roi_top:
            return None

        roi = frame_bgr[roi_top:roi_bottom, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array((85, 45, 40), dtype=np.uint8), np.array((135, 255, 255), dtype=np.uint8))
        red_low = cv2.inRange(hsv, np.array((0, 55, 55), dtype=np.uint8), np.array((12, 255, 255), dtype=np.uint8))
        red_high = cv2.inRange(hsv, np.array((168, 55, 55), dtype=np.uint8), np.array((179, 255, 255), dtype=np.uint8))
        color_mask = cv2.bitwise_or(blue, cv2.bitwise_or(red_low, red_high))
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        for bbox in suppress_bboxes or []:
            x0, y0, x1, y1 = bbox
            mask_y0 = max(0, y0 - roi_top)
            mask_y1 = min(color_mask.shape[0], y1 - roi_top)
            if x1 > x0 and mask_y1 > mask_y0:
                color_mask[mask_y0:mask_y1, max(0, x0) : min(frame_width, x1)] = 0

        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
        best_detection: RoadSignCameraDetection | None = None
        best_priority = 0.0

        for label_index in range(1, component_count):
            colored_area = int(stats[label_index, cv2.CC_STAT_AREA])
            if colored_area < self.config.min_colored_area_px:
                continue
            x = int(stats[label_index, cv2.CC_STAT_LEFT])
            y = int(stats[label_index, cv2.CC_STAT_TOP])
            w = int(stats[label_index, cv2.CC_STAT_WIDTH])
            h = int(stats[label_index, cv2.CC_STAT_HEIGHT])
            if w <= 0 or h < self.config.min_bbox_height_px:
                continue
            if max(w / float(max(1, h)), h / float(max(1, w))) > 5.0:
                continue

            pad_x = max(5, int(round(w * 0.34)))
            pad_y = max(5, int(round(h * 0.34)))
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(frame_width, x + w + pad_x)
            y1 = min(roi_bottom - roi_top, y + h + pad_y)
            if x1 <= x0 or y1 <= y0:
                continue

            crop = roi[y0:y1, x0:x1]
            classified = self.classify_crop(crop)
            if classified is None:
                continue
            template, score, scores_by_kind, blue_ratio, red_ratio, white_ratio, ambiguous = classified
            if score < self.config.min_score:
                continue

            bbox_width = x1 - x0
            bbox_height = y1 - y0
            if template.kind == "speed_limit_sign":
                bbox_aspect = max(bbox_width / float(max(1, bbox_height)), bbox_height / float(max(1, bbox_width)))
                if bbox_aspect > 1.85:
                    continue

            bbox_area = bbox_width * bbox_height
            priority = score + min(0.20, bbox_area / float(max(1, frame_width * frame_height)) * 5.0)
            center_x = int(round(float(centroids[label_index][0])))
            center_y = int(round(float(centroids[label_index][1]) + roi_top))
            detection = RoadSignCameraDetection(
                name=template.name,
                kind=template.kind,
                maneuver=template.maneuver,
                score=score,
                area_px=bbox_area,
                colored_area_px=colored_area,
                center_x=center_x,
                center_y=center_y,
                left_x=x0,
                top_y=y0 + roi_top,
                right_x=x1,
                bottom_y=y1 + roi_top,
                width_px=bbox_width,
                height_px=bbox_height,
                frame_width=frame_width,
                frame_height=frame_height,
                center_ratio=center_x / float(max(1, frame_width)),
                blue_ratio=blue_ratio,
                red_ratio=red_ratio,
                white_ratio=white_ratio,
                scores_by_kind=scores_by_kind,
                ambiguous=ambiguous,
            )
            if best_detection is None or priority > best_priority:
                best_detection = detection
                best_priority = priority

        return best_detection

    def required_stable_frames(self, detection: RoadSignCameraDetection) -> int:
        if detection.maneuver in {"left", "right", "straight"}:
            return max(self.config.stable_frames, self.config.maneuver_stable_frames)
        return self.config.stable_frames

    def action_candidate(self, detection: RoadSignCameraDetection) -> bool:
        center_ok = CAMERA_SIGN_MANEUVER_CENTER_MIN_RATIO <= detection.center_ratio <= CAMERA_SIGN_MANEUVER_CENTER_MAX_RATIO
        if detection.maneuver in {"left", "right", "straight"} and not center_ok:
            return False
        if detection.ambiguous and detection.maneuver in {"left", "right", "straight"}:
            return False
        return (
            detection.area_px >= self.config.action_min_area_px
            and detection.height_px >= self.config.action_min_height_px
        )


class RoadSignCameraDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("gazebo_road_sign_camera_detector")
        self.declare_parameter("image_topic", "/track_preview_car/camera/image_raw")
        self.declare_parameter("publish_topic", "/lane_pid_experiment/camera_sign_detection")
        self.declare_parameter("traffic_light_detection_topic", "/lane_pid_experiment/camera_traffic_light_state")
        self.declare_parameter("traffic_light_suppression_enabled", True)
        self.declare_parameter("roi_top_ratio", 0.08)
        self.declare_parameter("roi_bottom_ratio", 0.84)
        self.declare_parameter("min_colored_area_px", 55)
        self.declare_parameter("min_bbox_height_px", 14)
        self.declare_parameter("min_score", 0.30)
        self.declare_parameter("action_min_area_px", 1500)
        self.declare_parameter("action_min_height_px", 34)
        self.declare_parameter("stable_frames", 2)
        self.declare_parameter("maneuver_stable_frames", 4)
        self.declare_parameter("timeout_s", 0.45)
        self.declare_parameter("log_period_s", 1.0)

        self.bridge = CvBridge()
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.publish_topic = str(self.get_parameter("publish_topic").value)
        self.traffic_light_detection_topic = str(self.get_parameter("traffic_light_detection_topic").value)
        self.traffic_light_suppression_enabled = bool(self.get_parameter("traffic_light_suppression_enabled").value)
        self.timeout_s = max(0.05, float(self.get_parameter("timeout_s").value))
        self.log_period_s = max(0.1, float(self.get_parameter("log_period_s").value))

        self.detector = RoadSignCameraDetector(
            RoadSignCameraConfig(
                roi_top_ratio=float(self.get_parameter("roi_top_ratio").value),
                roi_bottom_ratio=float(self.get_parameter("roi_bottom_ratio").value),
                min_colored_area_px=int(self.get_parameter("min_colored_area_px").value),
                min_bbox_height_px=int(self.get_parameter("min_bbox_height_px").value),
                min_score=float(self.get_parameter("min_score").value),
                action_min_area_px=int(self.get_parameter("action_min_area_px").value),
                action_min_height_px=int(self.get_parameter("action_min_height_px").value),
                stable_frames=int(self.get_parameter("stable_frames").value),
                maneuver_stable_frames=int(self.get_parameter("maneuver_stable_frames").value),
            )
        )

        self.pub = self.create_publisher(String, self.publish_topic, 10)
        self.create_subscription(Image, self.image_topic, self.handle_image, 10)
        if self.traffic_light_suppression_enabled:
            self.create_subscription(String, self.traffic_light_detection_topic, self.handle_traffic_light_detection, 10)

        self.latest_detection: RoadSignCameraDetection | None = None
        self.last_detection_monotonic = 0.0
        self.stable_name: str | None = None
        self.stable_count = 0
        self.last_logged_name: str | None = None
        self.last_log_monotonic = 0.0
        self.latest_traffic_light_payload: dict[str, object] | None = None
        self.latest_traffic_light_monotonic = 0.0

        self.get_logger().info(
            f"Road sign camera detector listening on {self.image_topic}; "
            f"publishing {self.publish_topic}; templates={len(self.detector.templates)} "
            f"roi={self.detector.config.roi_top_ratio:.2f}-{self.detector.config.roi_bottom_ratio:.2f}; "
            f"score>={self.detector.config.min_score:.2f}; camera_only=True."
        )

    def handle_traffic_light_detection(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            self.latest_traffic_light_payload = payload
            self.latest_traffic_light_monotonic = time.monotonic()

    def current_traffic_light_suppress_bboxes(self) -> list[tuple[int, int, int, int]]:
        if not self.traffic_light_suppression_enabled or self.latest_traffic_light_payload is None:
            return []
        if time.monotonic() - self.latest_traffic_light_monotonic > 0.65:
            return []
        if not bool(self.latest_traffic_light_payload.get("visible")):
            return []
        center = self.latest_traffic_light_payload.get("center")
        size = self.latest_traffic_light_payload.get("bbox_px") or self.latest_traffic_light_payload.get("size_px")
        if isinstance(center, dict):
            cx = center.get("x")
            cy = center.get("y")
        elif isinstance(center, (list, tuple)) and len(center) >= 2:
            cx, cy = center[0], center[1]
        else:
            return []
        if not isinstance(size, (list, tuple)) or len(size) < 2:
            return []
        if not all(isinstance(value, (int, float)) for value in (cx, cy, size[0], size[1])):
            return []
        width = max(26.0, float(size[0]) * 2.8)
        height = max(26.0, float(size[1]) * 2.8)
        return [
            (
                int(round(float(cx) - width * 0.5)),
                int(round(float(cy) - height * 0.5)),
                int(round(float(cx) + width * 0.5)),
                int(round(float(cy) + height * 0.5)),
            )
        ]

    def handle_image(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Could not convert road sign camera image: {exc}")
            return

        now = time.monotonic()
        detection = self.detector.detect(frame, self.current_traffic_light_suppress_bboxes())
        if detection is not None:
            self.latest_detection = detection
            self.last_detection_monotonic = now
            if detection.name == self.stable_name:
                self.stable_count += 1
            else:
                self.stable_name = detection.name
                self.stable_count = 1
            if detection.name != self.last_logged_name or now - self.last_log_monotonic >= self.log_period_s:
                self.get_logger().info(
                    f"camera_sign={detection.name} kind={detection.kind} maneuver={detection.maneuver or '-'} "
                    f"score={detection.score:.2f} area={detection.area_px}px "
                    f"bbox={detection.width_px}x{detection.height_px}px "
                    f"center=({detection.center_x},{detection.center_y}) "
                    f"ambiguous={detection.ambiguous}"
                )
                self.last_logged_name = detection.name
                self.last_log_monotonic = now
        elif now - self.last_detection_monotonic > self.timeout_s:
            self.latest_detection = None
            self.stable_name = None
            self.stable_count = 0
            self.last_logged_name = None

        self.publish_detection(now)

    def publish_detection(self, now: float) -> None:
        detection = self.latest_detection
        visible = detection is not None and now - self.last_detection_monotonic <= self.timeout_s
        payload: dict[str, object] = {
            "stamp_ns": self.get_clock().now().nanoseconds,
            "visible": visible,
            "source": "camera",
            "ground_truth_used": False,
            "stable_count": self.stable_count,
            "last_applied": None,
        }
        if not visible or detection is None:
            payload.update(
                {
                    "stable": False,
                    "actionable": False,
                    "required_stable_frames": self.detector.config.stable_frames,
                }
            )
        else:
            required_stable_frames = self.detector.required_stable_frames(detection)
            stable = self.stable_count >= required_stable_frames
            action_candidate = self.detector.action_candidate(detection)
            payload.update(
                {
                    "name": detection.name,
                    "kind": detection.kind,
                    "maneuver": detection.maneuver,
                    "score": detection.score,
                    "area_px": detection.area_px,
                    "colored_area_px": detection.colored_area_px,
                    "bbox_px": [detection.width_px, detection.height_px],
                    "bbox": {
                        "left": detection.left_x,
                        "top": detection.top_y,
                        "right": detection.right_x,
                        "bottom": detection.bottom_y,
                    },
                    "center": [detection.center_x, detection.center_y],
                    "center_ratio": detection.center_ratio,
                    "required_stable_frames": required_stable_frames,
                    "stable": stable,
                    "action_candidate": action_candidate,
                    "actionable": stable and action_candidate,
                    "ambiguous": detection.ambiguous,
                    "blue_ratio": detection.blue_ratio,
                    "red_ratio": detection.red_ratio,
                    "white_ratio": detection.white_ratio,
                    "scores_by_kind": detection.scores_by_kind,
                }
            )

        msg = String()
        msg.data = json.dumps(payload, sort_keys=True)
        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoadSignCameraDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
