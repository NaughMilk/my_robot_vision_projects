from __future__ import annotations

import json
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


SIGN_COLORS = {
    "go_straight_sign": (35, 210, 235),
    "turn_left_sign": (235, 120, 55),
    "turn_right_sign": (245, 75, 55),
    "parking_sign": (235, 160, 45),
    "speed_limit_sign": (45, 70, 235),
}

TRAFFIC_LIGHT_COLORS = {
    "red": (35, 45, 245),
    "yellow": (25, 220, 245),
    "green": (70, 220, 75),
}


class GazeboCameraViewer(Node):
    def __init__(self) -> None:
        super().__init__("gazebo_camera_viewer")
        self.declare_parameter("image_topic", "/track_preview_car/camera/image_raw")
        self.declare_parameter("window_name", "Track Car Camera")
        self.declare_parameter("show_window", True)
        self.declare_parameter("display_width", 760)
        self.declare_parameter("display_height", 428)
        self.declare_parameter("window_x", 40)
        self.declare_parameter("window_y", 40)
        self.declare_parameter("sign_overlay_enabled", True)
        self.declare_parameter("sign_detection_topic", "/gazebo_track_car/camera_sign_detection")
        self.declare_parameter("traffic_light_overlay_enabled", True)
        self.declare_parameter("traffic_light_detection_topic", "/gazebo_track_car/camera_traffic_light_state")
        self.declare_parameter("overlay_topic", "/track_preview_car/camera/sign_overlay")
        self.declare_parameter("sign_overlay_timeout_s", 0.65)
        self.declare_parameter("traffic_light_overlay_timeout_s", 0.65)

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.window_name = self.get_parameter("window_name").get_parameter_value().string_value
        self.show_window = bool(self.get_parameter("show_window").get_parameter_value().bool_value)
        self.display_width = int(self.get_parameter("display_width").get_parameter_value().integer_value)
        self.display_height = int(self.get_parameter("display_height").get_parameter_value().integer_value)
        self.window_x = int(self.get_parameter("window_x").get_parameter_value().integer_value)
        self.window_y = int(self.get_parameter("window_y").get_parameter_value().integer_value)
        self.sign_overlay_enabled = bool(
            self.get_parameter("sign_overlay_enabled").get_parameter_value().bool_value
        )
        self.sign_detection_topic = (
            self.get_parameter("sign_detection_topic").get_parameter_value().string_value
        )
        self.traffic_light_overlay_enabled = bool(
            self.get_parameter("traffic_light_overlay_enabled").get_parameter_value().bool_value
        )
        self.traffic_light_detection_topic = (
            self.get_parameter("traffic_light_detection_topic").get_parameter_value().string_value
        )
        self.overlay_topic = self.get_parameter("overlay_topic").get_parameter_value().string_value
        self.sign_overlay_timeout_s = max(
            0.05,
            float(self.get_parameter("sign_overlay_timeout_s").get_parameter_value().double_value),
        )
        self.traffic_light_overlay_timeout_s = max(
            0.05,
            float(self.get_parameter("traffic_light_overlay_timeout_s").get_parameter_value().double_value),
        )

        self.bridge = CvBridge()
        self.frame_received = False
        self.latest_sign_payload: dict[str, object] | None = None
        self.latest_sign_monotonic = 0.0
        self.latest_traffic_light_payload: dict[str, object] | None = None
        self.latest_traffic_light_monotonic = 0.0
        self.overlay_pub = self.create_publisher(Image, self.overlay_topic, 10)

        if self.show_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
            cv2.moveWindow(self.window_name, self.window_x, self.window_y)
            self.show_waiting_frame()

        self.create_subscription(Image, self.image_topic, self.handle_image, 10)
        if self.sign_overlay_enabled:
            self.create_subscription(String, self.sign_detection_topic, self.handle_sign_detection, 10)
        if self.traffic_light_overlay_enabled:
            self.create_subscription(
                String,
                self.traffic_light_detection_topic,
                self.handle_traffic_light_detection,
                10,
            )
        self.create_timer(0.03, self.keep_window_responsive)
        self.get_logger().info(
            f"Camera viewer listening on {self.image_topic}; "
            f"sign_overlay={'on' if self.sign_overlay_enabled else 'off'} "
            f"topic={self.sign_detection_topic}; "
            f"traffic_light_overlay={'on' if self.traffic_light_overlay_enabled else 'off'} "
            f"topic={self.traffic_light_detection_topic}"
        )

    def show_waiting_frame(self) -> None:
        frame = np.full((self.display_height, self.display_width, 3), (30, 34, 40), dtype=np.uint8)
        cv2.putText(
            frame,
            "Waiting for Gazebo camera...",
            (24, self.display_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 235, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def handle_image(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Could not convert camera image: {exc}")
            return

        if self.sign_overlay_enabled or self.traffic_light_overlay_enabled:
            frame = self.draw_sign_overlay(frame)
            overlay_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            overlay_msg.header = msg.header
            self.overlay_pub.publish(overlay_msg)

        if self.show_window and (
            frame.shape[1] != self.display_width or frame.shape[0] != self.display_height
        ):
            frame = cv2.resize(
                frame,
                (self.display_width, self.display_height),
                interpolation=cv2.INTER_AREA,
            )

        if not self.frame_received:
            self.get_logger().info("Receiving Gazebo camera frames.")
            self.frame_received = True

        if self.show_window:
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)

    def handle_sign_detection(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Could not parse sign detection JSON: {exc}")
            return
        if not isinstance(payload, dict):
            return
        self.latest_sign_payload = payload
        self.latest_sign_monotonic = time.monotonic()

    def handle_traffic_light_detection(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Could not parse traffic light detection JSON: {exc}")
            return
        if not isinstance(payload, dict):
            return
        self.latest_traffic_light_payload = payload
        self.latest_traffic_light_monotonic = time.monotonic()

    def draw_sign_overlay(self, frame: np.ndarray) -> np.ndarray:
        output = frame.copy()
        sign_payload = self.current_sign_payload() if self.sign_overlay_enabled else None
        traffic_payload = (
            self.current_traffic_light_payload() if self.traffic_light_overlay_enabled else None
        )
        self.draw_status_panel(output, sign_payload, traffic_payload)

        if traffic_payload is not None and bool(traffic_payload.get("visible")):
            self.draw_traffic_light_detection(output, traffic_payload)

        if sign_payload is None or not bool(sign_payload.get("visible")):
            return output

        bbox = self.payload_bbox(sign_payload, output.shape[1], output.shape[0])
        if bbox is None:
            return output

        x0, y0, x1, y1 = bbox
        name = str(sign_payload.get("name") or sign_payload.get("kind") or "unknown_sign")
        kind = str(sign_payload.get("kind") or name)
        color = SIGN_COLORS.get(str(sign_payload.get("kind") or name), (75, 220, 85))
        stable = bool(sign_payload.get("stable"))
        thickness = 3 if stable else 2
        cv2.rectangle(output, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)

        score = sign_payload.get("score")
        score_text = f"{float(score):.2f}" if isinstance(score, (int, float)) else "--"
        stable_count = int(sign_payload.get("stable_count") or 0)
        required = int(sign_payload.get("required_stable_frames") or 0)
        label = f"{kind} {score_text} stable {stable_count}/{required}"
        label_y = y0 - 8 if y0 > 82 else y1 + 28
        self.draw_label(output, label, x0, label_y, color)
        return output

    def draw_traffic_light_detection(
        self,
        frame: np.ndarray,
        payload: dict[str, object],
    ) -> None:
        bbox = self.payload_bbox(payload, frame.shape[1], frame.shape[0])
        if bbox is None:
            return
        x0, y0, x1, y1 = bbox
        state = str(payload.get("state") or payload.get("control_state") or "unknown").lower()
        color = TRAFFIC_LIGHT_COLORS.get(state, (210, 218, 226))
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        score = payload.get("score")
        area = payload.get("area_px")
        score_text = f"{float(score):.1f}" if isinstance(score, (int, float)) else "--"
        area_text = f"{int(area)}px" if isinstance(area, (int, float)) else "--"
        label = f"traffic_light {state.upper()} score {score_text} area {area_text}"
        label_y = y0 - 8 if y0 > 82 else y1 + 28
        self.draw_label(frame, label, x0, label_y, color)

    def current_sign_payload(self) -> dict[str, object] | None:
        if self.latest_sign_payload is None:
            return None
        if time.monotonic() - self.latest_sign_monotonic > self.sign_overlay_timeout_s:
            return None
        return self.latest_sign_payload

    def current_traffic_light_payload(self) -> dict[str, object] | None:
        if self.latest_traffic_light_payload is None:
            return None
        if (
            time.monotonic() - self.latest_traffic_light_monotonic
            > self.traffic_light_overlay_timeout_s
        ):
            return None
        return self.latest_traffic_light_payload

    def payload_bbox(
        self,
        payload: dict[str, object],
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int] | None:
        center = payload.get("center")
        size = payload.get("bbox_px") or payload.get("size_px")
        if not (
            ((isinstance(center, (list, tuple)) and len(center) >= 2) or isinstance(center, dict))
            and isinstance(size, (list, tuple))
            and len(size) >= 2
            and all(isinstance(value, (int, float)) for value in size[:2])
        ):
            return None

        if isinstance(center, dict):
            cx_raw = center.get("x")
            cy_raw = center.get("y")
            if not isinstance(cx_raw, (int, float)) or not isinstance(cy_raw, (int, float)):
                return None
            cx, cy = float(cx_raw), float(cy_raw)
        else:
            if not all(isinstance(value, (int, float)) for value in center[:2]):
                return None
            cx, cy = float(center[0]), float(center[1])
        width, height = max(1.0, float(size[0])), max(1.0, float(size[1]))
        x0 = int(round(cx - width * 0.5))
        y0 = int(round(cy - height * 0.5))
        x1 = int(round(cx + width * 0.5))
        y1 = int(round(cy + height * 0.5))
        return (
            max(0, min(frame_width - 1, x0)),
            max(0, min(frame_height - 1, y0)),
            max(0, min(frame_width - 1, x1)),
            max(0, min(frame_height - 1, y1)),
        )

    def draw_status_panel(
        self,
        frame: np.ndarray,
        sign_payload: dict[str, object] | None,
        traffic_payload: dict[str, object] | None,
    ) -> None:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 64), (18, 22, 28), -1)
        if sign_payload is None:
            text = "Sign detector: waiting / no current sign"
            color = (210, 218, 226)
        elif not bool(sign_payload.get("visible")):
            text = "Sign detector: none visible"
            color = (210, 218, 226)
        else:
            name = str(sign_payload.get("name") or sign_payload.get("kind") or "unknown_sign")
            kind = str(sign_payload.get("kind") or name)
            maneuver = str(sign_payload.get("maneuver") or "-")
            score = sign_payload.get("score")
            score_text = f"{float(score):.2f}" if isinstance(score, (int, float)) else "--"
            stable_count = int(sign_payload.get("stable_count") or 0)
            required = int(sign_payload.get("required_stable_frames") or 0)
            stable_text = "stable" if bool(sign_payload.get("stable")) else "warming"
            text = f"Sign: {kind} ({name}) cmd={maneuver} score={score_text} {stable_text}={stable_count}/{required}"
            color = SIGN_COLORS.get(str(sign_payload.get("kind") or name), (75, 220, 85))

        cv2.putText(
            frame,
            text,
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )

        if traffic_payload is None:
            traffic_text = "Traffic light: waiting / no current light"
            traffic_color = (210, 218, 226)
        elif not bool(traffic_payload.get("visible")):
            traffic_text = "Traffic light: none visible"
            traffic_color = (210, 218, 226)
        else:
            state = str(traffic_payload.get("state") or "-").lower()
            score = traffic_payload.get("score")
            area = traffic_payload.get("area_px")
            score_text = f"{float(score):.1f}" if isinstance(score, (int, float)) else "--"
            area_text = f"{int(area)}px" if isinstance(area, (int, float)) else "--"
            action = "stop" if bool(traffic_payload.get("stop")) else "slow" if bool(traffic_payload.get("slow")) else "-"
            traffic_text = f"Traffic light: {state.upper()} score={score_text} area={area_text} action={action}"
            traffic_color = TRAFFIC_LIGHT_COLORS.get(state, (210, 218, 226))

        cv2.putText(
            frame,
            traffic_text,
            (12, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            traffic_color,
            2,
            cv2.LINE_AA,
        )

    def draw_label(self, frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        x = max(0, min(frame.shape[1] - text_w - 10, x))
        y = max(text_h + baseline + 4, min(frame.shape[0] - 2, y))
        y0 = max(0, y - text_h - baseline - 4)
        x1 = min(frame.shape[1] - 1, x + text_w + 10)
        y1 = min(frame.shape[0] - 1, y0 + text_h + baseline + 8)
        cv2.rectangle(frame, (x, y0), (x1, y1), color, -1)
        cv2.putText(
            frame,
            text,
            (x + 5, y0 + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def keep_window_responsive(self) -> None:
        if self.show_window:
            cv2.waitKey(1)


def main() -> None:
    rclpy.init()
    node = GazeboCameraViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            pass
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        try:
            rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
