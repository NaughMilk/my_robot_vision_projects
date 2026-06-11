from __future__ import annotations

import time

import cv2
import numpy as np

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image


def as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class LineFollower(Node):
    def __init__(self) -> None:
        super().__init__("line_follower")

        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("show_debug", True)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("target_ratio", 0.50)
        self.declare_parameter("steering_sign", 1.0)

        self.camera_topic = str(self.get_parameter("camera_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.show_debug = as_bool(self.get_parameter("show_debug").value)
        self.publish_debug_image = as_bool(self.get_parameter("publish_debug_image").value)
        self.target_ratio = float(self.get_parameter("target_ratio").value)
        self.steering_sign = float(self.get_parameter("steering_sign").value)

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, self.camera_topic, self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.debug_pub = self.create_publisher(Image, "/line_follower/debug_image", 10)

        self.state = "SEARCH"
        self.locked = False

        self.prev_track_x: int | None = None
        self.prev_error = 0.0
        self.last_error = 0.0
        self.lost_count = 0
        self.max_lost_frames = 18

        self.kp = 0.42
        self.kd = 0.22
        self.max_angular = 0.50

        self.linear_fast = 0.22
        self.linear_mid = 0.19
        self.linear_slow = 0.15
        self.linear_memory = 0.12
        self.linear_search = 0.10
        self.search_angular = 0.22
        self.last_drive_log_time = 0.0
        self.last_drive_status: str | None = None
        self.last_linear_stopped: bool | None = None

        self.get_logger().info(
            f"Line follower started. camera={self.camera_topic}, cmd={self.cmd_topic}, "
            f"target_ratio={self.target_ratio:.2f}, steering_sign={self.steering_sign:+.1f}"
        )

    def get_track_center(self, roi: np.ndarray, expected_x: int) -> tuple[int | None, list[int]]:
        h, w = roi.shape
        scan_rows = [int(h * 0.38), int(h * 0.64), int(h * 0.88)]
        row_weights = [0.7, 1.0, 1.5]
        samples: list[tuple[float, float]] = []

        for y, weight in zip(scan_rows, row_weights):
            xs = np.where(roi[y, :] > 0)[0]
            if len(xs) < 8:
                continue

            gaps = np.where(np.diff(xs) > 1)[0] + 1
            runs = np.split(xs, gaps)
            candidates: list[tuple[float, int]] = []
            for run in runs:
                if len(run) < 8:
                    continue
                center = float((int(run[0]) + int(run[-1])) / 2.0)
                candidates.append((center, len(run)))

            if not candidates:
                continue

            # On this rectangular map the camera can see two yellow sections at once.
            # Prefer the run that is closest to the previous prediction, not the leftmost pixels.
            center, _ = min(
                candidates,
                key=lambda item: abs(item[0] - expected_x) - min(item[1], w * 0.30) * 0.02,
            )
            samples.append((center, weight))

        if len(samples) < 2:
            return None, scan_rows

        weighted_sum = sum(center * weight for center, weight in samples)
        total_weight = sum(weight for _, weight in samples)
        return int(weighted_sum / total_weight), scan_rows

    def make_twist(self, linear: float, angular: float) -> Twist:
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        return twist

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge failed: {exc}")
            return

        h, w, _ = frame.shape
        debug_frame = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([15, 70, 70], dtype=np.uint8)
        upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        roi_top = int(h * 0.72)
        roi_bottom = int(h * 0.95)
        roi = np.zeros_like(mask)
        roi[roi_top:roi_bottom, :] = mask[roi_top:roi_bottom, :]
        roi_band = roi[roi_top:roi_bottom, :]

        target_x = int(self.target_ratio * w)
        cv2.line(debug_frame, (target_x, 0), (target_x, h), (255, 0, 0), 2)
        cv2.line(debug_frame, (0, roi_top), (w, roi_top), (255, 255, 0), 2)
        cv2.line(debug_frame, (0, roi_bottom), (w, roi_bottom), (255, 255, 0), 2)

        expected_x = self.prev_track_x if self.prev_track_x is not None else target_x
        track_x, scan_rows = self.get_track_center(roi_band, expected_x)
        for y in scan_rows:
            yy = roi_top + y
            cv2.line(debug_frame, (0, yy), (w, yy), (255, 0, 255), 1)

        twist = Twist()
        drive_status = "UNKNOWN"
        drive_reason = "controller did not choose a command"

        if not self.locked:
            if track_x is None:
                twist = self.make_twist(self.linear_search, self.steering_sign * self.search_angular)
                drive_status = "SEARCH"
                drive_reason = "yellow line missing from ROI before lock; crawl while searching"
                self.draw_label(debug_frame, "SEARCH", (0, 0, 255))
            else:
                self.locked = True
                self.state = "FOLLOW"
                self.prev_track_x = track_x
                self.prev_error = 0.0
                self.last_error = 0.0
                self.lost_count = 0

                error = (track_x - target_x) / (w / 2)
                self.last_error = error
                angular = self.steering_sign * self.clipped(self.kp * error)
                twist = self.make_twist(self.linear_slow, angular)
                drive_status = "LOCK"
                drive_reason = "yellow line acquired"

                cv2.circle(debug_frame, (track_x, int((roi_top + roi_bottom) / 2)), 8, (0, 255, 0), -1)
                self.draw_label(
                    debug_frame,
                    f"LOCK err={error:+.2f} lin={twist.linear.x:.2f} ang={twist.angular.z:+.2f}",
                    (0, 255, 0),
                    scale=0.58,
                )
        else:
            if track_x is not None:
                self.lost_count = 0
                if self.prev_track_x is None:
                    smooth_x = track_x
                else:
                    smooth_x = int(0.58 * self.prev_track_x + 0.42 * track_x)
                self.prev_track_x = smooth_x

                error = (smooth_x - target_x) / (w / 2)
                d_error = error - self.prev_error
                self.prev_error = error
                self.last_error = error

                angular = self.steering_sign * self.clipped(self.kp * error + self.kd * d_error)

                if abs(error) < 0.045:
                    twist = self.make_twist(self.linear_fast, 0.0)
                    drive_status = "FOLLOW_FAST"
                    drive_reason = "track centered"
                elif abs(error) < 0.14:
                    twist = self.make_twist(self.linear_mid, angular)
                    drive_status = "FOLLOW_MID"
                    drive_reason = "small line error"
                else:
                    twist = self.make_twist(self.linear_slow, angular)
                    drive_status = "FOLLOW_SLOW"
                    drive_reason = "large line error"

                cv2.circle(debug_frame, (smooth_x, int((roi_top + roi_bottom) / 2)), 8, (0, 0, 255), -1)
                self.draw_label(
                    debug_frame,
                    f"FOLLOW err={error:+.2f} lin={twist.linear.x:.2f} ang={twist.angular.z:+.2f}",
                    (0, 255, 0),
                    scale=0.58,
                )
            else:
                self.lost_count += 1
                if self.lost_count <= self.max_lost_frames:
                    angular = self.steering_sign * float(np.clip(0.45 * self.last_error, -0.10, 0.10))
                    twist = self.make_twist(self.linear_memory, angular)
                    drive_status = "MEMORY"
                    drive_reason = "line briefly missing; reuse last error"
                    self.draw_label(
                        debug_frame,
                        f"FOLLOW MEMORY lost={self.lost_count}",
                        (0, 200, 255),
                        scale=0.58,
                    )
                else:
                    lost_count = self.lost_count
                    self.locked = False
                    self.state = "SEARCH"
                    self.prev_track_x = None
                    self.prev_error = 0.0
                    self.last_error = 0.0
                    twist = self.make_twist(self.linear_search, self.steering_sign * self.search_angular)
                    drive_status = "RESEARCH"
                    drive_reason = f"line missing for {lost_count} frames; crawl while searching"
                    self.draw_label(debug_frame, "RE-SEARCH", (0, 0, 255))

        self.log_drive_command(twist, drive_status, drive_reason, track_x, target_x)
        self.cmd_pub.publish(twist)
        self.publish_debug(debug_frame, roi)

    def clipped(self, angular: float) -> float:
        return float(np.clip(angular, -self.max_angular, self.max_angular))

    def log_drive_command(
        self,
        twist: Twist,
        status: str,
        reason: str,
        track_x: int | None,
        target_x: int,
    ) -> None:
        now = time.monotonic()
        linear_stopped = abs(float(twist.linear.x)) < 1e-6
        status_changed = status != self.last_drive_status
        stop_changed = linear_stopped != self.last_linear_stopped
        if not status_changed and not stop_changed and now - self.last_drive_log_time < 1.0:
            return

        track_text = "none" if track_x is None else str(track_x)
        message = (
            f"drive status={status} reason={reason}; locked={self.locked}; "
            f"track_x={track_text} target_x={target_x} lost_frames={self.lost_count}; "
            f"cmd linear.x={twist.linear.x:+.2f} angular.z={twist.angular.z:+.2f}"
        )
        if linear_stopped:
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)

        self.last_drive_log_time = now
        self.last_drive_status = status
        self.last_linear_stopped = linear_stopped

    def draw_label(
        self,
        frame: np.ndarray,
        label: str,
        color: tuple[int, int, int],
        scale: float = 0.8,
    ) -> None:
        cv2.putText(frame, label, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    def publish_debug(self, debug_frame: np.ndarray, roi: np.ndarray) -> None:
        mask_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((debug_frame, mask_bgr))

        if self.publish_debug_image:
            try:
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(combined, encoding="bgr8"))
            except Exception as exc:
                self.get_logger().warn(f"Could not publish debug image: {exc}")

        if self.show_debug:
            cv2.imshow("Line Follower Debug", combined)
            cv2.waitKey(1)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
