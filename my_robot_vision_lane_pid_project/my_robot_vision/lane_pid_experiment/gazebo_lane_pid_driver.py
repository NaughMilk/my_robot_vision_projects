from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from my_robot_vision.gazebo_track_car import (
    DirectedTrack,
    ROUTE_GRAPH_CONFIG_ENV,
    normalize_angle,
    quaternion_to_rpy,
)
from my_robot_vision.lane_pid_experiment.traffic_light_camera_detector import (
    TrafficLightCameraConfig,
    TrafficLightCameraDetection,
    TrafficLightCameraDetector,
)


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


@dataclass
class LaneSegment:
    x1: int
    y1: int
    x2: int
    y2: int
    length: float

    def x_at_y(self, target_y: float) -> float:
        dy = self.y2 - self.y1
        if abs(dy) < 1e-6:
            return 0.5 * (self.x1 + self.x2)
        ratio = (target_y - self.y1) / dy
        return self.x1 + ratio * (self.x2 - self.x1)


@dataclass
class LaneDetectionResult:
    visible: bool
    center_x: float
    target_x: float
    error_px: float
    error_norm: float
    confidence: float
    line_count: int
    left_count: int
    right_count: int
    source: str


@dataclass
class RouteGuardResult:
    enabled: bool
    available: bool
    distance_m: float
    angular: float
    weight: float
    segment: str
    reason: str


@dataclass
class SceneTrafficLightPose:
    name: str
    x: float
    y: float
    yaw: float
    from_model_state: bool = False


@dataclass
class TrafficLightProximityResult:
    actionable: bool
    visual_near: bool
    approach_near: bool
    scene_near: bool
    scene_pose_live: bool
    stable_count: int
    required_stable_frames: int
    distance_m: float | None
    forward_m: float | None
    lateral_m: float | None
    bottom_ratio: float
    area_px: int
    height_px: int
    reason: str


class PID:
    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.previous_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None

    def update_gains(self, kp: float, ki: float, kd: float, integral_limit: float) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)

    def update(self, error: float, dt: float) -> tuple[float, float, float, float]:
        dt = max(0.0, dt)
        self.integral += error * dt
        self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)
        derivative = 0.0
        if self.previous_error is not None and dt > 1e-6:
            derivative = (error - self.previous_error) / dt
        self.previous_error = error

        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative
        return p_term + i_term + d_term, p_term, i_term, d_term


class GazeboLanePidDriver(Node):
    def __init__(self) -> None:
        super().__init__("gazebo_lane_pid_driver")
        self.declare_parameter("camera_topic", "/track_preview_car/camera/image_raw")
        self.declare_parameter("cmd_topic", "/track_preview_car/cmd_vel")
        self.declare_parameter("debug_overlay_topic", "/lane_pid_experiment/debug_overlay")
        self.declare_parameter("ipm_topic", "/lane_pid_experiment/ipm")
        self.declare_parameter("mask_topic", "/lane_pid_experiment/mask")
        self.declare_parameter("lane_debug_topic", "/lane_pid_experiment/lane_debug")
        self.declare_parameter("traffic_light_topic", "/lane_pid_experiment/camera_traffic_light_state")
        self.declare_parameter("route_config", "")
        self.declare_parameter("route_guard_enabled", True)
        self.declare_parameter("route_guard_reverse_direction", True)
        self.declare_parameter("route_guard_clearance_px", 12)
        self.declare_parameter("route_guard_lookahead_m", 0.34)
        self.declare_parameter("route_guard_blend_start_m", 0.06)
        self.declare_parameter("route_guard_blend_full_m", 0.28)
        self.declare_parameter("route_guard_slow_distance_m", 0.38)
        self.declare_parameter("route_guard_stop_distance_m", 0.95)
        self.declare_parameter("route_guard_heading_gain", 2.6)
        self.declare_parameter("route_guard_lateral_gain", 2.0)
        self.declare_parameter("route_guard_max_angular", 1.5)
        self.declare_parameter("route_guard_low_confidence_weight", 0.40)
        self.declare_parameter("route_guard_single_lane_weight", 0.26)
        self.declare_parameter("publish_debug_images", True)
        self.declare_parameter("bev_width", 420)
        self.declare_parameter("bev_height", 420)
        self.declare_parameter("src_bottom_left_ratio", [0.04, 0.99])
        self.declare_parameter("src_bottom_right_ratio", [0.96, 0.99])
        self.declare_parameter("src_top_right_ratio", [0.73, 0.44])
        self.declare_parameter("src_top_left_ratio", [0.27, 0.44])
        self.declare_parameter("dst_margin_ratio", 0.12)
        self.declare_parameter("roi_top_ratio", 0.34)
        self.declare_parameter("lookahead_y_ratio", 0.76)
        self.declare_parameter("target_ratio", 0.50)
        self.declare_parameter("lane_width_px", 190.0)
        self.declare_parameter("min_lane_pixels", 120)
        self.declare_parameter("histogram_band_count", 6)
        self.declare_parameter("histogram_top_ratio", 0.50)
        self.declare_parameter("histogram_min_pixels_per_band", 18)
        self.declare_parameter("histogram_smooth_kernel_px", 31)
        self.declare_parameter("center_smoothing_alpha", 0.28)
        self.declare_parameter("max_center_jump_px", 28.0)
        self.declare_parameter("hough_pair_blend_weight", 0.08)
        self.declare_parameter("hough_single_blend_weight", 0.0)
        self.declare_parameter("hough_fallback_pair_weight", 0.10)
        self.declare_parameter("hough_fallback_single_weight", 0.05)
        self.declare_parameter("filled_no_touch_mask_enabled", True)
        self.declare_parameter("filled_no_touch_min_area_px", 260)
        self.declare_parameter("filled_no_touch_close_kernel_px", 31)
        self.declare_parameter("filled_no_touch_dilate_kernel_px", 13)
        self.declare_parameter("max_debug_hough_lines", 0)
        self.declare_parameter("hough_threshold", 32)
        self.declare_parameter("hough_min_line_length", 32)
        self.declare_parameter("hough_max_line_gap", 26)
        self.declare_parameter("hough_min_vertical_ratio", 0.48)
        self.declare_parameter("kp", 0.005)
        self.declare_parameter("ki", 0.0)
        self.declare_parameter("kd", 0.0007)
        self.declare_parameter("integral_limit", 100.0)
        self.declare_parameter("max_angular", 1.5)
        self.declare_parameter("base_speed", 0.22)
        self.declare_parameter("min_speed", 0.08)
        self.declare_parameter("lost_speed", 0.06)
        self.declare_parameter("speed_error_slowdown", 0.65)
        self.declare_parameter("lane_error_deadband_px", 3.5)
        self.declare_parameter("angular_smoothing_alpha", 0.36)
        self.declare_parameter("max_angular_delta_per_s", 2.8)
        self.declare_parameter("curve_slowdown_min_factor", 0.62)
        self.declare_parameter("curve_slowdown_angular_start", 0.45)
        self.declare_parameter("steering_sign", -1.0)
        self.declare_parameter("max_lost_frames", 12)
        self.declare_parameter("traffic_light_enabled", True)
        self.declare_parameter("traffic_light_slow_speed", 0.08)
        self.declare_parameter("traffic_light_timeout_s", 0.35)
        self.declare_parameter("traffic_light_stop_min_area_px", 900)
        self.declare_parameter("traffic_light_slow_min_area_px", 650)
        self.declare_parameter("traffic_light_release_min_area_px", 450)
        self.declare_parameter("traffic_light_action_min_height_px", 22)
        self.declare_parameter("traffic_light_action_min_bottom_y_ratio", 0.28)
        self.declare_parameter("traffic_light_action_min_score", 120.0)
        self.declare_parameter("traffic_light_action_stable_frames", 2)
        self.declare_parameter("traffic_light_proximity_gate_enabled", True)
        self.declare_parameter("traffic_light_scene_gate_strict", False)
        self.declare_parameter("traffic_light_red_approach_slow_enabled", True)
        self.declare_parameter("traffic_light_release_on_non_actionable", True)
        self.declare_parameter("traffic_light_lost_release_after_min_stop", True)
        self.declare_parameter("traffic_light_action_max_distance_m", 1.55)
        self.declare_parameter("traffic_light_action_min_forward_m", -0.05)
        self.declare_parameter("traffic_light_action_fov_rad", 2.20)
        self.declare_parameter("traffic_light_min_stop_s", 2.0)
        self.declare_parameter("camera_stale_stop_enabled", True)
        self.declare_parameter("camera_stale_timeout_s", 0.75)
        self.declare_parameter("camera_stale_log_period_s", 1.0)
        self.declare_parameter("initial_pose_reset_enabled", True)
        self.declare_parameter("initial_pose_from_route", True)
        self.declare_parameter("initial_pose_reset_delay_s", 1.0)
        self.declare_parameter("entity_name", "track_preview_car")
        self.declare_parameter("initial_pose_x_m", 2.1796875)
        self.declare_parameter("initial_pose_y_m", 1.8046875)
        self.declare_parameter("initial_pose_z_m", 0.02)
        self.declare_parameter("initial_pose_yaw_rad", -math.pi * 0.5)
        self.declare_parameter("log_period_s", 1.0)

        self.bridge = CvBridge()
        self.camera_topic = str(self.get_parameter("camera_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.publish_debug_images = bool(self.get_parameter("publish_debug_images").value)
        self.bev_width = int(self.get_parameter("bev_width").value)
        self.bev_height = int(self.get_parameter("bev_height").value)
        self.last_frame_shape: tuple[int, int] | None = None
        self.h_matrix: np.ndarray | None = None

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.overlay_pub = self.create_publisher(Image, str(self.get_parameter("debug_overlay_topic").value), 10)
        self.ipm_pub = self.create_publisher(Image, str(self.get_parameter("ipm_topic").value), 10)
        self.mask_pub = self.create_publisher(Image, str(self.get_parameter("mask_topic").value), 10)
        self.lane_debug_pub = self.create_publisher(String, str(self.get_parameter("lane_debug_topic").value), 10)
        self.traffic_light_pub = self.create_publisher(String, str(self.get_parameter("traffic_light_topic").value), 10)
        self.image_sub = self.create_subscription(Image, self.camera_topic, self.handle_image, 10)
        self.model_states_sub = self.create_subscription(ModelStates, "/model_states", self.handle_model_states, 10)

        self.pid = PID(
            float(self.get_parameter("kp").value),
            float(self.get_parameter("ki").value),
            float(self.get_parameter("kd").value),
            float(self.get_parameter("integral_limit").value),
        )
        now_monotonic = time.monotonic()
        self.last_control_time = now_monotonic
        self.last_image_time = now_monotonic
        self.camera_frames_seen = 0
        self.camera_stale_active = False
        self.last_camera_stale_log_time = 0.0
        self.lost_frames = 0
        self.last_lane_result: LaneDetectionResult | None = None
        self.smoothed_center_x: float | None = None
        self.smoothed_angular = 0.0
        self.last_log_time = 0.0
        self.last_traffic_detection: TrafficLightCameraDetection | None = None
        self.last_traffic_detection_time = 0.0
        self.last_traffic_proximity = TrafficLightProximityResult(
            actionable=False,
            visual_near=False,
            approach_near=False,
            scene_near=False,
            scene_pose_live=False,
            stable_count=0,
            required_stable_frames=0,
            distance_m=None,
            forward_m=None,
            lateral_m=None,
            bottom_ratio=0.0,
            area_px=0,
            height_px=0,
            reason="no_detection",
        )
        self.traffic_action_candidate_state: str | None = None
        self.traffic_action_candidate_count = 0
        self.stop_latched = False
        self.stop_latched_since = 0.0
        self.last_traffic_control_state = "none"
        self.actual_pose: tuple[float, float, float] | None = None
        self.route_track: DirectedTrack | None = self.create_route_track()
        self.scene_traffic_lights = self.load_scene_traffic_lights()
        self.scene_traffic_light_live_log_names: set[str] = set()
        self.last_route_segment = "-"

        self.traffic_detector = TrafficLightCameraDetector(
            TrafficLightCameraConfig(
                min_area_px=18,
                min_score=35.0,
                min_fill_ratio=0.42,
            )
        )

        self.entity_name = str(self.get_parameter("entity_name").value)
        self.initial_pose_reset_enabled = bool(self.get_parameter("initial_pose_reset_enabled").value)
        self.reset_client = self.create_client(SetEntityState, "/set_entity_state")
        self.reset_sent = False
        self.reset_done = not self.initial_pose_reset_enabled
        self.reset_future = None
        self.reset_warned = False
        self.reset_start_time = time.monotonic()
        self.reset_settle_until = 0.0
        if self.initial_pose_reset_enabled:
            self.create_timer(0.1, self.try_initial_pose_reset)
        self.create_timer(0.1, self.watchdog_camera_stale)

        self.get_logger().info(
            f"Lane PID driver camera={self.camera_topic} cmd={self.cmd_topic}; "
            f"pipeline=IPM->HSV white/yellow mask->HoughLinesP->PID; "
            f"traffic_light=camera-only; initial_reset={self.initial_pose_reset_enabled}; "
            f"route_guard={'on' if self.route_track is not None else 'off'}"
        )

    def create_route_track(self) -> DirectedTrack | None:
        if not bool(self.get_parameter("route_guard_enabled").value):
            return None
        route_config = str(self.get_parameter("route_config").value).strip()
        if route_config:
            os.environ[ROUTE_GRAPH_CONFIG_ENV] = route_config
        try:
            track = DirectedTrack(
                reverse_direction=bool(self.get_parameter("route_guard_reverse_direction").value),
                branch_probability=0.0,
                route_seed=7,
                route_explore_enabled=False,
                clearance_px=int(self.get_parameter("route_guard_clearance_px").value),
                route_offset_x_m=0.0,
                route_offset_y_m=0.0,
            )
        except Exception as exc:
            self.get_logger().warn(f"Route guard disabled; could not load route config: {exc}")
            return None
        route_x, route_y, route_yaw = track.sample()
        self.get_logger().info(
            f"Route guard start segment={track.current_segment.name} "
            f"pose=({route_x:+.3f},{route_y:+.3f},{route_yaw:+.3f})"
        )
        return track

    def load_scene_traffic_lights(self) -> list[SceneTrafficLightPose]:
        route_config = str(self.get_parameter("route_config").value).strip()
        if not route_config:
            return []
        try:
            with open(route_config, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            self.get_logger().warn(f"Traffic-light proximity gate has no scene config: {exc}")
            return []

        lights: list[SceneTrafficLightPose] = []
        for item in data.get("traffic_lights", {}).get("scene_models", []):
            if not isinstance(item, dict):
                continue
            pose = item.get("default_pose")
            if not isinstance(pose, list) or len(pose) < 4:
                continue
            lights.append(
                SceneTrafficLightPose(
                    name=str(item.get("name") or "traffic_light"),
                    x=float(pose[0]),
                    y=float(pose[1]),
                    yaw=float(pose[3]),
                )
            )
        if lights:
            self.get_logger().info(
                f"Traffic-light proximity gate loaded {len(lights)} scene positions from route config."
            )
        return lights

    def parameter_pair(self, name: str, default: tuple[float, float]) -> tuple[float, float]:
        values = self.get_parameter(name).value
        if isinstance(values, (list, tuple)) and len(values) >= 2:
            return float(values[0]), float(values[1])
        return default

    def refresh_tunable_parameters(self) -> None:
        self.pid.update_gains(
            float(self.get_parameter("kp").value),
            float(self.get_parameter("ki").value),
            float(self.get_parameter("kd").value),
            float(self.get_parameter("integral_limit").value),
        )

    def make_entity_state(self, x: float, y: float, z: float, yaw: float) -> EntityState:
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        state = EntityState()
        state.name = self.entity_name
        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)
        state.pose.orientation.x = qx
        state.pose.orientation.y = qy
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw
        state.reference_frame = "world"
        return state

    def initial_pose_target(self) -> tuple[float, float, float]:
        if bool(self.get_parameter("initial_pose_from_route").value) and self.route_track is not None:
            return self.route_track.sample()
        return (
            float(self.get_parameter("initial_pose_x_m").value),
            float(self.get_parameter("initial_pose_y_m").value),
            float(self.get_parameter("initial_pose_yaw_rad").value),
        )

    def try_initial_pose_reset(self) -> None:
        if self.reset_done or self.reset_sent:
            return
        delay_s = float(self.get_parameter("initial_pose_reset_delay_s").value)
        if time.monotonic() - self.reset_start_time < delay_s:
            return
        if not self.reset_client.service_is_ready():
            self.reset_client.wait_for_service(timeout_sec=0.0)
            if not self.reset_warned:
                self.get_logger().warn("Waiting for /set_entity_state before initial lane experiment pose reset.")
                self.reset_warned = True
            return

        request = SetEntityState.Request()
        target_x, target_y, target_yaw = self.initial_pose_target()
        request.state = self.make_entity_state(
            target_x,
            target_y,
            float(self.get_parameter("initial_pose_z_m").value),
            target_yaw,
        )
        self.reset_future = self.reset_client.call_async(request)
        self.reset_future.add_done_callback(self.handle_reset_response)
        self.reset_sent = True
        self.reset_settle_until = time.monotonic() + 0.45
        self.publish_stop_command()
        self.get_logger().info(
            f"Initial lane experiment reset target=({target_x:+.3f},{target_y:+.3f},{target_yaw:+.3f})"
        )

    def handle_reset_response(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warn(f"Initial pose reset failed: {exc}")
            self.reset_done = True
            return
        if response is not None and not response.success:
            status = getattr(response, "status_message", "reset rejected")
            self.get_logger().warn(f"Initial pose reset rejected: {status}")
        else:
            self.get_logger().info("Initial lane experiment pose reset sent.")
        self.reset_done = True

    def publish_stop_command(self) -> None:
        if hasattr(self, "smoothed_angular"):
            self.smoothed_angular = 0.0
        self.cmd_pub.publish(Twist())

    def watchdog_camera_stale(self) -> None:
        if not bool(self.get_parameter("camera_stale_stop_enabled").value):
            return
        now = time.monotonic()
        timeout_s = max(0.05, float(self.get_parameter("camera_stale_timeout_s").value))
        if now - self.last_image_time <= timeout_s:
            return

        self.camera_stale_active = True
        self.pid.reset()
        self.smoothed_angular = 0.0
        self.cmd_pub.publish(Twist())

        log_period_s = max(0.1, float(self.get_parameter("camera_stale_log_period_s").value))
        if now - self.last_camera_stale_log_time >= log_period_s:
            frames_text = "no frames yet" if self.camera_frames_seen == 0 else f"last frame {now - self.last_image_time:.2f}s ago"
            self.get_logger().warn(f"Camera stale watchdog stopping car; {frames_text}.")
            self.last_camera_stale_log_time = now

    def handle_model_states(self, msg: ModelStates) -> None:
        name_to_index = {name: index for index, name in enumerate(msg.name)}
        index = name_to_index.get(self.entity_name)
        if index is not None:
            pose = msg.pose[index]
            _, _, yaw = quaternion_to_rpy(pose.orientation)
            self.actual_pose = (float(pose.position.x), float(pose.position.y), normalize_angle(yaw))

        self.update_scene_traffic_light_poses(msg, name_to_index)

    def update_scene_traffic_light_poses(self, msg: ModelStates, name_to_index: dict[str, int]) -> None:
        if not self.scene_traffic_lights:
            return
        for light in self.scene_traffic_lights:
            index = name_to_index.get(light.name)
            if index is None:
                short_name = light.name.split("::")[-1]
                index = name_to_index.get(short_name)
            if index is None:
                continue

            pose = msg.pose[index]
            _, _, yaw = quaternion_to_rpy(pose.orientation)
            light.x = float(pose.position.x)
            light.y = float(pose.position.y)
            light.yaw = normalize_angle(yaw)
            if not light.from_model_state:
                light.from_model_state = True
                if light.name not in self.scene_traffic_light_live_log_names:
                    self.get_logger().info(
                        f"Traffic-light proximity gate using live Gazebo pose for {light.name}: "
                        f"({light.x:+.2f},{light.y:+.2f},{light.yaw:+.2f})."
                    )
                    self.scene_traffic_light_live_log_names.add(light.name)

    def handle_image(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Could not convert Gazebo camera image: {exc}")
            return

        now = time.monotonic()
        if self.camera_stale_active:
            self.get_logger().info("Camera frames recovered; lane PID control resumed.")
        self.camera_stale_active = False
        self.last_image_time = now
        self.camera_frames_seen += 1

        self.refresh_tunable_parameters()
        dt = max(1e-3, now - self.last_control_time)
        self.last_control_time = now

        if self.initial_pose_reset_enabled and (not self.reset_done or now < self.reset_settle_until):
            self.publish_stop_command()

        traffic_detection = self.update_camera_traffic_light(frame, now)
        ipm, mask, overlay, lane_result = self.detect_lane(frame)
        route_guard = self.compute_route_guard(lane_result)
        twist, pid_payload = self.compute_command(lane_result, traffic_detection, route_guard, dt, now)
        if self.initial_pose_reset_enabled and (not self.reset_done or now < self.reset_settle_until):
            twist = Twist()
        self.cmd_pub.publish(twist)

        self.publish_traffic_light_state(msg, traffic_detection, now)
        self.publish_lane_debug(msg, ipm, mask, overlay, lane_result, route_guard, twist, pid_payload)
        self.log_status(lane_result, route_guard, twist, pid_payload, traffic_detection, now)

    def compute_route_guard(self, lane_result: LaneDetectionResult) -> RouteGuardResult:
        if self.route_track is None:
            return RouteGuardResult(False, False, 0.0, 0.0, 0.0, "-", "disabled")
        if self.actual_pose is None:
            return RouteGuardResult(True, False, 0.0, 0.0, 0.0, "-", "waiting_for_model_states")

        x, y, yaw = self.actual_pose
        try:
            projection = self.route_track.nearest_pose_projection(x, y, yaw, outer_only=True)
            self.route_track.switch_to_projection(projection, f"lane_pid_guard: {projection.segment.name}")
            lookahead_x, lookahead_y = self.route_track._xy_ahead_from(
                projection.segment,
                projection.distance_on_segment,
                float(self.get_parameter("route_guard_lookahead_m").value),
            )
        except Exception as exc:
            return RouteGuardResult(True, False, 0.0, 0.0, 0.0, "-", f"route_error:{exc}")

        dx = lookahead_x - x
        dy = lookahead_y - y
        target_yaw = yaw if math.hypot(dx, dy) < 1e-6 else math.atan2(dy, dx)
        heading_error = normalize_angle(target_yaw - yaw)
        lateral_m = -dx * math.sin(yaw) + dy * math.cos(yaw)
        route_angular = (
            float(self.get_parameter("route_guard_heading_gain").value) * heading_error
            + float(self.get_parameter("route_guard_lateral_gain").value) * lateral_m
        )
        route_angular = clamp(
            route_angular,
            -float(self.get_parameter("route_guard_max_angular").value),
            float(self.get_parameter("route_guard_max_angular").value),
        )

        blend_start = float(self.get_parameter("route_guard_blend_start_m").value)
        blend_full = max(blend_start + 1e-3, float(self.get_parameter("route_guard_blend_full_m").value))
        weight = clamp((projection.distance_m - blend_start) / (blend_full - blend_start), 0.0, 1.0)
        if lane_result.source in {"lost", "memory"}:
            weight = max(weight, 0.35)
        if "hist_single" in lane_result.source or "hough_left" in lane_result.source or "hough_right" in lane_result.source:
            weight = max(weight, float(self.get_parameter("route_guard_single_lane_weight").value))
        if lane_result.confidence < 0.30:
            weight = max(weight, float(self.get_parameter("route_guard_low_confidence_weight").value))
        self.last_route_segment = projection.segment.name
        return RouteGuardResult(
            enabled=True,
            available=True,
            distance_m=float(projection.distance_m),
            angular=float(route_angular),
            weight=float(weight),
            segment=projection.segment.name,
            reason="ok",
        )

    def update_camera_traffic_light(
        self,
        frame: np.ndarray,
        now: float,
    ) -> TrafficLightCameraDetection | None:
        if not bool(self.get_parameter("traffic_light_enabled").value):
            self.last_traffic_detection = None
            self.last_traffic_proximity = self.make_empty_traffic_proximity("disabled")
            return None

        detection = self.traffic_detector.detect(frame)
        if detection is not None:
            self.last_traffic_detection = detection
            self.last_traffic_detection_time = now
        elif now - self.last_traffic_detection_time > float(self.get_parameter("traffic_light_timeout_s").value):
            self.last_traffic_detection = None

        self.last_traffic_proximity = self.evaluate_traffic_light_proximity(self.last_traffic_detection)
        return self.last_traffic_detection

    def make_empty_traffic_proximity(self, reason: str) -> TrafficLightProximityResult:
        self.traffic_action_candidate_state = None
        self.traffic_action_candidate_count = 0
        return TrafficLightProximityResult(
            actionable=False,
            visual_near=False,
            approach_near=False,
            scene_near=False,
            scene_pose_live=self.has_live_scene_traffic_light_pose(),
            stable_count=0,
            required_stable_frames=int(self.get_parameter("traffic_light_action_stable_frames").value),
            distance_m=None,
            forward_m=None,
            lateral_m=None,
            bottom_ratio=0.0,
            area_px=0,
            height_px=0,
            reason=reason,
        )

    def traffic_light_action_min_area_px(self, state: str) -> int:
        if state == "red":
            return int(self.get_parameter("traffic_light_stop_min_area_px").value)
        if state == "yellow":
            return int(self.get_parameter("traffic_light_slow_min_area_px").value)
        if state == "green":
            return int(self.get_parameter("traffic_light_release_min_area_px").value)
        return int(self.get_parameter("traffic_light_stop_min_area_px").value)

    def has_live_scene_traffic_light_pose(self) -> bool:
        return any(light.from_model_state for light in self.scene_traffic_lights)

    def nearest_scene_traffic_light_context(self) -> tuple[float, float, float] | None:
        if self.actual_pose is None or not self.scene_traffic_lights:
            return None
        car_x, car_y, car_yaw = self.actual_pose
        forward_x = math.cos(car_yaw)
        forward_y = math.sin(car_yaw)
        left_x = -math.sin(car_yaw)
        left_y = math.cos(car_yaw)
        best: tuple[float, float, float] | None = None
        use_live_poses = self.has_live_scene_traffic_light_pose()
        fov_half = float(self.get_parameter("traffic_light_action_fov_rad").value) * 0.5
        min_forward = float(self.get_parameter("traffic_light_action_min_forward_m").value)
        max_distance = float(self.get_parameter("traffic_light_action_max_distance_m").value)
        for light in self.scene_traffic_lights:
            if use_live_poses and not light.from_model_state:
                continue
            dx = light.x - car_x
            dy = light.y - car_y
            distance_m = math.hypot(dx, dy)
            if distance_m <= 1e-6 or distance_m > max_distance:
                continue
            forward_m = dx * forward_x + dy * forward_y
            if forward_m < min_forward:
                continue
            lateral_m = dx * left_x + dy * left_y
            angle = math.atan2(lateral_m, max(1e-6, forward_m))
            if abs(angle) > fov_half:
                continue
            if best is None or distance_m < best[0]:
                best = (distance_m, forward_m, lateral_m)
        return best

    def evaluate_traffic_light_proximity(
        self,
        detection: TrafficLightCameraDetection | None,
    ) -> TrafficLightProximityResult:
        if detection is None:
            return self.make_empty_traffic_proximity("no_detection")

        frame_height = max(1, detection.frame_height)
        bottom_ratio = detection.bottom_y / float(frame_height)
        min_area = self.traffic_light_action_min_area_px(detection.state)
        min_height = int(self.get_parameter("traffic_light_action_min_height_px").value)
        min_bottom_ratio = float(self.get_parameter("traffic_light_action_min_bottom_y_ratio").value)
        min_score = float(self.get_parameter("traffic_light_action_min_score").value)

        scene_pose_live = self.has_live_scene_traffic_light_pose()
        scene_context = self.nearest_scene_traffic_light_context()
        scene_near = scene_context is not None
        distance_m = None
        forward_m = None
        lateral_m = None
        if scene_context is not None:
            distance_m, forward_m, lateral_m = scene_context

        visual_candidate = (
            detection.area_px >= min_area
            and detection.height_px >= min_height
            and detection.score >= min_score
        )
        visual_near = visual_candidate and (scene_near or bottom_ratio >= min_bottom_ratio)
        approach_near = (
            detection.state in {"red", "yellow"}
            and detection.area_px >= int(self.get_parameter("traffic_light_slow_min_area_px").value)
            and detection.height_px >= max(1, min_height)
            and detection.score >= min_score
            and (scene_near or (not scene_pose_live and bottom_ratio >= min_bottom_ratio))
        )

        gate_enabled = bool(self.get_parameter("traffic_light_proximity_gate_enabled").value)
        if not gate_enabled:
            raw_actionable = True
            reason = "gate_disabled"
        elif detection.state == "green" and not self.stop_latched:
            raw_actionable = False
            reason = "green_without_stop_latch"
        elif scene_pose_live:
            raw_actionable = bool(visual_candidate and scene_near)
            if not visual_candidate:
                reason = "visual_too_far"
            elif scene_near:
                reason = "near_scene"
            else:
                reason = "scene_too_far"
        elif not visual_near:
            raw_actionable = False
            reason = "visual_too_far"
        elif self.scene_traffic_lights and self.actual_pose is not None:
            strict_scene_gate = bool(self.get_parameter("traffic_light_scene_gate_strict").value)
            raw_actionable = visual_near and (scene_near or not strict_scene_gate)
            if scene_near:
                reason = "near"
            elif strict_scene_gate:
                reason = "scene_too_far"
            else:
                reason = "near_camera_scene_unmatched"
        else:
            raw_actionable = True
            reason = "near_visual_only"

        if raw_actionable:
            if self.traffic_action_candidate_state == detection.state:
                self.traffic_action_candidate_count += 1
            else:
                self.traffic_action_candidate_state = detection.state
                self.traffic_action_candidate_count = 1
        else:
            self.traffic_action_candidate_state = None
            self.traffic_action_candidate_count = 0

        required_frames = int(self.get_parameter("traffic_light_action_stable_frames").value)
        actionable = raw_actionable and self.traffic_action_candidate_count >= required_frames
        if raw_actionable and not actionable:
            reason = "warming"

        return TrafficLightProximityResult(
            actionable=actionable,
            visual_near=visual_near,
            approach_near=approach_near,
            scene_near=scene_near,
            scene_pose_live=scene_pose_live,
            stable_count=self.traffic_action_candidate_count,
            required_stable_frames=required_frames,
            distance_m=distance_m,
            forward_m=forward_m,
            lateral_m=lateral_m,
            bottom_ratio=bottom_ratio,
            area_px=detection.area_px,
            height_px=detection.height_px,
            reason=reason,
        )

    def publish_traffic_light_state(
        self,
        msg: Image,
        detection: TrafficLightCameraDetection | None,
        now: float,
    ) -> None:
        visible = detection is not None and now - self.last_traffic_detection_time <= float(
            self.get_parameter("traffic_light_timeout_s").value
        )
        payload: dict[str, object] = {
            "stamp_ns": self.get_clock().now().nanoseconds,
            "visible": visible,
            "state": detection.state if visible and detection is not None else None,
            "source": "camera",
            "ground_truth_used": False,
            "control_state": self.last_traffic_control_state,
            "stop": self.last_traffic_control_state == "stop",
            "slow": self.last_traffic_control_state == "slow",
            "stop_latched": self.stop_latched,
            "stop_latched_elapsed_s": (
                max(0.0, now - self.stop_latched_since) if self.stop_latched else 0.0
            ),
            "actionable": self.last_traffic_proximity.actionable,
            "camera_age_s": max(0.0, now - self.last_image_time),
            "camera_stale": self.camera_stale_active,
            "proximity": {
                "reason": self.last_traffic_proximity.reason,
                "visual_near": self.last_traffic_proximity.visual_near,
                "approach_near": self.last_traffic_proximity.approach_near,
                "scene_near": self.last_traffic_proximity.scene_near,
                "scene_pose_live": self.last_traffic_proximity.scene_pose_live,
                "stable_count": self.last_traffic_proximity.stable_count,
                "required_stable_frames": self.last_traffic_proximity.required_stable_frames,
                "distance_m": self.last_traffic_proximity.distance_m,
                "forward_m": self.last_traffic_proximity.forward_m,
                "lateral_m": self.last_traffic_proximity.lateral_m,
                "bottom_ratio": self.last_traffic_proximity.bottom_ratio,
                "area_px": self.last_traffic_proximity.area_px,
                "height_px": self.last_traffic_proximity.height_px,
            },
        }
        if visible and detection is not None:
            payload.update(
                {
                    "score": detection.score,
                    "area_px": detection.area_px,
                    "center": {"x": detection.center_x, "y": detection.center_y},
                    "bbox": {
                        "left": detection.left_x,
                        "top": detection.top_y,
                        "right": detection.right_x,
                        "bottom": detection.bottom_y,
                    },
                    "size_px": [detection.width_px, detection.height_px],
                    "fill_ratio": detection.fill_ratio,
                }
            )
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self.traffic_light_pub.publish(message)

    def detect_lane(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, LaneDetectionResult]:
        ipm = self.warp_birdseye(frame)
        hsv = cv2.cvtColor(ipm, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(
            hsv,
            np.array([15, 55, 70], dtype=np.uint8),
            np.array([42, 255, 255], dtype=np.uint8),
        )
        white = cv2.inRange(
            hsv,
            np.array([0, 0, 150], dtype=np.uint8),
            np.array([179, 92, 255], dtype=np.uint8),
        )
        roi_top = int(self.bev_height * float(self.get_parameter("roi_top_ratio").value))
        yellow[:roi_top, :] = 0
        white[:roi_top, :] = 0

        yellow_clean = self.clean_binary_mask(yellow)
        white_clean = self.clean_binary_mask(white)
        combined_mask = cv2.bitwise_or(yellow_clean, white_clean)
        center_mask = yellow_clean
        if int(np.count_nonzero(center_mask)) < int(self.get_parameter("min_lane_pixels").value):
            center_mask = combined_mask

        no_touch_mask = self.solidify_no_touch_mask(yellow_clean)
        segments = self.hough_lane_segments(combined_mask)
        result = self.estimate_lane_center(center_mask, segments)
        overlay = self.draw_debug_overlay(ipm, no_touch_mask, combined_mask, segments, result)
        return ipm, no_touch_mask, overlay, result

    def odd_kernel_size(self, value: int, minimum: int = 3) -> int:
        value = max(minimum, int(value))
        return value if value % 2 == 1 else value + 1

    def clean_binary_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        return cleaned

    def solidify_no_touch_mask(self, yellow_mask: np.ndarray) -> np.ndarray:
        if not bool(self.get_parameter("filled_no_touch_mask_enabled").value):
            return yellow_mask

        close_size = self.odd_kernel_size(int(self.get_parameter("filled_no_touch_close_kernel_px").value), 5)
        dilate_size = self.odd_kernel_size(int(self.get_parameter("filled_no_touch_dilate_kernel_px").value), 3)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        solid_seed = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, close_kernel)
        solid_seed = cv2.dilate(solid_seed, dilate_kernel, iterations=1)

        contours, _ = cv2.findContours(solid_seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(yellow_mask)
        min_area = float(self.get_parameter("filled_no_touch_min_area_px").value)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            cv2.drawContours(filled, [cv2.convexHull(contour)], -1, 255, thickness=cv2.FILLED)
        return cv2.bitwise_or(filled, solid_seed)

    def warp_birdseye(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        shape = (width, height)
        if self.h_matrix is None or self.last_frame_shape != shape:
            self.last_frame_shape = shape
            bl = self.parameter_pair("src_bottom_left_ratio", (0.07, 0.98))
            br = self.parameter_pair("src_bottom_right_ratio", (0.93, 0.98))
            tr = self.parameter_pair("src_top_right_ratio", (0.63, 0.52))
            tl = self.parameter_pair("src_top_left_ratio", (0.37, 0.52))
            src = np.float32(
                [
                    [bl[0] * width, bl[1] * height],
                    [br[0] * width, br[1] * height],
                    [tr[0] * width, tr[1] * height],
                    [tl[0] * width, tl[1] * height],
                ]
            )
            margin = float(self.get_parameter("dst_margin_ratio").value) * self.bev_width
            dst = np.float32(
                [
                    [margin, self.bev_height - 1],
                    [self.bev_width - margin, self.bev_height - 1],
                    [self.bev_width - margin, 0],
                    [margin, 0],
                ]
            )
            self.h_matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, self.h_matrix, (self.bev_width, self.bev_height), flags=cv2.INTER_LINEAR)

    def hough_lane_segments(self, mask: np.ndarray) -> list[LaneSegment]:
        lines = cv2.HoughLinesP(
            mask,
            rho=1,
            theta=np.pi / 180.0,
            threshold=int(self.get_parameter("hough_threshold").value),
            minLineLength=int(self.get_parameter("hough_min_line_length").value),
            maxLineGap=int(self.get_parameter("hough_max_line_gap").value),
        )
        if lines is None:
            return []

        min_vertical_ratio = float(self.get_parameter("hough_min_vertical_ratio").value)
        segments: list[LaneSegment] = []
        for raw_line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in raw_line]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = math.hypot(dx, dy)
            if length < 1.0:
                continue
            if abs(dy) / length < min_vertical_ratio:
                continue
            segments.append(LaneSegment(x1, y1, x2, y2, length))
        segments.sort(key=lambda segment: segment.length, reverse=True)
        return segments

    def estimate_lane_center(self, mask: np.ndarray, segments: list[LaneSegment]) -> LaneDetectionResult:
        target_x = self.bev_width * float(self.get_parameter("target_ratio").value)
        lane_width_px = float(self.get_parameter("lane_width_px").value)
        hough_center, hough_source, left_count, right_count = self.estimate_hough_center(
            segments,
            target_x,
            lane_width_px,
        )
        hist_center, hist_confidence, hist_source = self.histogram_lane_center(mask, target_x, lane_width_px)

        candidates: list[tuple[float, float, str]] = []
        if hist_center is not None:
            candidates.append((hist_center, max(0.05, hist_confidence), hist_source))
        if hough_center is not None:
            hough_weight = (
                float(self.get_parameter("hough_pair_blend_weight").value)
                if hough_source == "hough_pair"
                else float(self.get_parameter("hough_single_blend_weight").value)
            )
            if hist_center is None:
                hough_weight = (
                    float(self.get_parameter("hough_fallback_pair_weight").value)
                    if hough_source == "hough_pair"
                    else float(self.get_parameter("hough_fallback_single_weight").value)
                )
            if hough_weight > 0.0:
                candidates.append((hough_center, hough_weight, hough_source))

        visible = bool(candidates)
        source = "lost"
        if candidates:
            total_weight = sum(weight for _, weight, _ in candidates)
            if total_weight <= 1e-6:
                center_x = sum(value for value, _, _ in candidates) / len(candidates)
            else:
                center_x = sum(value * weight for value, weight, _ in candidates) / total_weight
            center_x = self.smooth_lane_center(center_x)
            source = "+".join(item_source for _, _, item_source in candidates)
        elif self.last_lane_result is not None and self.lost_frames < int(self.get_parameter("max_lost_frames").value):
            center_x = self.last_lane_result.center_x
            visible = True
            source = "memory"
        else:
            center_x = target_x
            self.smoothed_center_x = None

        lane_pixels = int(np.count_nonzero(mask[int(self.bev_height * float(self.get_parameter("roi_top_ratio").value)) :, :]))
        confidence = max(
            hist_confidence,
            clamp(lane_pixels / float(max(1, int(self.get_parameter("min_lane_pixels").value) * 8)), 0.0, 1.0),
        )
        error_px = center_x - target_x
        error_norm = clamp(error_px / max(1.0, self.bev_width * 0.5), -1.0, 1.0)
        result = LaneDetectionResult(
            visible=visible,
            center_x=center_x,
            target_x=target_x,
            error_px=error_px,
            error_norm=error_norm,
            confidence=confidence,
            line_count=len(segments),
            left_count=left_count,
            right_count=right_count,
            source=source,
        )
        if visible:
            self.lost_frames = 0 if source != "memory" else self.lost_frames + 1
            self.last_lane_result = result
        else:
            self.lost_frames += 1
            self.pid.reset()
        return result

    def estimate_hough_center(
        self,
        segments: list[LaneSegment],
        target_x: float,
        lane_width_px: float,
    ) -> tuple[float | None, str, int, int]:
        lookahead_y = self.bev_height * float(self.get_parameter("lookahead_y_ratio").value)
        left_values: list[tuple[float, float]] = []
        right_values: list[tuple[float, float]] = []
        for segment in segments:
            x_at_lookahead = segment.x_at_y(lookahead_y)
            if x_at_lookahead < target_x:
                left_values.append((x_at_lookahead, segment.length))
            else:
                right_values.append((x_at_lookahead, segment.length))

        if left_values and right_values:
            left_x = self.weighted_average(left_values)
            right_x = self.weighted_average(right_values)
            return 0.5 * (left_x + right_x), "hough_pair", len(left_values), len(right_values)
        if left_values:
            return self.weighted_average(left_values) + lane_width_px * 0.5, "hough_left", len(left_values), 0
        if right_values:
            return self.weighted_average(right_values) - lane_width_px * 0.5, "hough_right", 0, len(right_values)
        return None, "hough_none", 0, 0

    def histogram_lane_center(
        self,
        mask: np.ndarray,
        target_x: float,
        lane_width_px: float,
    ) -> tuple[float | None, float, str]:
        roi_top = int(self.bev_height * float(self.get_parameter("histogram_top_ratio").value))
        if int(np.count_nonzero(mask[roi_top:, :])) < int(self.get_parameter("min_lane_pixels").value):
            return None, 0.0, "hist_lost"

        band_count = max(1, int(self.get_parameter("histogram_band_count").value))
        min_pixels = int(self.get_parameter("histogram_min_pixels_per_band").value)
        smooth_size = self.odd_kernel_size(int(self.get_parameter("histogram_smooth_kernel_px").value), 3)
        tracking_x = self.smoothed_center_x
        if tracking_x is None and self.last_lane_result is not None:
            tracking_x = self.last_lane_result.center_x
        if tracking_x is None:
            tracking_x = target_x

        band_height = max(1, (self.bev_height - roi_top) // band_count)
        centers: list[tuple[float, float]] = []
        pair_count = 0
        single_count = 0
        for index in range(band_count):
            y2 = self.bev_height - index * band_height
            y1 = max(roi_top, y2 - band_height)
            if y2 <= y1:
                continue
            band = mask[y1:y2, :]
            if int(np.count_nonzero(band)) < min_pixels:
                continue
            histogram = np.sum(band, axis=0).astype(np.float32)
            smoothed = cv2.GaussianBlur(histogram.reshape(1, -1), (smooth_size, 1), 0).reshape(-1)
            split_x = int(clamp(tracking_x, 1.0, self.bev_width - 2.0))
            left_peak_x: int | None = None
            right_peak_x: int | None = None
            left_peak_value = 0.0
            right_peak_value = 0.0
            if split_x > 1:
                left_slice = smoothed[:split_x]
                if left_slice.size:
                    left_peak_x = int(np.argmax(left_slice))
                    left_peak_value = float(left_slice[left_peak_x])
            if split_x < self.bev_width - 1:
                right_slice = smoothed[split_x:]
                if right_slice.size:
                    right_offset = int(np.argmax(right_slice))
                    right_peak_x = split_x + right_offset
                    right_peak_value = float(right_slice[right_offset])

            min_peak_value = float(min_pixels * 255)
            band_center: float | None = None
            band_weight = 0.0
            if (
                left_peak_x is not None
                and right_peak_x is not None
                and left_peak_value >= min_peak_value
                and right_peak_value >= min_peak_value
            ):
                band_center = 0.5 * (float(left_peak_x) + float(right_peak_x))
                band_weight = left_peak_value + right_peak_value
                pair_count += 1
            elif left_peak_x is not None and left_peak_value >= min_peak_value:
                band_center = float(left_peak_x) + lane_width_px * 0.5
                band_weight = left_peak_value
                single_count += 1
            elif right_peak_x is not None and right_peak_value >= min_peak_value:
                band_center = float(right_peak_x) - lane_width_px * 0.5
                band_weight = right_peak_value
                single_count += 1

            if band_center is None:
                continue
            bottom_bias = 1.0 + 0.18 * (band_count - index)
            centers.append((band_center, band_weight * bottom_bias))

        if not centers:
            return None, 0.0, "hist_lost"
        center_x = self.weighted_average(centers)
        confidence = clamp(
            sum(weight for _, weight in centers) / float(band_count * max(1, min_pixels) * 255 * 6),
            0.0,
            1.0,
        )
        source = "hist_pair" if pair_count >= single_count and pair_count > 0 else "hist_single"
        return center_x, confidence, source

    def smooth_lane_center(self, center_x: float) -> float:
        if self.smoothed_center_x is None:
            self.smoothed_center_x = float(center_x)
            return self.smoothed_center_x
        max_jump = float(self.get_parameter("max_center_jump_px").value)
        limited_delta = clamp(center_x - self.smoothed_center_x, -max_jump, max_jump)
        limited_center = self.smoothed_center_x + limited_delta
        alpha = clamp(float(self.get_parameter("center_smoothing_alpha").value), 0.0, 1.0)
        self.smoothed_center_x = (1.0 - alpha) * self.smoothed_center_x + alpha * limited_center
        return self.smoothed_center_x

    def weighted_average(self, values: list[tuple[float, float]]) -> float:
        total_weight = sum(max(weight, 1.0) for _, weight in values)
        if total_weight <= 1e-6:
            return sum(value for value, _ in values) / max(1, len(values))
        return sum(value * max(weight, 1.0) for value, weight in values) / total_weight

    def compute_command(
        self,
        lane_result: LaneDetectionResult,
        traffic_detection: TrafficLightCameraDetection | None,
        route_guard: RouteGuardResult,
        dt: float,
        now: float,
    ) -> tuple[Twist, dict[str, float | str | bool]]:
        twist = Twist()
        max_lost_frames = int(self.get_parameter("max_lost_frames").value)
        lane_reliable = lane_result.visible and self.lost_frames <= max_lost_frames
        if lane_reliable:
            deadband_px = float(self.get_parameter("lane_error_deadband_px").value)
            control_error_px = 0.0 if abs(lane_result.error_px) < deadband_px else lane_result.error_px
            raw, p_term, i_term, d_term = self.pid.update(control_error_px, dt)
            angular = clamp(
                float(self.get_parameter("steering_sign").value) * raw,
                -float(self.get_parameter("max_angular").value),
                float(self.get_parameter("max_angular").value),
            )
            error_slowdown = clamp(
                1.0 - abs(lane_result.error_norm) * float(self.get_parameter("speed_error_slowdown").value),
                0.0,
                1.0,
            )
            linear = max(
                float(self.get_parameter("min_speed").value),
                float(self.get_parameter("base_speed").value) * error_slowdown,
            )
        else:
            self.pid.reset()
            p_term = i_term = d_term = raw = 0.0
            linear = float(self.get_parameter("lost_speed").value)
            angular = 0.0

        if route_guard.enabled and route_guard.available:
            angular = (1.0 - route_guard.weight) * angular + route_guard.weight * route_guard.angular
            curve_start = float(self.get_parameter("curve_slowdown_angular_start").value)
            route_guard_max = max(1e-3, float(self.get_parameter("route_guard_max_angular").value))
            if abs(route_guard.angular) > curve_start:
                curve_ratio = clamp((abs(route_guard.angular) - curve_start) / (route_guard_max - curve_start), 0.0, 1.0)
                min_factor = clamp(float(self.get_parameter("curve_slowdown_min_factor").value), 0.1, 1.0)
                linear *= 1.0 - curve_ratio * (1.0 - min_factor)
            if route_guard.distance_m >= float(self.get_parameter("route_guard_slow_distance_m").value):
                linear = min(linear, float(self.get_parameter("lost_speed").value))
            if route_guard.distance_m >= float(self.get_parameter("route_guard_stop_distance_m").value):
                linear = 0.0
                angular = route_guard.angular

        control_state = self.traffic_light_control_state(traffic_detection, now)
        self.last_traffic_control_state = control_state
        if control_state == "stop":
            linear = 0.0
            angular = 0.0
            self.smoothed_angular = 0.0
        elif control_state == "slow":
            linear = min(linear, float(self.get_parameter("traffic_light_slow_speed").value))

        angular = self.smooth_angular_command(angular, dt)
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        return twist, {
            "raw": float(raw),
            "p": float(p_term),
            "i": float(i_term),
            "d": float(d_term),
            "dt": float(dt),
            "traffic_control": control_state,
            "lane_reliable": bool(lane_reliable),
            "route_guard_weight": float(route_guard.weight),
            "route_guard_distance_m": float(route_guard.distance_m),
            "route_guard_angular": float(route_guard.angular),
            "traffic_actionable": bool(self.last_traffic_proximity.actionable),
            "traffic_proximity_reason": self.last_traffic_proximity.reason,
        }

    def smooth_angular_command(self, angular: float, dt: float) -> float:
        alpha = clamp(float(self.get_parameter("angular_smoothing_alpha").value), 0.0, 1.0)
        max_delta = max(0.0, float(self.get_parameter("max_angular_delta_per_s").value)) * max(0.0, dt)
        limited_target = self.smoothed_angular + clamp(angular - self.smoothed_angular, -max_delta, max_delta)
        self.smoothed_angular = (1.0 - alpha) * self.smoothed_angular + alpha * limited_target
        return self.smoothed_angular

    def traffic_light_stop_elapsed_s(self, now: float) -> float:
        if not self.stop_latched:
            return 0.0
        return max(0.0, now - self.stop_latched_since)

    def latch_traffic_light_stop(self, detection: TrafficLightCameraDetection, now: float) -> None:
        if self.stop_latched:
            return
        self.stop_latched = True
        self.stop_latched_since = now
        distance_text = (
            "-"
            if self.last_traffic_proximity.distance_m is None
            else f"{self.last_traffic_proximity.distance_m:.2f}m"
        )
        self.get_logger().warn(
            f"Camera-only red traffic light stop latched; area={detection.area_px}px "
            f"h={detection.height_px}px b={self.last_traffic_proximity.bottom_ratio:.2f} d={distance_text} "
            f"score={detection.score:.1f} proximity={self.last_traffic_proximity.reason} "
            f"min_stop={float(self.get_parameter('traffic_light_min_stop_s').value):.1f}s"
        )

    def release_traffic_light_stop(self, now: float, reason: str) -> None:
        if not self.stop_latched:
            return
        elapsed_s = self.traffic_light_stop_elapsed_s(now)
        self.stop_latched = False
        self.stop_latched_since = 0.0
        self.get_logger().info(f"Camera-only traffic light stop released; reason={reason} elapsed={elapsed_s:.1f}s.")

    def traffic_light_control_state(
        self,
        detection: TrafficLightCameraDetection | None,
        now: float,
    ) -> str:
        min_stop_s = float(self.get_parameter("traffic_light_min_stop_s").value)
        if detection is None:
            if (
                self.stop_latched
                and self.traffic_light_stop_elapsed_s(now) >= min_stop_s
                and bool(self.get_parameter("traffic_light_lost_release_after_min_stop").value)
            ):
                self.release_traffic_light_stop(now, "camera_lost_after_min_stop")
                return "none"
            return "stop" if self.stop_latched else "none"

        if not self.last_traffic_proximity.actionable:
            if self.stop_latched:
                can_release = (
                    self.traffic_light_stop_elapsed_s(now) >= min_stop_s
                    and self.last_traffic_proximity.reason != "warming"
                    and bool(self.get_parameter("traffic_light_release_on_non_actionable").value)
                )
                if can_release:
                    self.release_traffic_light_stop(
                        now,
                        f"not_actionable:{self.last_traffic_proximity.reason}",
                    )
                    return "none"
                return "stop"
            if (
                bool(self.get_parameter("traffic_light_red_approach_slow_enabled").value)
                and self.last_traffic_proximity.approach_near
                and detection.state in {"red", "yellow"}
            ):
                return "slow"
            return "none"

        if detection.state == "red":
            self.latch_traffic_light_stop(detection, now)
            return "stop"

        if detection.state == "green" and self.stop_latched:
            if self.traffic_light_stop_elapsed_s(now) >= min_stop_s:
                self.release_traffic_light_stop(now, "camera_green_after_min_stop")
                return "none"
            return "stop"

        if self.stop_latched:
            return "stop"

        if detection.state == "yellow":
            return "slow"
        return "none"

    def draw_debug_overlay(
        self,
        ipm: np.ndarray,
        no_touch_mask: np.ndarray,
        lane_edge_mask: np.ndarray,
        segments: list[LaneSegment],
        result: LaneDetectionResult,
    ) -> np.ndarray:
        overlay = ipm.copy()
        no_touch_layer = np.zeros_like(overlay)
        no_touch_layer[no_touch_mask > 0] = (0, 210, 255)
        overlay = cv2.addWeighted(overlay, 0.74, no_touch_layer, 0.42, 0.0)

        edge_layer = np.zeros_like(overlay)
        edge_layer[lane_edge_mask > 0] = (255, 255, 255)
        overlay = cv2.addWeighted(overlay, 0.92, edge_layer, 0.18, 0.0)

        max_debug_lines = int(self.get_parameter("max_debug_hough_lines").value)
        for segment in segments[:max_debug_lines]:
            cv2.line(overlay, (segment.x1, segment.y1), (segment.x2, segment.y2), (255, 180, 0), 1, cv2.LINE_AA)
        target_x = int(round(result.target_x))
        center_x = int(round(result.center_x))
        cv2.line(overlay, (target_x, 0), (target_x, self.bev_height - 1), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.line(overlay, (center_x, 0), (center_x, self.bev_height - 1), (0, 0, 255), 2, cv2.LINE_AA)
        short_source = (
            result.source.replace("hist_pair", "hist2")
            .replace("hist_single", "hist1")
            .replace("hough_pair", "hough2")
            .replace("hough_left", "houghL")
            .replace("hough_right", "houghR")
        )
        label = f"{short_source} e={result.error_px:+.1f}px c={result.confidence:.2f} n={result.line_count}"
        cv2.putText(overlay, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 245, 245), 1, cv2.LINE_AA)
        return overlay

    def publish_lane_debug(
        self,
        source_msg: Image,
        ipm: np.ndarray,
        mask: np.ndarray,
        overlay: np.ndarray,
        lane_result: LaneDetectionResult,
        route_guard: RouteGuardResult,
        twist: Twist,
        pid_payload: dict[str, float | str | bool],
    ) -> None:
        if self.publish_debug_images:
            try:
                ipm_msg = self.bridge.cv2_to_imgmsg(ipm, encoding="bgr8")
                ipm_msg.header = source_msg.header
                self.ipm_pub.publish(ipm_msg)

                mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
                mask_msg.header = source_msg.header
                self.mask_pub.publish(mask_msg)

                overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
                overlay_msg.header = source_msg.header
                self.overlay_pub.publish(overlay_msg)
            except Exception as exc:
                self.get_logger().warn(f"Could not publish lane debug images: {exc}")

        message = String()
        message.data = json.dumps(
            {
                "stamp_ns": self.get_clock().now().nanoseconds,
                "visible": lane_result.visible,
                "center_x": lane_result.center_x,
                "target_x": lane_result.target_x,
                "error_px": lane_result.error_px,
                "error_norm": lane_result.error_norm,
                "confidence": lane_result.confidence,
                "line_count": lane_result.line_count,
                "left_count": lane_result.left_count,
                "right_count": lane_result.right_count,
                "source": lane_result.source,
                "lost_frames": self.lost_frames,
                "cmd_linear_x": twist.linear.x,
                "cmd_angular_z": twist.angular.z,
                "camera_stale": self.camera_stale_active,
                "pid": pid_payload,
                "traffic_light": {
                    "control_state": self.last_traffic_control_state,
                    "stop_latched": self.stop_latched,
                    "stop_latched_elapsed_s": self.traffic_light_stop_elapsed_s(time.monotonic()),
                    "actionable": self.last_traffic_proximity.actionable,
                    "reason": self.last_traffic_proximity.reason,
                    "visual_near": self.last_traffic_proximity.visual_near,
                    "approach_near": self.last_traffic_proximity.approach_near,
                    "scene_near": self.last_traffic_proximity.scene_near,
                    "scene_pose_live": self.last_traffic_proximity.scene_pose_live,
                    "area_px": self.last_traffic_proximity.area_px,
                    "height_px": self.last_traffic_proximity.height_px,
                },
                "route_guard": {
                    "enabled": route_guard.enabled,
                    "available": route_guard.available,
                    "distance_m": route_guard.distance_m,
                    "angular": route_guard.angular,
                    "weight": route_guard.weight,
                    "segment": route_guard.segment,
                    "reason": route_guard.reason,
                },
            },
            sort_keys=True,
        )
        self.lane_debug_pub.publish(message)

    def log_status(
        self,
        lane_result: LaneDetectionResult,
        route_guard: RouteGuardResult,
        twist: Twist,
        pid_payload: dict[str, float | str | bool],
        traffic_detection: TrafficLightCameraDetection | None,
        now: float,
    ) -> None:
        if now - self.last_log_time < float(self.get_parameter("log_period_s").value):
            return
        if traffic_detection is None:
            traffic_text = "lost_latched" if self.stop_latched else "-"
        else:
            distance_text = (
                "-"
                if self.last_traffic_proximity.distance_m is None
                else f"{self.last_traffic_proximity.distance_m:.2f}m"
            )
            if self.last_traffic_proximity.actionable:
                near_text = "actionable"
            elif self.last_traffic_proximity.approach_near:
                near_text = f"approach:{self.last_traffic_proximity.reason}"
            else:
                near_text = f"ignore:{self.last_traffic_proximity.reason}"
            scene_text = "live" if self.last_traffic_proximity.scene_pose_live else "cfg"
            traffic_text = (
                f"{traffic_detection.state}:{traffic_detection.area_px}px "
                f"h={traffic_detection.height_px}px b={self.last_traffic_proximity.bottom_ratio:.2f} "
                f"{near_text} d={distance_text} scene={scene_text} latch={self.stop_latched}"
            )
        self.get_logger().info(
            f"lane source={lane_result.source} err={lane_result.error_px:+.1f}px "
            f"norm={lane_result.error_norm:+.2f} lines={lane_result.line_count} "
            f"cmd=({twist.linear.x:.2f},{twist.angular.z:+.2f}) "
            f"pid=({pid_payload['p']:+.3f},{pid_payload['i']:+.3f},{pid_payload['d']:+.3f}) "
            f"route=({route_guard.segment} d={route_guard.distance_m:.2f} "
            f"w={route_guard.weight:.2f} wz={route_guard.angular:+.2f}) "
            f"traffic={traffic_text} control={pid_payload['traffic_control']}"
        )
        self.last_log_time = now


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GazeboLanePidDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_stop_command()
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
