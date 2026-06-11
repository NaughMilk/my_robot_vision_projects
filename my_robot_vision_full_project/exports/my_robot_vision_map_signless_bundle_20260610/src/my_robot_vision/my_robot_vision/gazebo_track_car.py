from __future__ import annotations

import heapq
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import SetEntityState, SetLightProperties
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import ColorRGBA, String

from my_robot_vision.world import share_file


MODEL_SIZE_M = 6.0
MAP_TEXTURE_SIZE = 256.0
DEFAULT_CLEARANCE_PX = 12
PARKING_CLEARANCE_PX = 4
PARKING_TRIGGER_MAX_DISTANCE_M = 1.1
MIN_BRANCH_DISTANCE_M = 1.6
MAP_VISUAL_NAME = "autonomous_map_visual.png"
MAP_IMAGE_ENV = "MY_ROBOT_MAP_IMAGE"
ROUTE_GRAPH_CONFIG_NAME = "autonomous_route_graph.json"
ROUTE_GRAPH_CONFIG_ENV = "MY_ROBOT_ROUTE_GRAPH_CONFIG"
SAFE_ROUTE_VERSION = "semantic_parking_spawn_turn_guard_v31"
MAP_VISUAL_OFFSET_X_M = 0.0
MAP_VISUAL_OFFSET_Y_M = 0.30
DEFAULT_ROUTE_OFFSET_X_M = 0.0
DEFAULT_ROUTE_OFFSET_Y_M = 0.0
ROUTE_CENTERLINE_WEIGHT = 18.0
STRAIGHT_ROUTE_AXIS_TOLERANCE_PX = 1.0
STRAIGHT_ROUTE_SIMPLIFY_MAX_LATERAL_PX = 4.0
YAW_LOOKAHEAD_M = 0.08
MANEUVER_YAW_LOOKAHEAD_M = 0.30
PARKING_YAW_LOOKAHEAD_M = 0.28
MANEUVER_LOOKAHEAD_M = 1.00
MANEUVER_MIN_TURN_RAD = math.radians(35.0)
MANEUVER_TIE_EPS_RAD = 0.05
MANEUVER_MAX_TURN_RAD = math.radians(135.0)
GUIDED_UPDATE_PERIOD = 0.02
CURB_VISUAL_BUFFER_PX = 3
FOOTPRINT_FRONT_M = 0.155
FOOTPRINT_REAR_M = 0.135
FOOTPRINT_HALF_WIDTH_M = 0.105
FOOTPRINT_MARGIN_M = 0.01
SPEED_LIMIT_PASS_MARGIN_M = FOOTPRINT_REAR_M + FOOTPRINT_MARGIN_M
FOOTPRINT_DEBUG_SAMPLE_M = 0.01
ROUTE_REJOIN_COMMAND_ERROR_M = 0.18
ROUTE_REJOIN_MAX_ROUTE_DISTANCE_M = 0.55
ROUTE_START_SYNC_MAX_ROUTE_DISTANCE_M = 1.25
ROUTE_REJOIN_COOLDOWN_S = 0.20
ROUTE_PROGRESS_SYNC_COMMAND_ERROR_M = 0.06
ROUTE_PROGRESS_SYNC_MAX_ROUTE_DISTANCE_M = 0.20
ROUTE_PROGRESS_SYNC_MAX_YAW_ERROR_RAD = 1.20
ROUTE_PROGRESS_SYNC_MIN_INTERVAL_S = 0.25
ROUTE_PROGRESS_SYNC_LOG_PERIOD_S = 1.00
ROUTE_MASK_DEBUG_PATH = "/tmp/gazebo_route_mask_debug.png"
ROUTE_CENTERLINE_DEBUG_PATH = "/tmp/gazebo_route_centerline_debug.png"
ROUTE_FOOTPRINT_DEBUG_PATH = "/tmp/gazebo_route_footprint_debug.png"
WHEEL_CONTROL_LOOKAHEAD_M = 0.32
WHEEL_CONTROL_PARKING_LOOKAHEAD_M = 0.20
WHEEL_CONTROL_HEADING_GAIN = 3.8
WHEEL_CONTROL_MAX_ANGULAR_RPS = 2.8
WHEEL_CONTROL_TURN_SLOWDOWN_RAD = 1.0
WHEEL_CONTROL_MIN_SPEED_SCALE = 0.35
WHEEL_CONTROL_TERMINAL_SLOWDOWN_M = 0.28
WHEEL_CONTROL_ROTATE_HEADING_ERROR_RAD = 1.05
WHEEL_CONTROL_SHARP_TURN_HEADING_ERROR_RAD = 0.70
WHEEL_CONTROL_SHARP_TURN_SPEED_SCALE = 0.18
WHEEL_CONTROL_BRANCH_APPROACH_DISTANCE_M = 0.45
WHEEL_CONTROL_BRANCH_APPROACH_MIN_SCALE = 0.10
WHEEL_CONTROL_BRANCH_ENDPOINT_SWITCH_DISTANCE_M = 0.045
WHEEL_CONTROL_BRANCH_EXIT_DISTANCE_M = 0.35
WHEEL_CONTROL_BRANCH_ROTATE_HEADING_ERROR_RAD = 1.45
WHEEL_CONTROL_NEAREST_ROUTE_MAX_DISTANCE_M = 0.65
WHEEL_CONTROL_NEAREST_ROUTE_SWITCH_YAW_ERROR_RAD = 0.95
WHEEL_CONTROL_NEAREST_ROUTE_REJOIN_YAW_ERROR_RAD = 1.25
WHEEL_CONTROL_NEAREST_ROUTE_REJOIN_MARGIN_M = 0.06
INITIAL_POSE_RESET_POSITION_TOLERANCE_M = 0.12
INITIAL_POSE_RESET_YAW_TOLERANCE_RAD = 0.35
PARKING_MANEUVER = "parking"
ROUTE_MANEUVERS = {"left", "right", "straight", PARKING_MANEUVER}
PARKING_DYNAMIC_ROUTE_NAME = "parking_lower_right_bay_active"
PARKING_SIGN_TRIGGER_POINT_PX = (189.0, 172.0)
PARKING_ENTRY_MAX_DISTANCE_M = 0.45
PARKING_SIGN_FACE_LOCAL_YAW_OFFSET_RAD = -math.pi * 0.5
PARKING_SIGN_FACE_MIN_DOT = 0.20
PARKING_SIGN_FACE_MAX_DISTANCE_M = 1.80
PARKING_SIGN_MODEL_NAME_TOKENS = ("parking",)
PARKING_CAMERA_APPROACH_SEGMENTS = frozenset(
    {
        "outer_bottom_right_mid_to_bottom_mid_reverse",
        "outer_bottom_right_to_bottom_right_mid_reverse",
        "outer_bottom_right_corner_reverse",
        "outer_right_mid_to_lower_reverse",
        "outer_right_upper_to_mid_reverse",
    }
)
PARKING_CAMERA_ENTRY_SEGMENTS = frozenset(
    {
        "outer_right_mid_to_lower_reverse",
        "outer_right_upper_to_mid_reverse",
    }
)
PARKING_EARLY_ENTRY_ANCHORS_PX = (
    (223.0, 150.0),
    (210.0, 130.0),
    (184.0, 113.0),
)
PARKING_ORTHOGONAL_TAIL_PX = (
    (221.0, 145.0),
    (175.0, 145.0),
    (175.0, 116.0),
)
PARKING_BAY_GOAL_POINTS_PX = (
    (173.0, 130.0),
    (173.0, 128.0),
    (169.0, 128.0),
    (169.0, 126.0),
    (177.0, 128.0),
)
PARKING_INITIAL_HEADING_WEIGHT_M = 0.35
PARKING_FINAL_YAW_RAD = math.pi * 0.5
PARKING_FINAL_HEADING_WEIGHT_M = 0.30
PARKING_FINAL_STRAIGHT_EXTENSION_PX = 12.0
PARKING_SEMANTIC_ROUTE_ENABLED = True
PARKING_SEMANTIC_STAGING_POSE_PX = (221.0, 145.0, -math.pi * 0.5)
PARKING_SEMANTIC_ENTRY_POSE_PX = (175.0, 145.0, math.pi)
PARKING_SEMANTIC_GOAL_POSE_PX = (175.0, 116.0, math.pi * 0.5)
PARKING_SEMANTIC_FINAL_EXTENSION_PX = 8.0
CAMERA_MANEUVER_MAX_BRANCH_DISTANCE_M = 3.40
NEARBY_MANEUVER_ROUTE_DISTANCE_M = 0.35
TRAFFIC_LIGHT_VALID_STATES = {"green", "yellow", "red"}
TRAFFIC_LIGHT_CYCLE_ORDER = ("green", "yellow", "red")
TRAFFIC_LIGHT_DEFAULT_DURATIONS_S = {
    "green": 5.0,
    "yellow": 1.0,
    "red": 8.0,
}
TRAFFIC_LIGHT_GLOW_NAMES = {}
TRAFFIC_LIGHT_GLOW_ENTITIES = {}
TRAFFIC_LIGHT_GLOW_COLORS = {
    "red": (6.0, 0.02, 0.01, 1.0),
    "yellow": (5.5, 4.2, 0.03, 1.0),
    "green": (0.01, 6.0, 0.08, 1.0),
}
TRAFFIC_LIGHT_GLOW_OFF_COLOR = (0.0, 0.0, 0.0, 1.0)
TRAFFIC_LIGHT_GLOW_ATTENUATION = (0.05, 0.02, 0.002)
TRAFFIC_LIGHT_SCENE_MODEL_NAMES = (
    "traffic_light_top_island",
    "traffic_light_lower_island",
)
TRAFFIC_LIGHT_SCENE_DEFAULT_POSES = {
    "traffic_light_top_island": (0.203395, 1.58911, 0.0, 1.5869),
    "traffic_light_lower_island": (-0.270238, -1.23571, 0.0, -1.52354),
}
TRAFFIC_LIGHT_CAPTURE_MAX_DISTANCE_M = 3.2
TRAFFIC_LIGHT_ACQUIRE_MIN_FORWARD_M = 0.05
TRAFFIC_LIGHT_SCENE_RELEASE_FOV_RAD = 1.95
TRAFFIC_LIGHT_DEFAULT_STOP_DISTANCE_M = 0.85
TRAFFIC_LIGHT_DEFAULT_MIN_STOP_S = 2.0
TRAFFIC_LIGHT_INDICATOR_HIDDEN_POSE = (0.0, 0.0, -5.0)
TRAFFIC_LIGHT_INDICATOR_LOCAL_X_M = 0.0
TRAFFIC_LIGHT_INDICATOR_LOCAL_Y_M = -0.038
TRAFFIC_LIGHT_INDICATOR_MODELS = {
    "red": (
        ("traffic_light_top_island_red_active_lamp", "traffic_light_top_island", 0.66),
        ("traffic_light_lower_island_red_active_lamp", "traffic_light_lower_island", 0.66),
    ),
    "yellow": (
        ("traffic_light_top_island_yellow_active_lamp", "traffic_light_top_island", 0.565),
        ("traffic_light_lower_island_yellow_active_lamp", "traffic_light_lower_island", 0.565),
    ),
    "green": (
        ("traffic_light_top_island_green_active_lamp", "traffic_light_top_island", 0.47),
        ("traffic_light_lower_island_green_active_lamp", "traffic_light_lower_island", 0.47),
    ),
}
TRAFFIC_LIGHT_CAMERA_COLOR_RANGES = {
    "red": (((0, 80, 110), (10, 255, 255)), ((170, 80, 110), (179, 255, 255))),
    "yellow": (((16, 80, 115), (42, 255, 255)),),
    "green": (((40, 40, 60), (100, 255, 255)),),
}
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
        PARKING_MANEUVER,
        ("models", "road_sign_parking", "materials", "textures", "parking.png"),
    ),
    (
        "camera_sign_speed_limit",
        "speed_limit_sign",
        None,
        ("models", "road_sign_speed_limit", "materials", "textures", "limit_speed.jpg"),
    ),
)
CAMERA_SIGN_SCENE_IDENTIFIERS = (
    ("go_straight_sign", "straight", ("go_straight", "straight")),
    ("turn_left_sign", "left", ("turn_left",)),
    ("turn_right_sign", "right", ("turn_right",)),
    ("parking_sign", PARKING_MANEUVER, ("parking",)),
    ("speed_limit_sign", None, ("speed_limit", "limit_speed")),
)
CAMERA_SIGN_FEATURE_SIZE = (96, 96)
CAMERA_SIGN_CONTEXT_PREFERRED_MARGIN = 0.08
CAMERA_SIGN_MANEUVER_CENTER_MIN_RATIO = 0.20
CAMERA_SIGN_MANEUVER_CENTER_MAX_RATIO = 0.80
CAMERA_SIGN_MANEUVER_STABLE_FRAMES = 4
CAMERA_SIGN_MANEUVER_KINDS = frozenset({"go_straight_sign", "turn_left_sign", "turn_right_sign"})
CAMERA_SIGN_MANEUVER_KIND_BY_ACTION = {
    "straight": "go_straight_sign",
    "left": "turn_left_sign",
    "right": "turn_right_sign",
}
CAMERA_SIGN_TURN_AMBIGUITY_MARGIN = 0.04
CAMERA_SIGN_TURN_OPPOSITE_KINDS = {
    "turn_left_sign": "turn_right_sign",
    "turn_right_sign": "turn_left_sign",
}
CAMERA_SIGN_PARKING_CONFUSION_MIN_SCORE = 0.30
CAMERA_SIGN_PARKING_CONFUSION_MARGIN = 0.16
CAMERA_SIGN_PARKING_CONFUSION_BLOCK_KINDS = frozenset({"turn_left_sign", "turn_right_sign"})


@dataclass
class TrafficLightCameraDetection:
    state: str
    score: float
    area_px: int
    center_x: int
    center_y: int
    width_px: int
    height_px: int
    fill_ratio: float


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
class CameraInstructionDetection:
    name: str
    kind: str
    maneuver: str | None
    score: float
    area_px: int
    center_x: int
    center_y: int
    width_px: int
    height_px: int
    frame_width: int
    frame_height: int


@dataclass(frozen=True)
class RouteSegmentSpec:
    name: str
    start_node: str
    end_node: str
    anchors: tuple[tuple[float, float], ...]
    inner_route: bool = False
    parking_route: bool = False
    terminal_stop: bool = False
    manual_centerline: bool = False


@dataclass(frozen=True)
class ParkingRouteConfig:
    dynamic_route_name: str
    dynamic_start_node: str
    stop_node: str
    entry_label: str
    approach_segments: frozenset[str]
    entry_segments: frozenset[str]
    entry_max_distance_m: float
    sign_model_name_tokens: tuple[str, ...]
    sign_face_local_yaw_offset_rad: float
    sign_face_min_dot: float
    sign_face_max_distance_m: float
    early_entry_anchors_px: tuple[tuple[float, float], ...]
    orthogonal_tail_px: tuple[tuple[float, float], ...]
    bay_goal_points_px: tuple[tuple[float, float], ...]
    initial_heading_weight_m: float
    final_yaw_rad: float
    final_heading_weight_m: float
    final_straight_extension_px: float
    semantic_route_enabled: bool
    semantic_staging_pose_px: tuple[float, float, float]
    semantic_entry_pose_px: tuple[float, float, float]
    semantic_goal_pose_px: tuple[float, float, float]
    semantic_final_extension_px: float


@dataclass(frozen=True)
class TrafficLightConfig:
    scene_model_names: tuple[str, ...]
    default_poses: dict[str, tuple[float, float, float, float]]
    release_fov_rad: float
    indicator_hidden_pose: tuple[float, float, float]
    indicator_local_xy_m: tuple[float, float]
    indicator_models: dict[str, tuple[tuple[str, str, float], ...]]


@dataclass(frozen=True)
class RouteBehaviorConfig:
    maneuver_max_branch_distance_m: float
    nearby_maneuver_route_distance_m: float


@dataclass(frozen=True)
class RouteZone:
    name: str
    kind: str
    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float

    def contains(self, x: float, y: float) -> bool:
        return self.min_x_m <= x <= self.max_x_m and self.min_y_m <= y <= self.max_y_m


@dataclass(frozen=True)
class RouteGraphConfig:
    start_segment_names: dict[str, str]
    segments: tuple[RouteSegmentSpec, ...]
    additional_directed_segments: dict[str, tuple[RouteSegmentSpec, ...]]
    behavior: RouteBehaviorConfig
    parking: ParkingRouteConfig
    traffic_lights: TrafficLightConfig
    zones: tuple[RouteZone, ...]


def _point_from_config(value: object, context: str) -> tuple[float, float]:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        return float(value[0]), float(value[1])
    raise ValueError(f"{context} must be a [x, y] point.")


def _points_from_config(value: object, context: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be a list of [x, y] points.")
    return tuple(_point_from_config(point, f"{context}[{index}]") for index, point in enumerate(value))


def _pose3_from_config(value: object, context: str) -> tuple[float, float, float]:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(item, (int, float)) for item in value)
    ):
        return float(value[0]), float(value[1]), float(value[2])
    raise ValueError(f"{context} must be a [x, y, z] pose.")


def _pose4_from_config(value: object, context: str) -> tuple[float, float, float, float]:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
    ):
        return float(value[0]), float(value[1]), float(value[2]), float(value[3])
    raise ValueError(f"{context} must be a [x, y, z, yaw] pose.")


def _indicator_models_from_config(
    value: object,
    context: str,
) -> dict[str, tuple[tuple[str, str, float], ...]]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object.")
    parsed: dict[str, tuple[tuple[str, str, float], ...]] = {}
    for state, entries in value.items():
        if not isinstance(entries, (list, tuple)):
            raise ValueError(f"{context}.{state} must be a list.")
        parsed_entries: list[tuple[str, str, float]] = []
        for index, entry in enumerate(entries):
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 3
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], str)
                or not isinstance(entry[2], (int, float))
            ):
                raise ValueError(f"{context}.{state}[{index}] must be [model_name, parent_model_name, local_z].")
            parsed_entries.append((entry[0], entry[1], float(entry[2])))
        parsed[str(state)] = tuple(parsed_entries)
    return parsed


def _resolve_route_anchor(
    value: object,
    nodes: dict[str, tuple[float, float]],
    context: str,
) -> tuple[float, float]:
    if isinstance(value, str):
        if value not in nodes:
            raise ValueError(f"{context} references unknown route node '{value}'.")
        return nodes[value]
    return _point_from_config(value, context)


def _route_segment_spec_from_config(
    value: object,
    nodes: dict[str, tuple[float, float]],
    context: str,
) -> RouteSegmentSpec:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object.")
    anchors_value = value.get("anchors", ())
    if not isinstance(anchors_value, (list, tuple)):
        raise ValueError(f"{context}.anchors must be a list.")
    anchors = tuple(
        _resolve_route_anchor(anchor, nodes, f"{context}.anchors[{index}]")
        for index, anchor in enumerate(anchors_value)
    )
    if len(anchors) < 2:
        raise ValueError(f"{context}.anchors must contain at least two points.")
    start_node = str(value["start"])
    end_node = str(value["end"])
    if start_node not in nodes:
        raise ValueError(f"{context}.start references unknown route node '{start_node}'.")
    if end_node not in nodes:
        raise ValueError(f"{context}.end references unknown route node '{end_node}'.")
    return RouteSegmentSpec(
        name=str(value["name"]),
        start_node=start_node,
        end_node=end_node,
        anchors=anchors,
        inner_route=bool(value.get("inner_route", False)),
        parking_route=bool(value.get("parking_route", False)),
        terminal_stop=bool(value.get("terminal_stop", False)),
        manual_centerline=bool(value.get("manual_centerline", False)),
    )


def _route_segment_specs_from_config(
    value: object,
    nodes: dict[str, tuple[float, float]],
    context: str,
) -> tuple[RouteSegmentSpec, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be a list.")
    return tuple(
        _route_segment_spec_from_config(item, nodes, f"{context}[{index}]")
        for index, item in enumerate(value)
    )


def load_route_graph_config() -> RouteGraphConfig:
    override_path = os.environ.get(ROUTE_GRAPH_CONFIG_ENV, "").strip()
    if override_path:
        path = Path(override_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise RuntimeError(f"Route graph config override does not exist: {path}")
    else:
        path = share_file("config", ROUTE_GRAPH_CONFIG_NAME)
    if path is None:
        raise RuntimeError(f"Route graph config not found: config/{ROUTE_GRAPH_CONFIG_NAME}")

    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")

    route_data = data.get("route", {})
    if not isinstance(route_data, dict):
        raise ValueError("route must be an object.")
    raw_nodes = route_data.get("nodes", {})
    if not isinstance(raw_nodes, dict):
        raise ValueError("route.nodes must be an object.")
    nodes = {
        str(name): _point_from_config(point, f"route.nodes.{name}")
        for name, point in raw_nodes.items()
    }
    raw_start_segments = route_data.get("start_segments", {})
    if not isinstance(raw_start_segments, dict):
        raise ValueError("route.start_segments must be an object.")
    start_segment_names = {str(direction): str(name) for direction, name in raw_start_segments.items()}
    segments = _route_segment_specs_from_config(route_data.get("segments", ()), nodes, "route.segments")
    segment_names = {spec.name for spec in segments}
    for direction, segment_name in start_segment_names.items():
        if segment_name not in segment_names:
            raise ValueError(f"route.start_segments.{direction} references unknown segment '{segment_name}'.")

    additional_directed_segments: dict[str, tuple[RouteSegmentSpec, ...]] = {}
    raw_additional = route_data.get("additional_directed_segments", {})
    if not isinstance(raw_additional, dict):
        raise ValueError("route.additional_directed_segments must be an object.")
    for direction, direction_segments in raw_additional.items():
        additional_directed_segments[str(direction)] = _route_segment_specs_from_config(
            direction_segments,
            nodes,
            f"route.additional_directed_segments.{direction}",
        )

    behavior_data = data.get("behavior", {})
    if not isinstance(behavior_data, dict):
        raise ValueError("behavior must be an object.")
    behavior = RouteBehaviorConfig(
        maneuver_max_branch_distance_m=float(
            behavior_data.get("maneuver_max_branch_distance_m", CAMERA_MANEUVER_MAX_BRANCH_DISTANCE_M)
        ),
        nearby_maneuver_route_distance_m=float(
            behavior_data.get("nearby_maneuver_route_distance_m", NEARBY_MANEUVER_ROUTE_DISTANCE_M)
        ),
    )

    parking_data = data.get("parking", {})
    if not isinstance(parking_data, dict):
        raise ValueError("parking must be an object.")
    parking = ParkingRouteConfig(
        dynamic_route_name=str(parking_data.get("dynamic_route_name", PARKING_DYNAMIC_ROUTE_NAME)),
        dynamic_start_node=str(parking_data.get("dynamic_start_node", "parking_dynamic_start")),
        stop_node=str(parking_data.get("stop_node", "parking_lower_right_stop")),
        entry_label=str(parking_data.get("entry_label", "parking entry")),
        approach_segments=frozenset(
            str(name) for name in parking_data.get("approach_segments", PARKING_CAMERA_APPROACH_SEGMENTS)
        ),
        entry_segments=frozenset(
            str(name) for name in parking_data.get("entry_segments", PARKING_CAMERA_ENTRY_SEGMENTS)
        ),
        entry_max_distance_m=float(parking_data.get("entry_max_distance_m", PARKING_ENTRY_MAX_DISTANCE_M)),
        sign_model_name_tokens=tuple(
            str(token).lower()
            for token in parking_data.get("sign_model_name_tokens", PARKING_SIGN_MODEL_NAME_TOKENS)
        ),
        sign_face_local_yaw_offset_rad=float(
            parking_data.get("sign_face_local_yaw_offset_rad", PARKING_SIGN_FACE_LOCAL_YAW_OFFSET_RAD)
        ),
        sign_face_min_dot=float(parking_data.get("sign_face_min_dot", PARKING_SIGN_FACE_MIN_DOT)),
        sign_face_max_distance_m=float(
            parking_data.get("sign_face_max_distance_m", PARKING_SIGN_FACE_MAX_DISTANCE_M)
        ),
        early_entry_anchors_px=_points_from_config(
            parking_data.get("early_entry_anchors_px", PARKING_EARLY_ENTRY_ANCHORS_PX),
            "parking.early_entry_anchors_px",
        ),
        orthogonal_tail_px=_points_from_config(
            parking_data.get("orthogonal_tail_px", PARKING_ORTHOGONAL_TAIL_PX),
            "parking.orthogonal_tail_px",
        ),
        bay_goal_points_px=_points_from_config(
            parking_data.get("bay_goal_points_px", PARKING_BAY_GOAL_POINTS_PX),
            "parking.bay_goal_points_px",
        ),
        initial_heading_weight_m=float(
            parking_data.get("initial_heading_weight_m", PARKING_INITIAL_HEADING_WEIGHT_M)
        ),
        final_yaw_rad=float(parking_data.get("final_yaw_rad", PARKING_FINAL_YAW_RAD)),
        final_heading_weight_m=float(parking_data.get("final_heading_weight_m", PARKING_FINAL_HEADING_WEIGHT_M)),
        final_straight_extension_px=float(
            parking_data.get("final_straight_extension_px", PARKING_FINAL_STRAIGHT_EXTENSION_PX)
        ),
        semantic_route_enabled=bool(
            parking_data.get("semantic_route_enabled", PARKING_SEMANTIC_ROUTE_ENABLED)
        ),
        semantic_staging_pose_px=_pose3_from_config(
            parking_data.get("semantic_staging_pose_px", PARKING_SEMANTIC_STAGING_POSE_PX),
            "parking.semantic_staging_pose_px",
        ),
        semantic_entry_pose_px=_pose3_from_config(
            parking_data.get("semantic_entry_pose_px", PARKING_SEMANTIC_ENTRY_POSE_PX),
            "parking.semantic_entry_pose_px",
        ),
        semantic_goal_pose_px=_pose3_from_config(
            parking_data.get("semantic_goal_pose_px", PARKING_SEMANTIC_GOAL_POSE_PX),
            "parking.semantic_goal_pose_px",
        ),
        semantic_final_extension_px=float(
            parking_data.get("semantic_final_extension_px", PARKING_SEMANTIC_FINAL_EXTENSION_PX)
        ),
    )

    traffic_light_data = data.get("traffic_lights", {})
    if not isinstance(traffic_light_data, dict):
        raise ValueError("traffic_lights must be an object.")
    raw_scene_models = traffic_light_data.get("scene_models")
    if raw_scene_models is None:
        raw_scene_models = [
            {"name": name, "default_pose": pose}
            for name, pose in TRAFFIC_LIGHT_SCENE_DEFAULT_POSES.items()
        ]
    if not isinstance(raw_scene_models, (list, tuple)):
        raise ValueError("traffic_lights.scene_models must be a list.")
    scene_model_names: list[str] = []
    default_poses: dict[str, tuple[float, float, float, float]] = {}
    for index, scene_model in enumerate(raw_scene_models):
        if not isinstance(scene_model, dict):
            raise ValueError(f"traffic_lights.scene_models[{index}] must be an object.")
        model_name = str(scene_model["name"])
        scene_model_names.append(model_name)
        default_poses[model_name] = _pose4_from_config(
            scene_model.get("default_pose", TRAFFIC_LIGHT_SCENE_DEFAULT_POSES.get(model_name, (0.0, 0.0, 0.0, 0.0))),
            f"traffic_lights.scene_models[{index}].default_pose",
        )

    indicator_models = _indicator_models_from_config(
        traffic_light_data.get("indicator_models", TRAFFIC_LIGHT_INDICATOR_MODELS),
        "traffic_lights.indicator_models",
    )
    missing_indicator_parents = sorted(
        {
            parent_name
            for entries in indicator_models.values()
            for _, parent_name, _ in entries
            if parent_name not in default_poses
        }
    )
    if missing_indicator_parents:
        raise ValueError(
            "traffic_lights.indicator_models references unknown parent model(s): "
            + ", ".join(missing_indicator_parents)
        )
    traffic_lights = TrafficLightConfig(
        scene_model_names=tuple(scene_model_names),
        default_poses=default_poses,
        release_fov_rad=float(traffic_light_data.get("release_fov_rad", TRAFFIC_LIGHT_SCENE_RELEASE_FOV_RAD)),
        indicator_hidden_pose=_pose3_from_config(
            traffic_light_data.get("indicator_hidden_pose", TRAFFIC_LIGHT_INDICATOR_HIDDEN_POSE),
            "traffic_lights.indicator_hidden_pose",
        ),
        indicator_local_xy_m=_point_from_config(
            traffic_light_data.get(
                "indicator_local_xy_m",
                (TRAFFIC_LIGHT_INDICATOR_LOCAL_X_M, TRAFFIC_LIGHT_INDICATOR_LOCAL_Y_M),
            ),
            "traffic_lights.indicator_local_xy_m",
        ),
        indicator_models=indicator_models,
    )

    zones: list[RouteZone] = []
    raw_zones = data.get("zones", ())
    if not isinstance(raw_zones, (list, tuple)):
        raise ValueError("zones must be a list.")
    for index, zone_data in enumerate(raw_zones):
        if not isinstance(zone_data, dict):
            raise ValueError(f"zones[{index}] must be an object.")
        zones.append(
            RouteZone(
                name=str(zone_data["name"]),
                kind=str(zone_data["kind"]),
                min_x_m=float(zone_data["min_x_m"]),
                max_x_m=float(zone_data["max_x_m"]),
                min_y_m=float(zone_data["min_y_m"]),
                max_y_m=float(zone_data["max_y_m"]),
            )
        )

    return RouteGraphConfig(
        start_segment_names=start_segment_names,
        segments=segments,
        additional_directed_segments=additional_directed_segments,
        behavior=behavior,
        parking=parking,
        traffic_lights=traffic_lights,
        zones=tuple(zones),
    )


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = yaw * 0.5
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def limit_guided_yaw(
    target_yaw: float,
    dt: float,
    last_yaw: float | None,
    last_yaw_rate: float,
    yaw_rate_limit: float,
    yaw_accel_limit: float,
) -> tuple[float, float]:
    if last_yaw is None or dt <= 1e-4:
        return target_yaw, 0.0

    yaw_error = normalize_angle(target_yaw - last_yaw)
    desired_yaw_rate = yaw_error / dt
    if yaw_rate_limit > 0.0:
        desired_yaw_rate = min(max(desired_yaw_rate, -yaw_rate_limit), yaw_rate_limit)

    if yaw_accel_limit > 0.0:
        rate_step = yaw_accel_limit * dt
        desired_yaw_rate = min(
            max(desired_yaw_rate, last_yaw_rate - rate_step),
            last_yaw_rate + rate_step,
        )

    yaw_step = desired_yaw_rate * dt
    if abs(yaw_step) > abs(yaw_error):
        yaw_step = yaw_error
        desired_yaw_rate = yaw_step / dt

    return normalize_angle(last_yaw + yaw_step), desired_yaw_rate


def quaternion_to_rpy(quaternion) -> tuple[float, float, float]:
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))

    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
    return roll, pitch, yaw


def map_point_to_model(
    px: float,
    py: float,
    route_offset_x_m: float = 0.0,
    route_offset_y_m: float = 0.0,
) -> tuple[float, float]:
    model_x = (px / MAP_TEXTURE_SIZE - 0.5) * MODEL_SIZE_M
    model_y = (0.5 - py / MAP_TEXTURE_SIZE) * MODEL_SIZE_M
    model_x += route_offset_x_m
    model_y += route_offset_y_m
    return model_x, model_y


def model_point_to_map(
    x: float,
    y: float,
    map_offset_x_m: float = MAP_VISUAL_OFFSET_X_M,
    map_offset_y_m: float = MAP_VISUAL_OFFSET_Y_M,
) -> tuple[float, float]:
    local_x = x - map_offset_x_m
    local_y = y - map_offset_y_m
    map_x = (local_x / MODEL_SIZE_M + 0.5) * MAP_TEXTURE_SIZE
    map_y = (0.5 - local_y / MODEL_SIZE_M) * MAP_TEXTURE_SIZE
    return map_x, map_y


def visual_map_point_to_model(
    px: float,
    py: float,
    map_offset_x_m: float = MAP_VISUAL_OFFSET_X_M,
    map_offset_y_m: float = MAP_VISUAL_OFFSET_Y_M,
) -> tuple[float, float]:
    model_x = (px / MAP_TEXTURE_SIZE - 0.5) * MODEL_SIZE_M + map_offset_x_m
    model_y = (0.5 - py / MAP_TEXTURE_SIZE) * MODEL_SIZE_M + map_offset_y_m
    return model_x, model_y


def is_parking_sign_model_name(
    model_name: str,
    tokens: tuple[str, ...] = PARKING_SIGN_MODEL_NAME_TOKENS,
) -> bool:
    lower_name = model_name.lower()
    return "sign" in lower_name and any(token in lower_name for token in tokens)


def camera_sign_scene_identity_from_model_name(model_name: str) -> tuple[str, str | None] | None:
    lower_name = model_name.lower()
    if "sign" not in lower_name and not lower_name.startswith("palette_"):
        return None
    for kind, maneuver, tokens in CAMERA_SIGN_SCENE_IDENTIFIERS:
        if any(token in lower_name for token in tokens):
            return kind, maneuver
    return None


def footprint_corners_model(
    x: float,
    y: float,
    yaw: float,
    front_m: float,
    rear_m: float,
    half_width_m: float,
) -> np.ndarray:
    local = np.array(
        [
            [front_m, half_width_m],
            [front_m, -half_width_m],
            [-rear_m, -half_width_m],
            [-rear_m, half_width_m],
        ],
        dtype=np.float32,
    )
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    return local @ rotation.T + np.array([x, y], dtype=np.float32)


def model_polygon_to_map(
    points: np.ndarray,
    route_offset_x_m: float = DEFAULT_ROUTE_OFFSET_X_M,
    route_offset_y_m: float = DEFAULT_ROUTE_OFFSET_Y_M,
) -> np.ndarray:
    return np.round(
        [
            model_point_to_map(float(x), float(y), route_offset_x_m, route_offset_y_m)
            for x, y in points
        ]
    ).astype(np.int32)


@dataclass
class FootprintRoadCheck:
    corners_px: np.ndarray
    corners_in_image: bool
    on_road: bool
    min_clearance_px: float


def check_road_footprint(
    clearance: np.ndarray,
    x: float,
    y: float,
    yaw: float,
    front_m: float,
    rear_m: float,
    half_width_m: float,
    route_offset_x_m: float = DEFAULT_ROUTE_OFFSET_X_M,
    route_offset_y_m: float = DEFAULT_ROUTE_OFFSET_Y_M,
) -> FootprintRoadCheck:
    corners = footprint_corners_model(x, y, yaw, front_m, rear_m, half_width_m)
    corners_px = model_polygon_to_map(corners, route_offset_x_m, route_offset_y_m)
    footprint_mask = np.zeros_like(clearance, dtype=np.uint8)
    cv2.fillPoly(footprint_mask, [corners_px], 1)
    footprint_region = footprint_mask > 0
    height, width = clearance.shape[:2]
    corners_in_image = all(0 <= corner[0] < width and 0 <= corner[1] < height for corner in corners_px)
    if not np.any(footprint_region):
        return FootprintRoadCheck(corners_px, corners_in_image, False, 0.0)

    min_clearance_px = float(np.min(clearance[footprint_region]))
    on_road = corners_in_image and bool(np.all(clearance[footprint_region] > 0.0))
    return FootprintRoadCheck(corners_px, corners_in_image, on_road, min_clearance_px)


def resolve_map_image_path() -> Path:
    override_path = os.environ.get(MAP_IMAGE_ENV, "").strip()
    if override_path:
        path = Path(override_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise FileNotFoundError(f"Map image override does not exist: {path}")
        return path

    map_path = share_file("maps", MAP_VISUAL_NAME)
    if map_path is None:
        raise FileNotFoundError(f"Could not find maps/{MAP_VISUAL_NAME}")
    return map_path


def load_map_image() -> np.ndarray:
    map_path = resolve_map_image_path()

    image = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read map image: {map_path}")
    if image.shape[0] != int(MAP_TEXTURE_SIZE) or image.shape[1] != int(MAP_TEXTURE_SIZE):
        image = cv2.resize(image, (int(MAP_TEXTURE_SIZE), int(MAP_TEXTURE_SIZE)), interpolation=cv2.INTER_NEAREST)
    return image


def flood_outside(whiteish: np.ndarray) -> np.ndarray:
    height, width = whiteish.shape
    outside = np.zeros((height, width), dtype=np.uint8)
    stack = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]

    while stack:
        x, y = stack.pop()
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        if outside[y, x] or not whiteish[y, x]:
            continue
        outside[y, x] = 1
        stack.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))

    return outside.astype(bool)


def build_safe_road_mask(map_image: np.ndarray, clearance_px: int) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([40, 50, 40], dtype=np.uint8), np.array([95, 255, 255], dtype=np.uint8)) > 0
    yellow = cv2.inRange(hsv, np.array([15, 60, 80], dtype=np.uint8), np.array([45, 255, 255], dtype=np.uint8)) > 0

    whiteish = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 180)
    inside_map = ~flood_outside(whiteish)
    blocked = (~inside_map) | green | yellow
    curb_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (CURB_VISUAL_BUFFER_PX * 2 + 1, CURB_VISUAL_BUFFER_PX * 2 + 1),
    )
    # The rendered map filters curb pixels; keep the route off that visible halo.
    blocked = cv2.dilate(blocked.astype(np.uint8), curb_kernel) > 0
    drivable = inside_map & ~blocked

    clearance_px = max(1, int(clearance_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clearance_px * 2 + 1, clearance_px * 2 + 1))
    safe = cv2.erode(drivable.astype(np.uint8), kernel) > 0
    if not np.any(safe):
        raise RuntimeError("Safe road mask is empty. Reduce clearance_px.")

    clearance = cv2.distanceTransform(drivable.astype(np.uint8), cv2.DIST_L2, 5)
    return safe, clearance


def nearest_safe_point(point: tuple[float, float], safe_mask: np.ndarray) -> tuple[int, int]:
    height, width = safe_mask.shape
    x = int(round(min(max(point[0], 0.0), width - 1.0)))
    y = int(round(min(max(point[1], 0.0), height - 1.0)))
    if safe_mask[y, x]:
        return x, y

    safe_y, safe_x = np.nonzero(safe_mask)
    if len(safe_x) == 0:
        raise RuntimeError("No safe road pixels are available.")

    distances = (safe_x - x) * (safe_x - x) + (safe_y - y) * (safe_y - y)
    nearest_index = int(np.argmin(distances))
    return int(safe_x[nearest_index]), int(safe_y[nearest_index])


def astar_path(
    start_point: tuple[float, float],
    goal_point: tuple[float, float],
    safe_mask: np.ndarray,
    clearance: np.ndarray,
) -> list[tuple[int, int]]:
    start = nearest_safe_point(start_point, safe_mask)
    goal = nearest_safe_point(goal_point, safe_mask)
    if start == goal:
        return [start]

    height, width = safe_mask.shape
    open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    visited: set[tuple[int, int]] = set()

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))

    neighbors = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy = current
        current_g = g_score[current]
        for dx, dy, step_cost in neighbors:
            nx = cx + dx
            ny = cy + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height or not safe_mask[ny, nx]:
                continue
            if dx != 0 and dy != 0 and (not safe_mask[cy, nx] or not safe_mask[ny, cx]):
                continue

            center_bias = ROUTE_CENTERLINE_WEIGHT / max(float(clearance[ny, nx]), 1.0)
            tentative_g = current_g + step_cost * (1.0 + center_bias)
            neighbor = (nx, ny)
            if tentative_g >= g_score.get(neighbor, float("inf")):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            heapq.heappush(open_heap, (tentative_g + heuristic(neighbor, goal), neighbor))

    raise RuntimeError(f"Could not route safely from {start} to {goal}. Reduce clearance_px.")


def line_is_safe(
    start: tuple[float, float],
    end: tuple[float, float],
    clearance: np.ndarray,
) -> bool:
    height, width = clearance.shape
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    sample_count = max(2, int(math.ceil(max(abs(dx), abs(dy)))) + 1)
    for alpha in np.linspace(0.0, 1.0, sample_count):
        x = int(round(start[0] + dx * float(alpha)))
        y = int(round(start[1] + dy * float(alpha)))
        if x < 0 or x >= width or y < 0 or y >= height or clearance[y, x] <= 0.0:
            return False
    return True


def simplify_straight_route_segment(
    image_points: list[tuple[int, int]],
    start_anchor: tuple[float, float],
    goal_anchor: tuple[float, float],
    clearance: np.ndarray,
) -> list[tuple[float, float]]:
    if len(image_points) < 3:
        return [(float(x), float(y)) for x, y in image_points]

    anchor_dx = goal_anchor[0] - start_anchor[0]
    anchor_dy = goal_anchor[1] - start_anchor[1]
    if min(abs(anchor_dx), abs(anchor_dy)) > STRAIGHT_ROUTE_AXIS_TOLERANCE_PX:
        return [(float(x), float(y)) for x, y in image_points]

    points = np.array(image_points, dtype=np.float32)
    delta = points[-1] - points[0]
    length = float(np.linalg.norm(delta))
    if length < 1.0:
        return [(float(x), float(y)) for x, y in image_points]

    tangent = delta / length
    axis_alignment = max(abs(float(tangent[0])), abs(float(tangent[1])))
    if axis_alignment < 0.985:
        return [(float(x), float(y)) for x, y in image_points]

    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    lateral = (points - points[0]) @ normal
    lateral_range = float(np.max(lateral) - np.min(lateral))
    if lateral_range > STRAIGHT_ROUTE_SIMPLIFY_MAX_LATERAL_PX:
        return [(float(x), float(y)) for x, y in image_points]

    start = (float(points[0][0]), float(points[0][1]))
    end = (float(points[-1][0]), float(points[-1][1]))
    if not line_is_safe(start, end, clearance):
        return [(float(x), float(y)) for x, y in image_points]
    return [start, end]


def route_through_anchors(
    anchors: list[tuple[float, float]],
    safe_mask: np.ndarray,
    clearance: np.ndarray,
) -> list[tuple[float, float]]:
    routed: list[tuple[float, float]] = []
    for index in range(len(anchors) - 1):
        segment = astar_path(anchors[index], anchors[index + 1], safe_mask, clearance)
        segment = simplify_straight_route_segment(segment, anchors[index], anchors[index + 1], clearance)
        if routed:
            segment = segment[1:]
        routed.extend(segment)

    return remove_collinear_points(routed)


def manual_centerline_points(
    anchors: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for x, y in anchors:
        point = (float(x), float(y))
        if not points or math.hypot(point[0] - points[-1][0], point[1] - points[-1][1]) > 1e-6:
            points.append(point)
    return points


def extend_terminal_straight(
    points: list[tuple[float, float]],
    extension_px: float,
    clearance: np.ndarray,
) -> list[tuple[float, float]]:
    if len(points) < 2 or extension_px <= 0.0:
        return points

    prev_x, prev_y = points[-2]
    end_x, end_y = points[-1]
    direction_x = end_x - prev_x
    direction_y = end_y - prev_y
    direction_length = math.hypot(direction_x, direction_y)
    if direction_length < 1e-6:
        return points

    unit_x = direction_x / direction_length
    unit_y = direction_y / direction_length
    step_count = max(1, int(math.ceil(extension_px)))
    for distance_px in np.linspace(extension_px, 1.0, step_count):
        extended = (end_x + unit_x * float(distance_px), end_y + unit_y * float(distance_px))
        if line_is_safe((end_x, end_y), extended, clearance):
            return points + [extended]

    return points


def remove_collinear_points(points: list[tuple[float, float]], epsilon: float = 1e-5) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points

    simplified = [points[0]]
    for index in range(1, len(points) - 1):
        prev_x, prev_y = simplified[-1]
        x, y = points[index]
        next_x, next_y = points[index + 1]
        cross = (x - prev_x) * (next_y - y) - (y - prev_y) * (next_x - x)
        if abs(cross) > epsilon:
            simplified.append(points[index])
    simplified.append(points[-1])
    return simplified


def segment_lengths(points: list[tuple[float, float]]) -> tuple[list[float], float]:
    cumulative = [0.0]
    total = 0.0
    for index in range(len(points) - 1):
        point = points[index]
        next_point = points[index + 1]
        total += math.hypot(next_point[0] - point[0], next_point[1] - point[1])
        cumulative.append(total)
    return cumulative, total


@dataclass
class TrackSegment:
    name: str
    start_node: str
    end_node: str
    points: list[tuple[float, float]]
    cumulative: list[float]
    length: float
    inner_route: bool = False
    parking_route: bool = False
    terminal_stop: bool = False

    def xy_at(self, distance: float) -> tuple[float, float]:
        distance = min(max(distance, 0.0), self.length)
        for index in range(len(self.points) - 1):
            if self.cumulative[index] <= distance <= self.cumulative[index + 1]:
                start = self.points[index]
                end = self.points[index + 1]
                segment_start = self.cumulative[index]
                segment_length = max(self.cumulative[index + 1] - segment_start, 1e-6)
                alpha = (distance - segment_start) / segment_length
                x = start[0] + (end[0] - start[0]) * alpha
                y = start[1] + (end[1] - start[1]) * alpha
                return x, y

        return self.points[-1][0], self.points[-1][1]

    def sample(self, distance: float) -> tuple[float, float, float]:
        x, y = self.xy_at(distance)
        lookahead_distance = min(self.length, distance + YAW_LOOKAHEAD_M)
        if abs(lookahead_distance - distance) < 1e-4:
            lookahead_distance = max(0.0, distance - YAW_LOOKAHEAD_M)
            look_x, look_y = self.xy_at(lookahead_distance)
            yaw = math.atan2(y - look_y, x - look_x)
        else:
            look_x, look_y = self.xy_at(lookahead_distance)
            yaw = math.atan2(look_y - y, look_x - x)

        if len(self.points) >= 2:
            return x, y, yaw
        return x, y, 0.0

    def project(self, x: float, y: float) -> "RouteProjection":
        best_distance_on_segment = 0.0
        best_x = self.points[0][0]
        best_y = self.points[0][1]
        best_distance_sq = (x - best_x) * (x - best_x) + (y - best_y) * (y - best_y)

        for index in range(len(self.points) - 1):
            start_x, start_y = self.points[index]
            end_x, end_y = self.points[index + 1]
            segment_x = end_x - start_x
            segment_y = end_y - start_y
            segment_length_sq = segment_x * segment_x + segment_y * segment_y
            if segment_length_sq <= 1e-12:
                t = 0.0
            else:
                t = ((x - start_x) * segment_x + (y - start_y) * segment_y) / segment_length_sq
                t = min(max(t, 0.0), 1.0)

            projected_x = start_x + segment_x * t
            projected_y = start_y + segment_y * t
            dx = x - projected_x
            dy = y - projected_y
            distance_sq = dx * dx + dy * dy
            if distance_sq < best_distance_sq:
                segment_length = self.cumulative[index + 1] - self.cumulative[index]
                best_distance_on_segment = self.cumulative[index] + segment_length * t
                best_x = projected_x
                best_y = projected_y
                best_distance_sq = distance_sq

        _, _, yaw = self.sample(best_distance_on_segment)
        return RouteProjection(
            self,
            best_distance_on_segment,
            best_x,
            best_y,
            yaw,
            math.sqrt(best_distance_sq),
        )


@dataclass
class RouteProjection:
    segment: TrackSegment
    distance_on_segment: float
    x: float
    y: float
    yaw: float
    distance_m: float


@dataclass
class MotionSample:
    stamp_ns: int
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    speed_xy: float
    yaw_rate: float


@dataclass
class SceneSignPose:
    name: str
    x: float
    y: float
    z: float
    yaw: float
    kind: str | None = None
    maneuver: str | None = None


@dataclass
class TrafficLightSceneTarget:
    name: str
    distance_m: float
    forward_m: float
    lateral_m: float
    angle_rad: float


class DirectedTrack:
    def __init__(
        self,
        reverse_direction: bool,
        branch_probability: float,
        route_seed: int,
        route_explore_enabled: bool,
        clearance_px: int,
        route_offset_x_m: float = DEFAULT_ROUTE_OFFSET_X_M,
        route_offset_y_m: float = DEFAULT_ROUTE_OFFSET_Y_M,
    ) -> None:
        self.segments: dict[str, TrackSegment] = {}
        self.outgoing: dict[str, list[TrackSegment]] = {}
        self.random = random.Random(route_seed)
        self.branch_probability = max(0.0, min(1.0, branch_probability))
        self.route_explore_enabled = bool(route_explore_enabled)
        self.segment_visit_counts: dict[str, int] = {}
        self.distance_since_branch = MIN_BRANCH_DISTANCE_M
        self.route_offset_x_m = float(route_offset_x_m)
        self.route_offset_y_m = float(route_offset_y_m)
        self.pending_maneuver: str | None = None
        self.pending_maneuver_source: str | None = None
        self.last_decision_text = "route/default"
        self.stopped_at_terminal = False
        self.parking_started = False
        self.route_config = load_route_graph_config()
        self.behavior_config = self.route_config.behavior
        self.maneuver_max_branch_distance_m = self.behavior_config.maneuver_max_branch_distance_m
        self.nearby_maneuver_route_distance_m = self.behavior_config.nearby_maneuver_route_distance_m
        self.parking_config = self.route_config.parking
        self.parking_dynamic_route_name = self.parking_config.dynamic_route_name
        self.parking_dynamic_start_node = self.parking_config.dynamic_start_node
        self.parking_stop_node = self.parking_config.stop_node
        self.parking_entry_label = self.parking_config.entry_label
        self.parking_approach_segments = self.parking_config.approach_segments
        self.parking_entry_segments = self.parking_config.entry_segments
        self.parking_entry_max_distance_m = self.parking_config.entry_max_distance_m
        self.parking_sign_model_name_tokens = self.parking_config.sign_model_name_tokens
        self.parking_sign_face_local_yaw_offset_rad = self.parking_config.sign_face_local_yaw_offset_rad
        self.parking_sign_face_min_dot = self.parking_config.sign_face_min_dot
        self.parking_sign_face_max_distance_m = self.parking_config.sign_face_max_distance_m
        self.parking_early_entry_anchors_px = self.parking_config.early_entry_anchors_px
        self.parking_orthogonal_tail_px = self.parking_config.orthogonal_tail_px
        self.parking_bay_goal_points_px = self.parking_config.bay_goal_points_px
        self.parking_initial_heading_weight_m = self.parking_config.initial_heading_weight_m
        self.parking_final_yaw_rad = self.parking_config.final_yaw_rad
        self.parking_final_heading_weight_m = self.parking_config.final_heading_weight_m
        self.parking_final_straight_extension_px = self.parking_config.final_straight_extension_px
        self.parking_semantic_route_enabled = self.parking_config.semantic_route_enabled
        self.parking_semantic_staging_pose_px = self.parking_config.semantic_staging_pose_px
        self.parking_semantic_entry_pose_px = self.parking_config.semantic_entry_pose_px
        self.parking_semantic_goal_pose_px = self.parking_config.semantic_goal_pose_px
        self.parking_semantic_final_extension_px = self.parking_config.semantic_final_extension_px
        self.route_zones = self.route_config.zones
        if not self.parking_orthogonal_tail_px:
            raise ValueError("parking.orthogonal_tail_px must contain at least one point.")

        map_image = load_map_image()
        self.safe_mask, self.clearance = build_safe_road_mask(map_image, clearance_px)
        self.parking_safe_mask, _ = build_safe_road_mask(map_image, PARKING_CLEARANCE_PX)
        self.current_segment = self._build_segments(reverse_direction)
        self._validate_runtime_config(reverse_direction)
        self.distance_on_segment = 0.0
        self._mark_segment_visit(self.current_segment)

    def set_pending_maneuver(self, maneuver: str, source_name: str) -> None:
        if maneuver not in ROUTE_MANEUVERS:
            return
        self.pending_maneuver = maneuver
        self.pending_maneuver_source = source_name

    def segment_is_parking_choice(self, segment: TrackSegment) -> bool:
        return segment.parking_route or (segment.terminal_stop and "parking" in segment.name.lower())

    def _split_route_choices(
        self,
        choices: list[TrackSegment],
    ) -> tuple[list[TrackSegment], list[TrackSegment], list[TrackSegment]]:
        parking_choices = [segment for segment in choices if self.segment_is_parking_choice(segment)]
        drive_choices = [segment for segment in choices if not self.segment_is_parking_choice(segment)]
        outer_choices = [segment for segment in drive_choices if not segment.inner_route]
        inner_choices = [segment for segment in drive_choices if segment.inner_route and not segment.parking_route]
        return parking_choices, outer_choices, inner_choices

    def _choices_have_controlled_branch(self, segment: TrackSegment, choices: list[TrackSegment]) -> bool:
        parking_choices, outer_choices, inner_choices = self._split_route_choices(choices)
        if parking_choices:
            return True
        if segment.inner_route and not segment.parking_route:
            return bool(outer_choices or len(inner_choices) > 1)
        return bool(inner_choices)

    def _default_next_segment(self, segment: TrackSegment, choices: list[TrackSegment]) -> TrackSegment | None:
        parking_choices, outer_choices, inner_choices = self._split_route_choices(choices)
        if segment.inner_route and not segment.parking_route:
            if inner_choices:
                return inner_choices[0]
            return None
        if outer_choices:
            return outer_choices[0]
        if inner_choices:
            return inner_choices[0]
        if parking_choices:
            return None
        return choices[0] if choices else None

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def route_point_count(self) -> int:
        return sum(len(segment.points) for segment in self.segments.values())

    def point_in_zone(self, x: float, y: float, kind: str) -> bool:
        return any(zone.kind == kind and zone.contains(x, y) for zone in self.route_zones)

    def _validate_runtime_config(self, reverse_direction: bool) -> None:
        if reverse_direction:
            missing_approach = sorted(name for name in self.parking_approach_segments if name not in self.segments)
            missing_entry = sorted(name for name in self.parking_entry_segments if name not in self.segments)
            if missing_approach or missing_entry:
                details = []
                if missing_approach:
                    details.append(f"approach={missing_approach}")
                if missing_entry:
                    details.append(f"entry={missing_entry}")
                raise ValueError("parking config references route segments not built for reverse route: " + "; ".join(details))

    def _add_segment(
        self,
        name: str,
        start_node: str,
        end_node: str,
        anchors: list[tuple[float, float]],
        inner_route: bool,
        reverse_direction: bool,
        parking_route: bool = False,
        terminal_stop: bool = False,
        manual_centerline: bool = False,
    ) -> TrackSegment:
        if reverse_direction:
            name = f"{name}_reverse"
            start_node, end_node = end_node, start_node
            anchors = list(reversed(anchors))

        if manual_centerline:
            image_points = manual_centerline_points(anchors)
        else:
            image_points = route_through_anchors(anchors, self.safe_mask, self.clearance)
        model_points = [
            map_point_to_model(px, py, self.route_offset_x_m, self.route_offset_y_m)
            for px, py in image_points
        ]
        cumulative, length = segment_lengths(model_points)
        segment = TrackSegment(
            name,
            start_node,
            end_node,
            model_points,
            cumulative,
            length,
            inner_route,
            parking_route,
            terminal_stop,
        )
        self.segments[name] = segment
        self.outgoing.setdefault(start_node, []).append(segment)
        return segment

    def _add_directed_segment(
        self,
        name: str,
        start_node: str,
        end_node: str,
        anchors: list[tuple[float, float]],
        inner_route: bool = False,
        parking_route: bool = False,
        terminal_stop: bool = False,
        manual_centerline: bool = False,
    ) -> TrackSegment:
        route_mask = self.parking_safe_mask if parking_route else self.safe_mask
        if manual_centerline:
            image_points = manual_centerline_points(anchors)
        else:
            image_points = route_through_anchors(anchors, route_mask, self.clearance)
        model_points = [
            map_point_to_model(px, py, self.route_offset_x_m, self.route_offset_y_m)
            for px, py in image_points
        ]
        cumulative, length = segment_lengths(model_points)
        segment = TrackSegment(
            name,
            start_node,
            end_node,
            model_points,
            cumulative,
            length,
            inner_route,
            parking_route,
            terminal_stop,
        )
        self.segments[name] = segment
        self.outgoing.setdefault(start_node, []).append(segment)
        return segment

    def _make_segment_from_model_points(
        self,
        name: str,
        start_node: str,
        end_node: str,
        model_points: list[tuple[float, float]],
        inner_route: bool = False,
        parking_route: bool = False,
        terminal_stop: bool = False,
    ) -> TrackSegment:
        points = remove_collinear_points(model_points)
        cumulative, length = segment_lengths(points)
        return TrackSegment(
            name,
            start_node,
            end_node,
            points,
            cumulative,
            length,
            inner_route,
            parking_route,
            terminal_stop,
        )

    def start_parking_route(self, source_name: str, current_yaw: float | None = None) -> bool:
        if self.parking_started or self.current_segment.parking_route or self.stopped_at_terminal:
            return False

        current_x, current_y, route_yaw = self.sample()
        segment = self._plan_parking_segment(
            current_x,
            current_y,
            route_yaw if current_yaw is None else current_yaw,
        )
        if segment is None:
            self.last_decision_text = f"route/parking_failed from {source_name}: no safe path"
            return False

        self.segments[segment.name] = segment
        self.current_segment = segment
        self.distance_on_segment = 0.0
        self.distance_since_branch = 0.0
        self.pending_maneuver = None
        self.pending_maneuver_source = None
        self.stopped_at_terminal = False
        self.parking_started = True
        self.last_decision_text = f"route/parking_dynamic from {source_name}: {segment.name}"
        return True

    def _plan_parking_segment(
        self,
        current_x: float,
        current_y: float,
        current_yaw: float,
    ) -> TrackSegment | None:
        semantic_segment = self._plan_semantic_parking_segment(current_x, current_y)
        if semantic_segment is not None:
            return semantic_segment

        start_px = model_point_to_map(current_x, current_y, self.route_offset_x_m, self.route_offset_y_m)
        if not self._parking_entry_start_is_close(start_px):
            return None

        orthogonal_segment = self._plan_orthogonal_parking_segment(start_px)
        if orthogonal_segment is not None:
            return orthogonal_segment

        candidates: list[tuple[float, TrackSegment]] = []

        for goal_px in self.parking_bay_goal_points_px:
            try:
                image_points = route_through_anchors(
                    [start_px, *self.parking_early_entry_anchors_px, goal_px],
                    self.parking_safe_mask,
                    self.clearance,
                )
            except RuntimeError:
                continue
            if len(image_points) < 2:
                continue

            if math.hypot(image_points[0][0] - start_px[0], image_points[0][1] - start_px[1]) > 0.5:
                image_points = [start_px] + image_points
            else:
                image_points[0] = start_px

            image_points = extend_terminal_straight(
                image_points,
                self.parking_final_straight_extension_px,
                self.clearance,
            )
            model_points = [
                map_point_to_model(px, py, self.route_offset_x_m, self.route_offset_y_m)
                for px, py in image_points
            ]
            segment = self._make_segment_from_model_points(
                self.parking_dynamic_route_name,
                self.parking_dynamic_start_node,
                self.parking_stop_node,
                model_points,
                inner_route=True,
                parking_route=True,
                terminal_stop=True,
            )
            _, _, initial_yaw = segment.sample(0.0)
            _, _, final_yaw = segment.sample(segment.length)
            initial_heading_error = abs(normalize_angle(initial_yaw - current_yaw))
            final_heading_error = abs(normalize_angle(final_yaw - self.parking_final_yaw_rad))
            score = (
                segment.length
                + self.parking_initial_heading_weight_m * initial_heading_error
                + self.parking_final_heading_weight_m * final_heading_error
            )
            candidates.append((score, segment))

        if not candidates:
            return None

        return min(candidates, key=lambda item: item[0])[1]

    def _plan_semantic_parking_segment(
        self,
        current_x: float,
        current_y: float,
    ) -> TrackSegment | None:
        if not self.parking_semantic_route_enabled:
            return None

        start_px = model_point_to_map(current_x, current_y, self.route_offset_x_m, self.route_offset_y_m)
        staging_px = self.parking_semantic_staging_pose_px[:2]
        entry_px = self.parking_semantic_entry_pose_px[:2]
        goal_px = self.parking_semantic_goal_pose_px[:2]
        anchors = [start_px, staging_px, entry_px, goal_px]

        try:
            image_points = route_through_anchors(anchors, self.parking_safe_mask, self.clearance)
        except RuntimeError:
            return None
        if len(image_points) < 2:
            return None

        if math.hypot(image_points[0][0] - start_px[0], image_points[0][1] - start_px[1]) > 0.5:
            image_points = [start_px] + image_points
        else:
            image_points[0] = start_px

        image_points = self._append_terminal_yaw_extension(
            image_points,
            self.parking_semantic_goal_pose_px[2],
            self.parking_semantic_final_extension_px,
        )
        model_points = [
            map_point_to_model(px, py, self.route_offset_x_m, self.route_offset_y_m)
            for px, py in image_points
        ]
        return self._make_segment_from_model_points(
            self.parking_dynamic_route_name,
            self.parking_dynamic_start_node,
            self.parking_stop_node,
            model_points,
            inner_route=True,
            parking_route=True,
            terminal_stop=True,
        )

    def _append_terminal_yaw_extension(
        self,
        points: list[tuple[float, float]],
        yaw_rad: float,
        extension_px: float,
    ) -> list[tuple[float, float]]:
        if len(points) < 1 or extension_px <= 0.0:
            return points

        end_x, end_y = points[-1]
        unit_x = math.cos(yaw_rad)
        unit_y = -math.sin(yaw_rad)
        step_count = max(1, int(math.ceil(extension_px)))
        for distance_px in np.linspace(extension_px, 1.0, step_count):
            extended = (
                end_x + unit_x * float(distance_px),
                end_y + unit_y * float(distance_px),
            )
            if line_is_safe((end_x, end_y), extended, self.clearance):
                return points + [extended]

        return points

    def _parking_entry_start_is_close(self, start_px: tuple[float, float]) -> bool:
        max_distance_px = self.parking_entry_max_distance_m * MAP_TEXTURE_SIZE / MODEL_SIZE_M
        entry_px = self.parking_orthogonal_tail_px[0]
        return (
            math.hypot(
                start_px[0] - entry_px[0],
                start_px[1] - entry_px[1],
            )
            <= max_distance_px
        )

    def _plan_orthogonal_parking_segment(
        self,
        start_px: tuple[float, float],
    ) -> TrackSegment | None:
        if not self._parking_entry_start_is_close(start_px):
            return None

        image_points = [start_px, *self.parking_orthogonal_tail_px]
        model_points = [
            map_point_to_model(px, py, self.route_offset_x_m, self.route_offset_y_m)
            for px, py in image_points
        ]
        return self._make_segment_from_model_points(
            self.parking_dynamic_route_name,
            self.parking_dynamic_start_node,
            self.parking_stop_node,
            model_points,
            inner_route=True,
            parking_route=True,
            terminal_stop=True,
        )

    def rejoin_current_segment(self, x: float, y: float) -> RouteProjection:
        projection = self.current_segment.project(x, y)
        self.distance_on_segment = min(max(projection.distance_on_segment, 0.0), self.current_segment.length)
        self.distance_since_branch = 0.0 if self.current_segment.parking_route else MIN_BRANCH_DISTANCE_M
        self.pending_maneuver = None
        self.pending_maneuver_source = None
        self.stopped_at_terminal = False
        if self.current_segment.parking_route:
            self.parking_started = True
        self.last_decision_text = f"route/rejoin_current: {projection.segment.name}"
        return projection

    def _build_segments(self, reverse_direction: bool) -> TrackSegment:
        direction_key = "reverse" if reverse_direction else "forward"
        start_segment_name = self.route_config.start_segment_names.get(direction_key)
        if not start_segment_name:
            raise RuntimeError(f"Route config is missing route.start_segments.{direction_key}.")
        start_segment: TrackSegment | None = None
        for spec in self.route_config.segments:
            segment = self._add_segment(
                spec.name,
                spec.start_node,
                spec.end_node,
                list(spec.anchors),
                spec.inner_route,
                reverse_direction,
                spec.parking_route,
                spec.terminal_stop,
                spec.manual_centerline,
            )
            if spec.name == start_segment_name:
                start_segment = segment

        for spec in self.route_config.additional_directed_segments.get(direction_key, ()):
            self._add_directed_segment(
                spec.name,
                spec.start_node,
                spec.end_node,
                list(spec.anchors),
                spec.inner_route,
                spec.parking_route,
                spec.terminal_stop,
                spec.manual_centerline,
            )

        if start_segment is None:
            raise RuntimeError(f"Route config start segment '{start_segment_name}' was not built.")
        return start_segment

    def _mark_segment_visit(self, segment: TrackSegment) -> None:
        self.segment_visit_counts[segment.name] = self.segment_visit_counts.get(segment.name, 0) + 1

    def is_direct_successor(self, candidate: TrackSegment) -> bool:
        return any(segment is candidate for segment in self.outgoing.get(self.current_segment.end_node, ()))

    def direct_successor_count(self) -> int:
        return len(self.outgoing.get(self.current_segment.end_node, ()))

    def switch_to_projection(self, projection: RouteProjection, decision_text: str) -> None:
        previous_segment = self.current_segment
        self.current_segment = projection.segment
        self.distance_on_segment = min(max(projection.distance_on_segment, 0.0), projection.segment.length)
        self.stopped_at_terminal = False
        if projection.segment.parking_route:
            self.parking_started = True
        if projection.segment is not previous_segment:
            self._mark_segment_visit(projection.segment)
            if projection.segment.inner_route and not projection.segment.parking_route:
                self.distance_since_branch = 0.0
        self.last_decision_text = decision_text

    def _exploration_choice(self, choices: list[TrackSegment]) -> TrackSegment:
        min_visits = min(self.segment_visit_counts.get(choice.name, 0) for choice in choices)
        least_visited = [
            choice for choice in choices
            if self.segment_visit_counts.get(choice.name, 0) == min_visits
        ]
        inner_unvisited = [
            choice for choice in least_visited
            if choice.inner_route and not choice.parking_route and self.segment_visit_counts.get(choice.name, 0) == 0
        ]
        if inner_unvisited:
            return min(inner_unvisited, key=lambda choice: choice.name)
        return min(least_visited, key=lambda choice: choice.name)

    def advance(self, distance: float) -> None:
        if self.stopped_at_terminal:
            return

        remaining = max(0.0, distance)
        while remaining > 0.0:
            distance_left = self.current_segment.length - self.distance_on_segment
            if remaining < distance_left:
                self.distance_on_segment += remaining
                self.distance_since_branch += remaining
                return

            remaining -= max(distance_left, 0.0)
            self.distance_since_branch += max(distance_left, 0.0)
            if self.current_segment.terminal_stop:
                self.distance_on_segment = self.current_segment.length
                self.stopped_at_terminal = True
                self.last_decision_text = f"route/terminal_stop: {self.current_segment.name}"
                return
            self._choose_next_segment()
            if self.stopped_at_terminal:
                return

    def nearest_projection(self, x: float, y: float, outer_only: bool = True) -> RouteProjection:
        best: RouteProjection | None = None
        for segment in self.segments.values():
            if outer_only and segment.inner_route:
                continue
            projection = segment.project(x, y)
            if best is None or projection.distance_m < best.distance_m:
                best = projection

        if best is None:
            raise RuntimeError("Could not find a route segment to rejoin.")
        return best

    def nearest_pose_projection(self, x: float, y: float, yaw: float, outer_only: bool = True) -> RouteProjection:
        best: RouteProjection | None = None
        best_score = float("inf")
        best_remaining = -1.0
        for segment in self.segments.values():
            if outer_only and segment.inner_route:
                continue
            projection = segment.project(x, y)
            _, _, route_yaw = segment.sample(projection.distance_on_segment)
            yaw_error = abs(normalize_angle(route_yaw - yaw))
            remaining_m = segment.length - projection.distance_on_segment
            score = projection.distance_m + 0.35 * yaw_error
            if score < best_score - 1e-6 or (abs(score - best_score) <= 1e-6 and remaining_m > best_remaining):
                best = projection
                best_score = score
                best_remaining = remaining_m

        if best is None:
            raise RuntimeError("Could not find a pose-aligned route segment to rejoin.")
        return best

    def rejoin_at_nearest(self, x: float, y: float, outer_only: bool = True) -> RouteProjection:
        projection = self.nearest_projection(x, y, outer_only)
        self.current_segment = projection.segment
        self.distance_on_segment = min(max(projection.distance_on_segment, 0.0), projection.segment.length)
        self.distance_since_branch = MIN_BRANCH_DISTANCE_M
        self.pending_maneuver = None
        self.pending_maneuver_source = None
        self.last_decision_text = f"route/rejoin_nearest: {projection.segment.name}"
        self.stopped_at_terminal = False
        if projection.segment.parking_route:
            self.parking_started = True
        return projection

    def nearby_maneuver_projection(
        self,
        maneuver: str,
        x: float,
        y: float,
        current_yaw: float,
        max_distance_m: float | None = None,
    ) -> tuple[RouteProjection, float] | None:
        if max_distance_m is None:
            max_distance_m = self.nearby_maneuver_route_distance_m
        scored: list[tuple[RouteProjection, float]] = []
        for segment in self.segments.values():
            if not segment.inner_route or segment.parking_route:
                continue
            projection = segment.project(x, y)
            if projection.distance_m > max_distance_m:
                continue
            if segment.length - projection.distance_on_segment < MANEUVER_YAW_LOOKAHEAD_M:
                continue
            sample_distance = min(segment.length, projection.distance_on_segment + MANEUVER_LOOKAHEAD_M)
            candidate_yaw = segment.sample(sample_distance)[2]
            delta = normalize_angle(candidate_yaw - current_yaw)
            if maneuver == "left" and MANEUVER_MIN_TURN_RAD <= delta <= MANEUVER_MAX_TURN_RAD:
                scored.append((projection, delta))
            elif maneuver == "right" and -MANEUVER_MAX_TURN_RAD <= delta <= -MANEUVER_MIN_TURN_RAD:
                scored.append((projection, delta))
            elif maneuver == "straight" and abs(delta) <= 0.35:
                scored.append((projection, delta))

        if not scored:
            return None
        if maneuver == "straight":
            return min(
                scored,
                key=lambda item: (
                    item[0].distance_m,
                    abs(item[1]),
                    -(item[0].segment.length - item[0].distance_on_segment),
                ),
            )
        return min(
            scored,
            key=lambda item: (
                item[0].distance_m,
                -abs(item[1]),
                -(item[0].segment.length - item[0].distance_on_segment),
            ),
        )

    def rejoin_nearby_maneuver(
        self,
        maneuver: str,
        source_name: str,
        x: float,
        y: float,
        current_yaw: float,
        max_distance_m: float | None = None,
    ) -> tuple[RouteProjection, float] | None:
        result = self.nearby_maneuver_projection(maneuver, x, y, current_yaw, max_distance_m)
        if result is None:
            return None
        projection, delta = result
        self.current_segment = projection.segment
        self.distance_on_segment = min(max(projection.distance_on_segment, 0.0), projection.segment.length)
        self.distance_since_branch = 0.0
        self.pending_maneuver = None
        self.pending_maneuver_source = None
        self.last_decision_text = (
            f"route/{maneuver}_nearby from {source_name}: {projection.segment.name} "
            f"delta={math.degrees(delta):+.1f}deg"
        )
        self.stopped_at_terminal = False
        return result

    def _choose_next_segment(self) -> None:
        choices = self.outgoing.get(self.current_segment.end_node, [])
        if not choices:
            self.distance_on_segment = self.current_segment.length
            self.stopped_at_terminal = True
            self.last_decision_text = f"route/end_stop: {self.current_segment.name}"
            return

        parking_choices, outer_choices, inner_choices = self._split_route_choices(choices)
        chosen = self._default_next_segment(self.current_segment, choices)

        exploration_choices = outer_choices + inner_choices
        has_controlled_branch = self._choices_have_controlled_branch(self.current_segment, choices)
        can_branch = has_controlled_branch and self.distance_since_branch >= MIN_BRANCH_DISTANCE_M
        pending_can_branch = has_controlled_branch and self.pending_maneuver is not None
        can_explore = bool(exploration_choices)
        if (can_branch or pending_can_branch) and self.pending_maneuver == PARKING_MANEUVER and parking_choices:
            source_name = self.pending_maneuver_source or "unknown"
            chosen = parking_choices[0]
            self.last_decision_text = f"route/parking from {source_name}: {chosen.name}"
            self.pending_maneuver = None
            self.pending_maneuver_source = None
            self.distance_since_branch = 0.0
            self.parking_started = True
        elif (
            (can_branch or pending_can_branch)
            and self.pending_maneuver is not None
            and self.pending_maneuver != PARKING_MANEUVER
        ):
            maneuver = self.pending_maneuver
            source_name = self.pending_maneuver_source or "unknown"
            maneuver_choices = outer_choices + inner_choices
            if maneuver_choices:
                chosen, delta = self._choose_for_maneuver(maneuver_choices, maneuver)
                self.last_decision_text = (
                    f"route/{maneuver} from {source_name}: {chosen.name} "
                    f"delta={math.degrees(delta):+.1f}deg"
                )
                self.pending_maneuver = None
                self.pending_maneuver_source = None
                self.distance_since_branch = 0.0
            else:
                self.last_decision_text = f"route/{maneuver}_blocked from {source_name}: no drivable branch"
        elif self.route_explore_enabled and can_explore:
            chosen = self._exploration_choice(exploration_choices)
            if chosen.inner_route:
                self.distance_since_branch = 0.0
            visit_count = self.segment_visit_counts.get(chosen.name, 0)
            self.last_decision_text = f"route/explore_coverage visit={visit_count}: {chosen.name}"
        else:
            if chosen is None:
                self.distance_on_segment = self.current_segment.length
                self.stopped_at_terminal = True
                self.last_decision_text = f"route/hold_inner_until_sign: {self.current_segment.name}"
                return
            self.last_decision_text = f"route/default: {chosen.name}"

        if chosen is None:
            self.distance_on_segment = self.current_segment.length
            self.stopped_at_terminal = True
            self.last_decision_text = f"route/hold_inner_until_sign: {self.current_segment.name}"
            return

        self.current_segment = chosen
        self.distance_on_segment = 0.0
        self.stopped_at_terminal = False
        self._mark_segment_visit(chosen)

    def _choose_for_maneuver(self, choices: list[TrackSegment], maneuver: str) -> tuple[TrackSegment, float]:
        current_yaw = self.current_segment.sample(max(0.0, self.current_segment.length - YAW_LOOKAHEAD_M))[2]
        scored: list[tuple[TrackSegment, float]] = []
        for choice in choices:
            sample_distance = min(choice.length, MANEUVER_LOOKAHEAD_M)
            candidate_yaw = choice.sample(sample_distance)[2]
            scored.append((choice, normalize_angle(candidate_yaw - current_yaw)))

        if maneuver == "straight":
            return min(scored, key=lambda item: abs(item[1]))

        if maneuver == "left":
            left_turns = [item for item in scored if item[1] > 0.12]
            if left_turns:
                max_delta = max(delta for _, delta in left_turns)
                tied = [item for item in left_turns if max_delta - item[1] <= MANEUVER_TIE_EPS_RAD]
                return max(tied, key=lambda item: item[0].length)

        if maneuver == "right":
            right_turns = [item for item in scored if item[1] < -0.12]
            if right_turns:
                min_delta = min(delta for _, delta in right_turns)
                tied = [item for item in right_turns if item[1] - min_delta <= MANEUVER_TIE_EPS_RAD]
                return max(tied, key=lambda item: item[0].length)

        return min(scored, key=lambda item: abs(item[1]))

    def sample(self) -> tuple[float, float, float]:
        x, y = self.current_segment.xy_at(self.distance_on_segment)
        look_x, look_y = self._xy_ahead(self._yaw_lookahead_distance())
        if math.hypot(look_x - x, look_y - y) < 1e-4:
            return self.current_segment.sample(self.distance_on_segment)
        return x, y, math.atan2(look_y - y, look_x - x)

    def _yaw_lookahead_distance(self) -> float:
        if self.current_segment.parking_route:
            return PARKING_YAW_LOOKAHEAD_M
        if self.pending_maneuver in {"left", "right", "straight"}:
            return MANEUVER_YAW_LOOKAHEAD_M
        return YAW_LOOKAHEAD_M

    def _xy_ahead(self, distance: float) -> tuple[float, float]:
        return self._xy_ahead_from(self.current_segment, self.distance_on_segment, distance)

    def _xy_ahead_from(
        self,
        segment: TrackSegment,
        distance_on_segment: float,
        distance: float,
    ) -> tuple[float, float]:
        distance_on_segment = min(max(distance_on_segment, 0.0), segment.length)
        remaining = max(0.0, distance)

        while remaining > 0.0:
            distance_left = segment.length - distance_on_segment
            if remaining <= distance_left:
                return segment.xy_at(distance_on_segment + remaining)

            next_segment = self._lookahead_next_segment(segment)
            if next_segment is None:
                return segment.xy_at(segment.length)

            remaining -= max(0.0, distance_left)
            segment = next_segment
            distance_on_segment = 0.0

        return segment.xy_at(distance_on_segment)

    def _lookahead_next_segment(self, segment: TrackSegment) -> TrackSegment | None:
        if segment is not self.current_segment:
            return self._outer_next_segment(segment)

        choices = self.outgoing.get(segment.end_node, [])
        if not choices:
            return None

        parking_choices, outer_choices, inner_choices = self._split_route_choices(choices)
        exploration_choices = outer_choices + inner_choices
        has_controlled_branch = self._choices_have_controlled_branch(segment, choices)
        can_branch = has_controlled_branch and self.distance_since_branch >= MIN_BRANCH_DISTANCE_M
        pending_can_branch = has_controlled_branch and self.pending_maneuver is not None
        if (can_branch or pending_can_branch) and self.pending_maneuver == PARKING_MANEUVER and parking_choices:
            return parking_choices[0]
        if (
            (can_branch or pending_can_branch)
            and self.pending_maneuver is not None
            and self.pending_maneuver != PARKING_MANEUVER
        ):
            maneuver_choices = outer_choices + inner_choices
            if maneuver_choices:
                chosen, _ = self._choose_for_maneuver(maneuver_choices, self.pending_maneuver)
                return chosen
        if self.route_explore_enabled and exploration_choices:
            return self._exploration_choice(exploration_choices)
        return self._default_next_segment(segment, choices)

    def current_route_maneuver_delta(self, lookahead_m: float = MANEUVER_LOOKAHEAD_M) -> float | None:
        if self.stopped_at_terminal or self.current_segment.parking_route:
            return None

        x, y, current_yaw = self.current_segment.sample(self.distance_on_segment)
        ahead_x, ahead_y = self._xy_ahead_from(
            self.current_segment,
            self.distance_on_segment,
            lookahead_m,
        )
        if math.hypot(ahead_x - x, ahead_y - y) < 1e-4:
            return None
        ahead_yaw = math.atan2(ahead_y - y, ahead_x - x)
        return normalize_angle(ahead_yaw - current_yaw)

    def current_route_matches_maneuver(self, maneuver: str) -> tuple[bool, float | None]:
        delta = self.current_route_maneuver_delta()
        if delta is None:
            return False, None
        if maneuver == "left":
            return MANEUVER_MIN_TURN_RAD <= delta <= MANEUVER_MAX_TURN_RAD, delta
        if maneuver == "right":
            return -MANEUVER_MAX_TURN_RAD <= delta <= -MANEUVER_MIN_TURN_RAD, delta
        if maneuver == "straight":
            return abs(delta) <= 0.35, delta
        return False, delta

    def _outer_next_segment(self, segment: TrackSegment) -> TrackSegment | None:
        choices = self.outgoing.get(segment.end_node, [])
        parking_choices, outer_choices, inner_choices = self._split_route_choices(choices)
        if outer_choices:
            return outer_choices[0]
        if inner_choices:
            return inner_choices[0]
        if parking_choices:
            return None
        return None

    def inner_route_is_authorized(self) -> bool:
        if self.current_segment.inner_route or self.current_segment.parking_route:
            return True
        if self.parking_started:
            return True
        return self.pending_maneuver in {"left", "right", "straight", PARKING_MANEUVER}

    def current_segment_allows_maneuver_action(self) -> bool:
        if self.stopped_at_terminal or self.current_segment.parking_route:
            return False

        return self.distance_to_next_maneuver_branch(self.maneuver_max_branch_distance_m) is not None

    def distance_to_next_maneuver_branch(self, max_distance_m: float) -> float | None:
        if self.stopped_at_terminal or self.current_segment.parking_route:
            return None

        segment = self.current_segment
        distance_to_branch = max(0.0, segment.length - self.distance_on_segment)
        visited: set[str] = set()
        max_distance_m = max(0.0, max_distance_m)

        while distance_to_branch <= max_distance_m and segment.name not in visited:
            visited.add(segment.name)
            choices = self.outgoing.get(segment.end_node, [])
            has_maneuver_branch = self._choices_have_controlled_branch(segment, choices)
            if has_maneuver_branch:
                if self.distance_since_branch + distance_to_branch >= MIN_BRANCH_DISTANCE_M:
                    return distance_to_branch
                return None

            next_segment = self._outer_next_segment(segment)
            if next_segment is None:
                return None
            distance_to_branch += next_segment.length
            segment = next_segment

        return None

    def distance_to_next_parking_branch(self, max_distance_m: float) -> float | None:
        if self.stopped_at_terminal or self.current_segment.parking_route:
            return None

        segment = self.current_segment
        distance_to_branch = max(0.0, segment.length - self.distance_on_segment)
        visited: set[str] = set()
        max_distance_m = max(0.0, max_distance_m)

        while distance_to_branch <= max_distance_m and segment.name not in visited:
            visited.add(segment.name)
            choices = self.outgoing.get(segment.end_node, [])
            parking_choices, _, _ = self._split_route_choices(choices)
            if parking_choices:
                return distance_to_branch

            next_segment = self._default_next_segment(segment, choices)
            if next_segment is None:
                return None
            distance_to_branch += next_segment.length
            segment = next_segment

        return None

    def next_maneuver_branch_distance_text(self, max_distance_m: float | None = None) -> str:
        distance = self.distance_to_next_maneuver_branch(float("inf") if max_distance_m is None else max_distance_m)
        if distance is None:
            return "none"
        return f"{distance:.2f}m"

    def current_segment_has_maneuver_branch(self) -> bool:
        choices = self.outgoing.get(self.current_segment.end_node, [])
        return self._choices_have_controlled_branch(self.current_segment, choices)

    def force_pending_maneuver_branch_at_endpoint(
        self,
        x: float,
        y: float,
        max_distance_m: float,
    ) -> bool:
        if self.pending_maneuver not in {"left", "right", "straight", PARKING_MANEUVER}:
            return False
        if self.stopped_at_terminal or self.current_segment.parking_route:
            return False
        if not self.current_segment_has_maneuver_branch():
            return False

        end_x, end_y = self.current_segment.xy_at(self.current_segment.length)
        if math.hypot(x - end_x, y - end_y) > max_distance_m:
            return False

        previous_segment_name = self.current_segment.name
        self.distance_on_segment = self.current_segment.length
        self._choose_next_segment()
        if self.current_segment.name == previous_segment_name:
            return False

        projection = self.current_segment.project(x, y)
        if projection.distance_m <= max_distance_m * 1.5:
            self.distance_on_segment = min(
                max(projection.distance_on_segment, 0.0),
                self.current_segment.length,
            )
        return True

    def outer_lap_samples(self, step_m: float):
        saved_segment = self.current_segment
        saved_distance = self.distance_on_segment
        segment = self.current_segment
        visited: set[str] = set()
        step_m = max(0.001, step_m)

        try:
            while segment.name not in visited:
                visited.add(segment.name)
                sample_count = max(1, int(math.ceil(segment.length / step_m)))
                for distance in np.linspace(0.0, segment.length, sample_count, endpoint=False):
                    self.current_segment = segment
                    self.distance_on_segment = float(distance)
                    x, y, yaw = self.sample()
                    yield segment.name, float(distance), x, y, yaw

                next_segment = self._outer_next_segment(segment)
                if next_segment is None:
                    return
                segment = next_segment
        finally:
            self.current_segment = saved_segment
            self.distance_on_segment = saved_distance


def draw_route_mask_edges(image: np.ndarray, track: DirectedTrack) -> np.ndarray:
    overlay = image.copy()
    drivable_edges = cv2.Canny((track.clearance > 0.0).astype(np.uint8) * 255, 50, 120)
    safe_edges = cv2.Canny(track.safe_mask.astype(np.uint8) * 255, 50, 120)
    overlay[drivable_edges > 0] = (0, 0, 255)
    overlay[safe_edges > 0] = (255, 0, 255)
    return overlay


def save_route_debug_image(path: str, image: np.ndarray) -> None:
    cv2.imwrite(
        path,
        cv2.resize(
            image,
            (int(MAP_TEXTURE_SIZE) * 4, int(MAP_TEXTURE_SIZE) * 4),
            interpolation=cv2.INTER_NEAREST,
        ),
    )


def write_route_debug_images(
    track: DirectedTrack,
    speed: float,
    yaw_rate_limit: float,
    yaw_accel_limit: float,
    logger,
) -> None:
    map_path = resolve_map_image_path()
    map_image = load_map_image()
    logger.info(
        f"[MAP_TRANSFORM] image={map_image.shape[1]}x{map_image.shape[0]} "
        f"path={map_path} "
        f"map_size={MODEL_SIZE_M:.3f}x{MODEL_SIZE_M:.3f}m "
        f"px_per_m_x={MAP_TEXTURE_SIZE / MODEL_SIZE_M:.3f} "
        f"px_per_m_y={MAP_TEXTURE_SIZE / MODEL_SIZE_M:.3f} "
        f"route_offset=({track.route_offset_x_m:+.3f},{track.route_offset_y_m:+.3f})m "
        f"map_visual_offset=({MAP_VISUAL_OFFSET_X_M:+.3f},{MAP_VISUAL_OFFSET_Y_M:+.3f})m"
    )

    mask_overlay = draw_route_mask_edges(map_image, track)
    centerline_overlay = mask_overlay.copy()
    footprint_overlay = mask_overlay.copy()
    drivable_mask = track.clearance > 0.0
    centerline_points: list[list[int]] = []
    footprint_failures: list[tuple[int, str, float, float, float, float, float]] = []
    center_off_road = 0
    center_outside_safe = 0
    min_footprint_clearance = float("inf")
    debug_yaw: float | None = None
    debug_yaw_rate = 0.0
    debug_dt = max(GUIDED_UPDATE_PERIOD, FOOTPRINT_DEBUG_SAMPLE_M / max(speed, 1e-6))

    front_m = FOOTPRINT_FRONT_M + FOOTPRINT_MARGIN_M
    rear_m = FOOTPRINT_REAR_M + FOOTPRINT_MARGIN_M
    half_width_m = FOOTPRINT_HALF_WIDTH_M + FOOTPRINT_MARGIN_M

    for index, (segment_name, _, x, y, target_yaw) in enumerate(track.outer_lap_samples(FOOTPRINT_DEBUG_SAMPLE_M)):
        yaw, debug_yaw_rate = limit_guided_yaw(
            target_yaw,
            debug_dt,
            debug_yaw,
            debug_yaw_rate,
            yaw_rate_limit,
            yaw_accel_limit,
        )
        debug_yaw = yaw

        map_x, map_y = model_point_to_map(x, y, track.route_offset_x_m, track.route_offset_y_m)
        pixel_x = int(round(map_x))
        pixel_y = int(round(map_y))
        centerline_points.append([pixel_x, pixel_y])
        center_in_image = 0 <= pixel_x < track.safe_mask.shape[1] and 0 <= pixel_y < track.safe_mask.shape[0]
        if not center_in_image or not drivable_mask[pixel_y, pixel_x]:
            center_off_road += 1
        if not center_in_image or not track.safe_mask[pixel_y, pixel_x]:
            center_outside_safe += 1

        footprint_check = check_road_footprint(
            track.clearance,
            x,
            y,
            yaw,
            front_m,
            rear_m,
            half_width_m,
            track.route_offset_x_m,
            track.route_offset_y_m,
        )
        footprint_px = footprint_check.corners_px
        footprint_bad = not footprint_check.on_road
        clearance = footprint_check.min_clearance_px
        min_footprint_clearance = min(min_footprint_clearance, clearance)

        color = (0, 0, 255) if footprint_bad else (0, 255, 255)
        if footprint_bad or index % 16 == 0:
            cv2.polylines(footprint_overlay, [footprint_px], True, color, 1, cv2.LINE_AA)
        if footprint_bad:
            footprint_failures.append((index, segment_name, x, y, yaw, target_yaw, clearance))

    if centerline_points:
        centerline_poly = np.array(centerline_points, dtype=np.int32)
        cv2.polylines(centerline_overlay, [centerline_poly], False, (255, 80, 0), 2, cv2.LINE_AA)
        cv2.polylines(footprint_overlay, [centerline_poly], False, (255, 80, 0), 1, cv2.LINE_AA)

    save_route_debug_image(ROUTE_MASK_DEBUG_PATH, mask_overlay)
    save_route_debug_image(ROUTE_CENTERLINE_DEBUG_PATH, centerline_overlay)
    save_route_debug_image(ROUTE_FOOTPRINT_DEBUG_PATH, footprint_overlay)

    logger.info(
        f"[FOOTPRINT_CHECK] samples={len(centerline_points)} center_off_road={center_off_road} "
        f"center_outside_safe={center_outside_safe} failures={len(footprint_failures)} "
        f"min_footprint_clearance={min_footprint_clearance:.2f}px "
        f"footprint=front:{front_m:.3f}m rear:{rear_m:.3f}m half_width:{half_width_m:.3f}m"
    )
    logger.info(
        f"[FOOTPRINT_CHECK] overlays={ROUTE_MASK_DEBUG_PATH}, "
        f"{ROUTE_CENTERLINE_DEBUG_PATH}, {ROUTE_FOOTPRINT_DEBUG_PATH}"
    )
    for index, segment_name, x, y, yaw, target_yaw, clearance in footprint_failures[:12]:
        logger.warn(
            f"[FOOTPRINT_FAIL] idx={index} segment={segment_name} x={x:+.3f} y={y:+.3f} "
            f"yaw={yaw:+.3f} target_yaw={target_yaw:+.3f} clearance={clearance:.2f}px"
        )


class GazeboTrackCar(Node):
    def __init__(self) -> None:
        super().__init__("gazebo_track_car")
        self.declare_parameter("entity_name", "track_preview_car")
        self.declare_parameter("speed", 0.28)
        self.declare_parameter("z", 0.02)
        self.declare_parameter("yaw_offset", 0.0)
        self.declare_parameter("reverse_direction", True)
        self.declare_parameter("branch_probability", 0.0)
        self.declare_parameter("route_seed", 7)
        self.declare_parameter("route_explore_enabled", False)
        self.declare_parameter("start_distance_m", 0.0)
        self.declare_parameter("clearance_px", DEFAULT_CLEARANCE_PX)
        self.declare_parameter("route_offset_x_m", DEFAULT_ROUTE_OFFSET_X_M)
        self.declare_parameter("route_offset_y_m", DEFAULT_ROUTE_OFFSET_Y_M)
        self.declare_parameter("sign_detection_enabled", True)
        self.declare_parameter("sign_detection_max_distance_m", 3.4)
        self.declare_parameter("sign_detection_fov_rad", 1.95)
        self.declare_parameter("sign_detection_min_forward_m", 0.08)
        self.declare_parameter("speed_limit_mps", 0.14)
        self.declare_parameter("traffic_light_state", "green")
        self.declare_parameter("traffic_light_cycle_enabled", True)
        self.declare_parameter("traffic_light_green_duration_s", TRAFFIC_LIGHT_DEFAULT_DURATIONS_S["green"])
        self.declare_parameter("traffic_light_yellow_duration_s", TRAFFIC_LIGHT_DEFAULT_DURATIONS_S["yellow"])
        self.declare_parameter("traffic_light_red_duration_s", TRAFFIC_LIGHT_DEFAULT_DURATIONS_S["red"])
        self.declare_parameter("traffic_light_trigger_distance_m", 1.70)
        self.declare_parameter("traffic_light_stop_distance_m", TRAFFIC_LIGHT_DEFAULT_STOP_DISTANCE_M)
        self.declare_parameter("traffic_light_min_stop_s", TRAFFIC_LIGHT_DEFAULT_MIN_STOP_S)
        self.declare_parameter("traffic_light_visual_update_enabled", True)
        self.declare_parameter("traffic_light_light_service", "/set_light_properties")
        self.declare_parameter("traffic_light_camera_enabled", True)
        self.declare_parameter("traffic_light_camera_topic", "/track_preview_car/camera/image_raw")
        self.declare_parameter("traffic_light_camera_roi_top_ratio", 0.02)
        self.declare_parameter("traffic_light_camera_roi_bottom_ratio", 0.72)
        self.declare_parameter("traffic_light_camera_min_area_px", 18)
        self.declare_parameter("traffic_light_camera_max_area_ratio", 0.20)
        self.declare_parameter("traffic_light_camera_min_score", 35.0)
        self.declare_parameter("traffic_light_camera_min_fill_ratio", 0.42)
        self.declare_parameter("traffic_light_camera_slow_min_area_px", 45)
        self.declare_parameter("traffic_light_camera_stop_min_area_px", 160)
        self.declare_parameter("traffic_light_camera_slow_min_height_px", 6)
        self.declare_parameter("traffic_light_camera_stop_min_height_px", 12)
        self.declare_parameter("traffic_light_camera_timeout_s", 0.25)
        self.declare_parameter("camera_sign_roi_top_ratio", 0.08)
        self.declare_parameter("camera_sign_roi_bottom_ratio", 0.82)
        self.declare_parameter("camera_sign_min_colored_area_px", 70)
        self.declare_parameter("camera_sign_min_bbox_height_px", 16)
        self.declare_parameter("camera_sign_min_score", 0.32)
        self.declare_parameter("camera_sign_stable_frames", 2)
        self.declare_parameter("camera_sign_action_min_area_px", 260)
        self.declare_parameter("camera_sign_parking_min_area_px", 900)
        self.declare_parameter("parking_auto_route_enabled", False)
        self.declare_parameter("camera_sign_timeout_s", 0.45)
        self.declare_parameter("drive_wheels", True)
        self.declare_parameter("wheel_control_lookahead_m", WHEEL_CONTROL_LOOKAHEAD_M)
        self.declare_parameter("wheel_control_parking_lookahead_m", WHEEL_CONTROL_PARKING_LOOKAHEAD_M)
        self.declare_parameter("wheel_control_heading_gain", WHEEL_CONTROL_HEADING_GAIN)
        self.declare_parameter("wheel_control_max_angular_rps", WHEEL_CONTROL_MAX_ANGULAR_RPS)
        self.declare_parameter("wheel_control_turn_slowdown_rad", WHEEL_CONTROL_TURN_SLOWDOWN_RAD)
        self.declare_parameter("wheel_control_min_speed_scale", WHEEL_CONTROL_MIN_SPEED_SCALE)
        self.declare_parameter("wheel_control_terminal_slowdown_m", WHEEL_CONTROL_TERMINAL_SLOWDOWN_M)
        self.declare_parameter("wheel_control_rotate_heading_error_rad", WHEEL_CONTROL_ROTATE_HEADING_ERROR_RAD)
        self.declare_parameter("wheel_control_nearest_route_enabled", True)
        self.declare_parameter("wheel_control_nearest_route_max_distance_m", WHEEL_CONTROL_NEAREST_ROUTE_MAX_DISTANCE_M)
        self.declare_parameter("guided_yaw_rate_limit", 0.0)
        self.declare_parameter("guided_yaw_accel_limit", 0.0)
        self.declare_parameter("telemetry_period", 2.0)
        self.declare_parameter("route_trace_enabled", True)
        self.declare_parameter("route_trace_period", 0.10)
        self.declare_parameter("jitter_log_period", 1.0)
        self.declare_parameter("motion_log_verbose", False)
        self.declare_parameter("jitter_xy_step_delta_m", 0.006)
        self.declare_parameter("jitter_command_error_m", 0.05)
        self.declare_parameter("jitter_z_delta_m", 0.0005)
        self.declare_parameter("jitter_tilt_rad", 0.003)
        self.declare_parameter("jitter_speed_delta_mps", 0.03)
        self.declare_parameter("jitter_yaw_step_rad", 0.025)
        self.declare_parameter("jitter_yaw_rate_delta_rps", 0.40)
        self.declare_parameter("route_rejoin_enabled", True)
        self.declare_parameter("route_rejoin_outer_only", True)
        self.declare_parameter("route_rejoin_command_error_m", ROUTE_REJOIN_COMMAND_ERROR_M)
        self.declare_parameter("route_rejoin_max_route_distance_m", ROUTE_REJOIN_MAX_ROUTE_DISTANCE_M)
        self.declare_parameter("route_rejoin_cooldown_s", ROUTE_REJOIN_COOLDOWN_S)
        self.declare_parameter("route_progress_sync_enabled", True)
        self.declare_parameter("route_progress_sync_command_error_m", ROUTE_PROGRESS_SYNC_COMMAND_ERROR_M)
        self.declare_parameter("route_progress_sync_max_route_distance_m", ROUTE_PROGRESS_SYNC_MAX_ROUTE_DISTANCE_M)
        self.declare_parameter("route_progress_sync_max_yaw_error_rad", ROUTE_PROGRESS_SYNC_MAX_YAW_ERROR_RAD)
        self.declare_parameter("route_progress_sync_min_interval_s", ROUTE_PROGRESS_SYNC_MIN_INTERVAL_S)
        self.declare_parameter("route_progress_sync_log_period_s", ROUTE_PROGRESS_SYNC_LOG_PERIOD_S)
        self.declare_parameter("initial_pose_reset_enabled", True)
        self.declare_parameter("initial_pose_reset_position_tolerance_m", INITIAL_POSE_RESET_POSITION_TOLERANCE_M)
        self.declare_parameter("initial_pose_reset_yaw_tolerance_rad", INITIAL_POSE_RESET_YAW_TOLERANCE_RAD)

        self.entity_name = self.get_parameter("entity_name").get_parameter_value().string_value
        self.speed = float(self.get_parameter("speed").get_parameter_value().double_value)
        self.z = float(self.get_parameter("z").get_parameter_value().double_value)
        self.yaw_offset = float(self.get_parameter("yaw_offset").get_parameter_value().double_value)
        self.reverse_direction = bool(self.get_parameter("reverse_direction").get_parameter_value().bool_value)
        self.branch_probability = float(self.get_parameter("branch_probability").get_parameter_value().double_value)
        self.route_seed = int(self.get_parameter("route_seed").get_parameter_value().integer_value)
        self.route_explore_enabled = bool(
            self.get_parameter("route_explore_enabled").get_parameter_value().bool_value
        )
        self.start_distance_m = max(
            0.0,
            float(self.get_parameter("start_distance_m").get_parameter_value().double_value),
        )
        self.clearance_px = int(self.get_parameter("clearance_px").get_parameter_value().integer_value)
        self.route_offset_x_m = float(self.get_parameter("route_offset_x_m").get_parameter_value().double_value)
        self.route_offset_y_m = float(self.get_parameter("route_offset_y_m").get_parameter_value().double_value)
        self.sign_detection_enabled = bool(
            self.get_parameter("sign_detection_enabled").get_parameter_value().bool_value
        )
        self.sign_detection_max_distance_m = max(
            0.1,
            float(self.get_parameter("sign_detection_max_distance_m").get_parameter_value().double_value),
        )
        self.sign_detection_fov_rad = max(
            0.1,
            float(self.get_parameter("sign_detection_fov_rad").get_parameter_value().double_value),
        )
        self.sign_detection_min_forward_m = max(
            0.0,
            float(self.get_parameter("sign_detection_min_forward_m").get_parameter_value().double_value),
        )
        self.speed_limit_mps = max(
            0.01,
            float(self.get_parameter("speed_limit_mps").get_parameter_value().double_value),
        )
        self.traffic_light_state = (
            self.get_parameter("traffic_light_state").get_parameter_value().string_value.strip().lower()
        )
        if self.traffic_light_state not in TRAFFIC_LIGHT_VALID_STATES:
            self.get_logger().warn(
                f"Unknown traffic_light_state='{self.traffic_light_state}', falling back to green."
            )
            self.traffic_light_state = "green"
        self.traffic_light_cycle_enabled = bool(
            self.get_parameter("traffic_light_cycle_enabled").get_parameter_value().bool_value
        )
        self.traffic_light_durations_s = {
            "green": max(
                0.1,
                float(self.get_parameter("traffic_light_green_duration_s").get_parameter_value().double_value),
            ),
            "yellow": max(
                0.1,
                float(self.get_parameter("traffic_light_yellow_duration_s").get_parameter_value().double_value),
            ),
            "red": max(
                0.1,
                float(self.get_parameter("traffic_light_red_duration_s").get_parameter_value().double_value),
            ),
        }
        self.traffic_light_trigger_distance_m = max(
            0.1,
            float(self.get_parameter("traffic_light_trigger_distance_m").get_parameter_value().double_value),
        )
        self.traffic_light_stop_distance_m = min(
            self.traffic_light_trigger_distance_m,
            max(
                0.1,
                float(self.get_parameter("traffic_light_stop_distance_m").get_parameter_value().double_value),
            ),
        )
        self.traffic_light_min_stop_ns = int(
            max(
                0.0,
                float(self.get_parameter("traffic_light_min_stop_s").get_parameter_value().double_value),
            )
            * 1e9
        )
        self.traffic_light_visual_update_enabled = bool(
            self.get_parameter("traffic_light_visual_update_enabled").get_parameter_value().bool_value
        )
        self.traffic_light_light_service = (
            self.get_parameter("traffic_light_light_service").get_parameter_value().string_value.strip()
            or "/set_light_properties"
        )
        self.traffic_light_camera_enabled = bool(
            self.get_parameter("traffic_light_camera_enabled").get_parameter_value().bool_value
        )
        self.traffic_light_camera_topic = (
            self.get_parameter("traffic_light_camera_topic").get_parameter_value().string_value.strip()
            or "/track_preview_car/camera/image_raw"
        )
        self.traffic_light_camera_roi_top_ratio = min(
            max(float(self.get_parameter("traffic_light_camera_roi_top_ratio").get_parameter_value().double_value), 0.0),
            0.95,
        )
        self.traffic_light_camera_roi_bottom_ratio = min(
            max(
                float(self.get_parameter("traffic_light_camera_roi_bottom_ratio").get_parameter_value().double_value),
                self.traffic_light_camera_roi_top_ratio + 0.02,
            ),
            1.0,
        )
        self.traffic_light_camera_min_area_px = max(
            1,
            int(self.get_parameter("traffic_light_camera_min_area_px").get_parameter_value().integer_value),
        )
        self.traffic_light_camera_max_area_ratio = max(
            0.001,
            float(self.get_parameter("traffic_light_camera_max_area_ratio").get_parameter_value().double_value),
        )
        self.traffic_light_camera_min_score = max(
            1.0,
            float(self.get_parameter("traffic_light_camera_min_score").get_parameter_value().double_value),
        )
        self.traffic_light_camera_min_fill_ratio = min(
            max(
                float(self.get_parameter("traffic_light_camera_min_fill_ratio").get_parameter_value().double_value),
                0.0,
            ),
            1.0,
        )
        self.traffic_light_camera_slow_min_area_px = max(
            1,
            int(self.get_parameter("traffic_light_camera_slow_min_area_px").get_parameter_value().integer_value),
        )
        self.traffic_light_camera_stop_min_area_px = max(
            self.traffic_light_camera_slow_min_area_px,
            int(self.get_parameter("traffic_light_camera_stop_min_area_px").get_parameter_value().integer_value),
        )
        self.traffic_light_camera_slow_min_height_px = max(
            1,
            int(self.get_parameter("traffic_light_camera_slow_min_height_px").get_parameter_value().integer_value),
        )
        self.traffic_light_camera_stop_min_height_px = max(
            self.traffic_light_camera_slow_min_height_px,
            int(self.get_parameter("traffic_light_camera_stop_min_height_px").get_parameter_value().integer_value),
        )
        self.traffic_light_camera_timeout_ns = int(
            max(
                0.1,
                float(self.get_parameter("traffic_light_camera_timeout_s").get_parameter_value().double_value),
            )
            * 1e9
        )
        self.camera_sign_roi_top_ratio = min(
            max(float(self.get_parameter("camera_sign_roi_top_ratio").get_parameter_value().double_value), 0.0),
            0.95,
        )
        self.camera_sign_roi_bottom_ratio = min(
            max(
                float(self.get_parameter("camera_sign_roi_bottom_ratio").get_parameter_value().double_value),
                self.camera_sign_roi_top_ratio + 0.02,
            ),
            1.0,
        )
        self.camera_sign_min_colored_area_px = max(
            1,
            int(self.get_parameter("camera_sign_min_colored_area_px").get_parameter_value().integer_value),
        )
        self.camera_sign_min_bbox_height_px = max(
            1,
            int(self.get_parameter("camera_sign_min_bbox_height_px").get_parameter_value().integer_value),
        )
        self.camera_sign_min_score = max(
            0.0,
            float(self.get_parameter("camera_sign_min_score").get_parameter_value().double_value),
        )
        self.camera_sign_stable_frames = max(
            1,
            int(self.get_parameter("camera_sign_stable_frames").get_parameter_value().integer_value),
        )
        self.camera_sign_action_min_area_px = max(
            1,
            int(self.get_parameter("camera_sign_action_min_area_px").get_parameter_value().integer_value),
        )
        self.camera_sign_parking_min_area_px = max(
            self.camera_sign_action_min_area_px,
            int(self.get_parameter("camera_sign_parking_min_area_px").get_parameter_value().integer_value),
        )
        self.parking_auto_route_enabled = bool(
            self.get_parameter("parking_auto_route_enabled").get_parameter_value().bool_value
        )
        self.camera_sign_timeout_ns = int(
            max(
                0.1,
                float(self.get_parameter("camera_sign_timeout_s").get_parameter_value().double_value),
            )
            * 1e9
        )
        self.drive_wheels = bool(self.get_parameter("drive_wheels").get_parameter_value().bool_value)
        self.wheel_control_lookahead_m = max(
            0.05,
            float(self.get_parameter("wheel_control_lookahead_m").get_parameter_value().double_value),
        )
        self.wheel_control_parking_lookahead_m = max(
            0.05,
            float(self.get_parameter("wheel_control_parking_lookahead_m").get_parameter_value().double_value),
        )
        self.wheel_control_heading_gain = max(
            0.1,
            float(self.get_parameter("wheel_control_heading_gain").get_parameter_value().double_value),
        )
        self.wheel_control_max_angular_rps = max(
            0.1,
            float(self.get_parameter("wheel_control_max_angular_rps").get_parameter_value().double_value),
        )
        self.wheel_control_turn_slowdown_rad = max(
            0.1,
            float(self.get_parameter("wheel_control_turn_slowdown_rad").get_parameter_value().double_value),
        )
        self.wheel_control_min_speed_scale = min(
            1.0,
            max(
                0.05,
                float(self.get_parameter("wheel_control_min_speed_scale").get_parameter_value().double_value),
            ),
        )
        self.wheel_control_terminal_slowdown_m = max(
            0.05,
            float(self.get_parameter("wheel_control_terminal_slowdown_m").get_parameter_value().double_value),
        )
        self.wheel_control_rotate_heading_error_rad = max(
            0.1,
            float(
                self.get_parameter("wheel_control_rotate_heading_error_rad")
                .get_parameter_value()
                .double_value
            ),
        )
        self.wheel_control_nearest_route_enabled = bool(
            self.get_parameter("wheel_control_nearest_route_enabled").get_parameter_value().bool_value
        )
        self.wheel_control_nearest_route_max_distance_m = max(
            0.05,
            float(
                self.get_parameter("wheel_control_nearest_route_max_distance_m")
                .get_parameter_value()
                .double_value
            ),
        )
        self.guided_yaw_rate_limit = max(
            0.0, float(self.get_parameter("guided_yaw_rate_limit").get_parameter_value().double_value)
        )
        self.guided_yaw_accel_limit = max(
            0.0, float(self.get_parameter("guided_yaw_accel_limit").get_parameter_value().double_value)
        )
        self.telemetry_period_ns = int(
            max(0.05, float(self.get_parameter("telemetry_period").get_parameter_value().double_value)) * 1e9
        )
        self.route_trace_enabled = bool(
            self.get_parameter("route_trace_enabled").get_parameter_value().bool_value
        )
        self.route_trace_period_ns = int(
            max(0.02, float(self.get_parameter("route_trace_period").get_parameter_value().double_value)) * 1e9
        )
        self.jitter_log_period_ns = int(
            max(0.05, float(self.get_parameter("jitter_log_period").get_parameter_value().double_value)) * 1e9
        )
        self.motion_log_verbose = bool(
            self.get_parameter("motion_log_verbose").get_parameter_value().bool_value
        )
        self.jitter_xy_step_delta_m = float(
            self.get_parameter("jitter_xy_step_delta_m").get_parameter_value().double_value
        )
        self.jitter_command_error_m = float(
            self.get_parameter("jitter_command_error_m").get_parameter_value().double_value
        )
        self.jitter_z_delta_m = float(self.get_parameter("jitter_z_delta_m").get_parameter_value().double_value)
        self.jitter_tilt_rad = float(self.get_parameter("jitter_tilt_rad").get_parameter_value().double_value)
        self.jitter_speed_delta_mps = float(
            self.get_parameter("jitter_speed_delta_mps").get_parameter_value().double_value
        )
        self.jitter_yaw_step_rad = float(
            self.get_parameter("jitter_yaw_step_rad").get_parameter_value().double_value
        )
        self.jitter_yaw_rate_delta_rps = float(
            self.get_parameter("jitter_yaw_rate_delta_rps").get_parameter_value().double_value
        )
        self.route_rejoin_enabled = bool(
            self.get_parameter("route_rejoin_enabled").get_parameter_value().bool_value
        )
        self.route_rejoin_outer_only = bool(
            self.get_parameter("route_rejoin_outer_only").get_parameter_value().bool_value
        )
        self.route_rejoin_command_error_m = max(
            0.0,
            float(self.get_parameter("route_rejoin_command_error_m").get_parameter_value().double_value),
        )
        self.route_rejoin_max_route_distance_m = max(
            0.0,
            float(self.get_parameter("route_rejoin_max_route_distance_m").get_parameter_value().double_value),
        )
        self.route_rejoin_cooldown_ns = int(
            max(0.0, float(self.get_parameter("route_rejoin_cooldown_s").get_parameter_value().double_value)) * 1e9
        )
        self.route_progress_sync_enabled = bool(
            self.get_parameter("route_progress_sync_enabled").get_parameter_value().bool_value
        )
        self.route_progress_sync_command_error_m = max(
            0.0,
            float(self.get_parameter("route_progress_sync_command_error_m").get_parameter_value().double_value),
        )
        self.route_progress_sync_max_route_distance_m = max(
            0.0,
            float(self.get_parameter("route_progress_sync_max_route_distance_m").get_parameter_value().double_value),
        )
        self.route_progress_sync_max_yaw_error_rad = max(
            0.0,
            float(self.get_parameter("route_progress_sync_max_yaw_error_rad").get_parameter_value().double_value),
        )
        self.route_progress_sync_min_interval_ns = int(
            max(
                0.0,
                float(self.get_parameter("route_progress_sync_min_interval_s").get_parameter_value().double_value),
            )
            * 1e9
        )
        self.route_progress_sync_log_period_ns = int(
            max(
                0.0,
                float(self.get_parameter("route_progress_sync_log_period_s").get_parameter_value().double_value),
            )
            * 1e9
        )
        self.initial_pose_reset_enabled = bool(
            self.get_parameter("initial_pose_reset_enabled").get_parameter_value().bool_value
        )
        self.initial_pose_reset_position_tolerance_m = max(
            0.0,
            float(
                self.get_parameter("initial_pose_reset_position_tolerance_m")
                .get_parameter_value()
                .double_value
            ),
        )
        self.initial_pose_reset_yaw_tolerance_rad = max(
            0.0,
            float(
                self.get_parameter("initial_pose_reset_yaw_tolerance_rad")
                .get_parameter_value()
                .double_value
            ),
        )
        self.footprint_front_m = FOOTPRINT_FRONT_M + FOOTPRINT_MARGIN_M
        self.footprint_rear_m = FOOTPRINT_REAR_M + FOOTPRINT_MARGIN_M
        self.footprint_half_width_m = FOOTPRINT_HALF_WIDTH_M + FOOTPRINT_MARGIN_M

        self.track = DirectedTrack(
            self.reverse_direction,
            self.branch_probability,
            self.route_seed,
            self.route_explore_enabled,
            self.clearance_px,
            self.route_offset_x_m,
            self.route_offset_y_m,
        )
        self.traffic_light_config = self.track.route_config.traffic_lights
        self.traffic_light_scene_model_names = self.traffic_light_config.scene_model_names
        self.traffic_light_scene_default_poses = self.traffic_light_config.default_poses
        self.traffic_light_scene_release_fov_rad = self.traffic_light_config.release_fov_rad
        self.traffic_light_indicator_hidden_pose = self.traffic_light_config.indicator_hidden_pose
        self.traffic_light_indicator_local_x_m, self.traffic_light_indicator_local_y_m = (
            self.traffic_light_config.indicator_local_xy_m
        )
        self.traffic_light_indicator_models = self.traffic_light_config.indicator_models
        if self.start_distance_m > 0.0:
            self.track.advance(self.start_distance_m)
        write_route_debug_images(
            self.track,
            self.speed,
            self.guided_yaw_rate_limit,
            self.guided_yaw_accel_limit,
            self.get_logger(),
        )
        self.visual_state_client = self.create_client(SetEntityState, "/set_entity_state")
        self.route_rejoin_state_client = self.create_client(SetEntityState, "/set_entity_state")
        self.light_client = self.create_client(SetLightProperties, self.traffic_light_light_service)
        self.cmd_pub = self.create_publisher(Twist, "/track_preview_car/cmd_vel", 10)
        self.route_trace_pub = self.create_publisher(String, "/gazebo_track_car/route_trace", 10)
        self.camera_traffic_light_pub = self.create_publisher(
            String,
            "/gazebo_track_car/camera_traffic_light_state",
            10,
        )
        self.camera_sign_pub = self.create_publisher(
            String,
            "/gazebo_track_car/camera_sign_detection",
            10,
        )
        self.bridge = CvBridge()
        self.camera_sign_templates = self.load_camera_sign_templates()
        self.camera_image_sub = None
        if self.traffic_light_camera_enabled:
            self.camera_image_sub = self.create_subscription(
                Image,
                self.traffic_light_camera_topic,
                self.handle_camera_image,
                10,
            )
        self.model_states_sub = self.create_subscription(ModelStates, "/model_states", self.handle_model_states, 10)
        self.last_update_time = self.get_clock().now()
        self.wheel_pose_wait_logged = False
        self.wheel_disabled_logged = False
        self.model_missing_logged = False
        self.last_command_sample: MotionSample | None = None
        self.last_actual_sample: MotionSample | None = None
        self.last_actual_command_sample: MotionSample | None = None
        self.last_actual_footprint_check: FootprintRoadCheck | None = None
        self.last_actual_xy_step: float | None = None
        self.last_telemetry_log_ns = 0
        self.last_jitter_log_ns = 0
        self.last_actual_footprint_log_ns = 0
        self.last_route_trace_ns = 0
        self.last_route_trace_signature: tuple[str, str, bool] | None = None
        self.suppressed_jitter_count = 0
        self.last_route_rejoin_ns = 0
        self.last_route_progress_sync_ns = 0
        self.last_route_progress_sync_log_ns = 0
        self.route_rejoin_wait_logged = False
        self.route_rejoin_future = None
        self.initial_pose_reset_done = False
        self.initial_pose_reset_pending = False
        self.initial_pose_reset_future = None
        self.initial_pose_reset_wait_logged = False
        self.initial_pose_reset_settle_until_ns = 0
        self.last_cmd_linear_mps = 0.0
        self.last_cmd_angular_rps = 0.0
        self.sent_cmd_count = 0
        self.guided_pose_synced = False
        self.skip_route_advance_once = False
        self.startup_sync_logged = False
        self.last_segment_name = self.track.current_segment.name
        self.last_visible_instruction_name: str | None = None
        self.last_applied_instruction_name: str | None = None
        self.last_applied_instruction_key: tuple[str, str] | None = None
        self.active_speed_mps = self.speed
        self.pending_speed_limit_name: str | None = None
        self.active_speed_limit_name: str | None = None
        self.speed_limit_release_segment_name: str | None = None
        self.speed_limit_release_distance_m = 0.0
        self.applied_speed_limit_names: set[str] = set()
        self.camera_parking_armed_name: str | None = None
        self.stop_for_traffic_light = False
        self.slow_for_traffic_light = False
        self.last_logged_traffic_light_state: str | None = None
        self.last_visual_traffic_light_state: str | None = None
        self.last_indicator_traffic_light_state: str | None = None
        self.last_indicator_traffic_light_scene_signature: (
            tuple[tuple[str, float, float, float, float], ...] | None
        ) = None
        self.last_glow_traffic_light_scene_signature: (
            tuple[tuple[str, float, float, float, float], ...] | None
        ) = None
        self.traffic_light_scene_poses = dict(self.traffic_light_scene_default_poses)
        self.camera_sign_scene_poses: dict[str, SceneSignPose] = {}
        self.parking_sign_scene_poses: dict[str, SceneSignPose] = {}
        self.light_wait_logged = False
        self.indicator_wait_logged = False
        self.camera_traffic_light_state: str | None = None
        self.camera_traffic_light_score = 0.0
        self.camera_traffic_light_area_px = 0
        self.camera_traffic_light_center: tuple[int, int] | None = None
        self.camera_traffic_light_size: tuple[int, int] | None = None
        self.camera_traffic_light_fill_ratio = 0.0
        self.last_camera_traffic_light_state: str | None = None
        self.last_camera_detection_ns = 0
        self.tracked_traffic_light_name: str | None = None
        self.traffic_light_control_target: TrafficLightSceneTarget | None = None
        self.traffic_light_stop_latched = False
        self.traffic_light_stop_latched_since_ns = 0
        self.traffic_light_stop_latched_name: str | None = None
        self.last_traffic_light_route_guard_signature: tuple[str, str | None] | None = None
        self.camera_instruction_detection: CameraInstructionDetection | None = None
        self.last_camera_instruction_ns = 0
        self.camera_instruction_stable_name: str | None = None
        self.camera_instruction_stable_count = 0
        self.last_camera_instruction_log_name: str | None = None
        self.last_camera_parking_confusion_guard_signature: tuple[str, str] | None = None
        self.last_camera_parking_maneuver_guard_signature: tuple[str, str] | None = None
        self.last_camera_turn_confusion_guard_signature: tuple[str, str, str] | None = None
        self.last_camera_maneuver_conflict_guard_signature: tuple[str, str, str, str] | None = None
        self.last_camera_scene_context_guard_signature: tuple[str, tuple[str, ...]] | None = None
        self.parking_stop_logged = False

        self.create_timer(GUIDED_UPDATE_PERIOD, self.update_car)
        direction_text = "reversed to match map arrows" if self.reverse_direction else "texture point order"
        self.get_logger().info(
            f"SAFE_ROUTE_VERSION={SAFE_ROUTE_VERSION}; direction={direction_text}; "
            f"clearance={self.clearance_px}px; drive_wheels={self.drive_wheels}; "
            f"route_offset=({self.route_offset_x_m:+.3f},{self.route_offset_y_m:+.3f})m; "
            f"start_distance={self.start_distance_m:.3f}m; "
            f"route_explore={self.route_explore_enabled} branch_probability={self.branch_probability:.2f}; "
            f"sign_detection={'camera' if self.sign_detection_enabled else 'off'} "
            f"roi={self.camera_sign_roi_top_ratio:.2f}-{self.camera_sign_roi_bottom_ratio:.2f}; "
            f"map_visual_offset=({MAP_VISUAL_OFFSET_X_M:+.3f},{MAP_VISUAL_OFFSET_Y_M:+.3f})m; "
            f"base_speed={self.speed:.2f}m/s speed_limit={self.speed_limit_mps:.2f}m/s; "
            f"traffic_light={'cycle' if self.traffic_light_cycle_enabled else 'fixed'} "
            f"state={self.traffic_light_state} "
            f"green={self.traffic_light_durations_s['green']:.1f}s "
            f"yellow={self.traffic_light_durations_s['yellow']:.1f}s "
            f"red={self.traffic_light_durations_s['red']:.1f}s "
            f"trigger_distance={self.traffic_light_trigger_distance_m:.2f}m "
            f"stop_distance={self.traffic_light_stop_distance_m:.2f}m "
            f"min_stop={self.traffic_light_min_stop_ns * 1e-9:.1f}s "
            f"camera={'on' if self.traffic_light_camera_enabled else 'off'} "
            f"topic={self.traffic_light_camera_topic} "
            f"fill>={self.traffic_light_camera_min_fill_ratio:.2f} "
            f"slow_area>={self.traffic_light_camera_slow_min_area_px}px "
            f"stop_area>={self.traffic_light_camera_stop_min_area_px}px "
            "distance_gate=on; "
            f"camera_sign_templates={len(self.camera_sign_templates)} "
            f"sign_score>={self.camera_sign_min_score:.2f} "
            f"sign_stable={self.camera_sign_stable_frames}; "
            f"parking_auto_route={self.parking_auto_route_enabled}; "
            f"wheel_controller lookahead={self.wheel_control_lookahead_m:.2f}m "
            f"parking_lookahead={self.wheel_control_parking_lookahead_m:.2f}m "
            f"heading_gain={self.wheel_control_heading_gain:.2f} "
            f"max_wz={self.wheel_control_max_angular_rps:.2f}rad/s "
            f"rotate_heading>{self.wheel_control_rotate_heading_error_rad:.2f}rad; "
            f"nearest_route={self.wheel_control_nearest_route_enabled} "
            f"nearest_dist<={self.wheel_control_nearest_route_max_distance_m:.2f}m; "
            f"yaw_limits={self.guided_yaw_rate_limit:.2f}rad/s/{self.guided_yaw_accel_limit:.2f}rad/s2; "
            f"route_rejoin={self.route_rejoin_enabled} outer_only={self.route_rejoin_outer_only} "
            f"cmd_error>{self.route_rejoin_command_error_m:.2f}m "
            f"max_route_dist={self.route_rejoin_max_route_distance_m:.2f}m; "
            f"progress_sync={self.route_progress_sync_enabled} "
            f"cmd_error>{self.route_progress_sync_command_error_m:.2f}m "
            f"route_dist<={self.route_progress_sync_max_route_distance_m:.2f}m "
            f"yaw<={self.route_progress_sync_max_yaw_error_rad:.2f}rad; "
            f"initial_pose_reset={self.initial_pose_reset_enabled} "
            f"tol={self.initial_pose_reset_position_tolerance_m:.2f}m/"
            f"{self.initial_pose_reset_yaw_tolerance_rad:.2f}rad; "
            f"segments={self.track.segment_count}; route_points={self.track.route_point_count}."
        )
        self.get_logger().info(
            f"Motion log uses /model_states and /track_preview_car/cmd_vel: "
            f"MOTION every {self.telemetry_period_ns * 1e-9:.2f}s, "
            f"JITTER at most every {self.jitter_log_period_ns * 1e-9:.2f}s, "
            f"verbose={self.motion_log_verbose}; "
            f"jitter thresholds xy_step_delta={self.jitter_xy_step_delta_m:.4f}m, "
            f"cmd_error={self.jitter_command_error_m:.3f}m, z_delta={self.jitter_z_delta_m:.4f}m, "
            f"tilt={self.jitter_tilt_rad:.4f}rad, speed_delta={self.jitter_speed_delta_mps:.3f}m/s, "
            f"yaw_step_error={self.jitter_yaw_step_rad:.3f}rad, "
            f"yaw_rate_delta={self.jitter_yaw_rate_delta_rps:.3f}rad/s."
        )

    def publish_stop_command(self) -> None:
        twist = Twist()
        self.cmd_pub.publish(twist)
        self.last_cmd_linear_mps = 0.0
        self.last_cmd_angular_rps = 0.0
        self.sent_cmd_count += 1

    def publish_wheel_motion(self, stamp_ns: int) -> None:
        actual = self.last_actual_sample
        if actual is None:
            self.publish_stop_command()
            if not self.wheel_pose_wait_logged:
                self.get_logger().warn("Waiting for /model_states pose before publishing wheel commands.")
                self.wheel_pose_wait_logged = True
            return

        route_x, route_y, route_yaw = self.track.sample()
        route_yaw = normalize_angle(route_yaw + self.yaw_offset)
        speed = self.effective_speed()
        if speed <= 1e-6:
            self.publish_stop_command()
            self.last_command_sample = MotionSample(stamp_ns, route_x, route_y, self.z, 0.0, 0.0, route_yaw, 0.0, 0.0)
            return

        lookahead = (
            self.wheel_control_parking_lookahead_m
            if self.track.current_segment.parking_route
            else self.wheel_control_lookahead_m
        )
        controller_projection = self.track.current_segment.project(actual.x, actual.y)
        if self.wheel_control_nearest_route_enabled and not self.track.stopped_at_terminal:
            inner_route_authorized = self.track.inner_route_is_authorized()
            nearest_projection = self.track.nearest_pose_projection(
                actual.x,
                actual.y,
                actual.yaw - self.yaw_offset,
                outer_only=not inner_route_authorized,
            )
            current_projection = controller_projection
            nearest_yaw_error = abs(normalize_angle(nearest_projection.yaw + self.yaw_offset - actual.yaw))
            remaining_on_current = self.track.current_segment.length - current_projection.distance_on_segment
            is_current_segment = nearest_projection.segment is self.track.current_segment
            is_direct_successor = self.track.is_direct_successor(nearest_projection.segment)
            direct_successor_count = self.track.direct_successor_count()
            can_switch_to_successor = (
                is_direct_successor
                and direct_successor_count <= 1
                and remaining_on_current <= max(lookahead, WHEEL_CONTROL_BRANCH_ENDPOINT_SWITCH_DISTANCE_M)
                and nearest_yaw_error <= WHEEL_CONTROL_NEAREST_ROUTE_SWITCH_YAW_ERROR_RAD
            )
            should_rejoin_route = (
                current_projection.distance_m > self.route_progress_sync_max_route_distance_m
                and nearest_projection.distance_m + WHEEL_CONTROL_NEAREST_ROUTE_REJOIN_MARGIN_M
                < current_projection.distance_m
                and nearest_yaw_error <= WHEEL_CONTROL_NEAREST_ROUTE_REJOIN_YAW_ERROR_RAD
            )
            if (
                nearest_projection.distance_m <= self.wheel_control_nearest_route_max_distance_m
                and (is_current_segment or can_switch_to_successor or should_rejoin_route)
            ):
                controller_projection = nearest_projection
                self.track.switch_to_projection(
                    nearest_projection,
                    f"route/wheel_nearest: {nearest_projection.segment.name}",
                )

        remaining_m = self.track.current_segment.length - self.track.distance_on_segment
        branch_turn_pending = (
            self.track.pending_maneuver in {"left", "right", "straight"}
            and self.track.current_segment_has_maneuver_branch()
        )
        recently_entered_maneuver_branch = (
            self.track.current_segment.inner_route
            and not self.track.current_segment.parking_route
            and self.track.distance_since_branch < WHEEL_CONTROL_BRANCH_EXIT_DISTANCE_M
        )
        target_x, target_y = self.track._xy_ahead_from(
            controller_projection.segment,
            controller_projection.distance_on_segment,
            lookahead,
        )
        dx = target_x - actual.x
        dy = target_y - actual.y
        target_distance = math.hypot(dx, dy)
        forward_m = dx * math.cos(actual.yaw) + dy * math.sin(actual.yaw)
        lateral_m = -dx * math.sin(actual.yaw) + dy * math.cos(actual.yaw)
        if target_distance <= 1e-4:
            target_heading = route_yaw
        else:
            target_heading = math.atan2(dy, dx)

        heading_error = normalize_angle(target_heading - actual.yaw)
        pure_pursuit_distance = max(target_distance, 0.05)
        curvature = 2.0 * lateral_m / max(pure_pursuit_distance * pure_pursuit_distance, 1e-6)
        turn_ratio = min(abs(heading_error) / self.wheel_control_turn_slowdown_rad, 1.0)
        heading_speed_scale = 1.0 - (1.0 - self.wheel_control_min_speed_scale) * turn_ratio
        if abs(curvature) <= 1e-6:
            curvature_speed_scale = 1.0
        else:
            curvature_radius_m = 1.0 / abs(curvature)
            radius_scale = curvature_radius_m / max(lookahead * 2.4, 1e-6)
            curve_response_scale = 1.0 / (1.0 + abs(curvature) * lookahead * 0.85)
            curvature_speed_scale = min(
                1.0,
                max(
                    self.wheel_control_min_speed_scale,
                    min(radius_scale, curve_response_scale),
                ),
            )
        linear = speed * min(heading_speed_scale, curvature_speed_scale)
        if abs(heading_error) > WHEEL_CONTROL_SHARP_TURN_HEADING_ERROR_RAD:
            linear = min(linear, speed * WHEEL_CONTROL_SHARP_TURN_SPEED_SCALE)
        if branch_turn_pending and remaining_m < WHEEL_CONTROL_BRANCH_APPROACH_DISTANCE_M:
            branch_scale = max(
                WHEEL_CONTROL_BRANCH_APPROACH_MIN_SCALE,
                (remaining_m / max(WHEEL_CONTROL_BRANCH_APPROACH_DISTANCE_M, 1e-6)) ** 2,
            )
            linear = min(linear, speed * branch_scale)
        if recently_entered_maneuver_branch and abs(heading_error) > WHEEL_CONTROL_SHARP_TURN_HEADING_ERROR_RAD:
            linear = min(linear, speed * WHEEL_CONTROL_SHARP_TURN_SPEED_SCALE)
        rotate_heading_error_rad = self.wheel_control_rotate_heading_error_rad
        if branch_turn_pending or recently_entered_maneuver_branch:
            rotate_heading_error_rad = max(
                rotate_heading_error_rad,
                WHEEL_CONTROL_BRANCH_ROTATE_HEADING_ERROR_RAD,
            )
        rotate_to_path = (
            forward_m < -0.02
            or abs(heading_error) > rotate_heading_error_rad
        )
        if rotate_to_path:
            linear = 0.0

        if self.track.current_segment.terminal_stop and remaining_m < self.wheel_control_terminal_slowdown_m:
            terminal_scale = max(
                self.wheel_control_min_speed_scale,
                remaining_m / max(self.wheel_control_terminal_slowdown_m, 1e-6),
            )
            linear = min(linear, speed * terminal_scale)

        if rotate_to_path:
            angular = self.wheel_control_heading_gain * heading_error
        else:
            heading_correction = 0.35 * self.wheel_control_heading_gain * heading_error
            angular = linear * curvature + heading_correction
        angular = min(max(angular, -self.wheel_control_max_angular_rps), self.wheel_control_max_angular_rps)

        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_pub.publish(twist)
        self.last_cmd_linear_mps = linear
        self.last_cmd_angular_rps = angular
        self.sent_cmd_count += 1
        self.last_command_sample = MotionSample(
            stamp_ns,
            route_x,
            route_y,
            self.z,
            0.0,
            0.0,
            route_yaw,
            linear,
            angular,
        )

    def traffic_light_cycle_state(self, now_s: float) -> str:
        total_duration = sum(self.traffic_light_durations_s[state] for state in TRAFFIC_LIGHT_CYCLE_ORDER)
        phase = now_s % max(total_duration, 0.1)
        for state in TRAFFIC_LIGHT_CYCLE_ORDER:
            duration = self.traffic_light_durations_s[state]
            if phase < duration:
                return state
            phase -= duration
        return TRAFFIC_LIGHT_CYCLE_ORDER[-1]

    def refresh_traffic_light_state(self, stamp_ns: int, force_visual: bool = False) -> None:
        if self.traffic_light_cycle_enabled:
            self.traffic_light_state = self.traffic_light_cycle_state(stamp_ns * 1e-9)

        if self.traffic_light_state != self.last_logged_traffic_light_state:
            self.get_logger().info(f"traffic light is now {self.traffic_light_state.upper()}.")
            self.last_logged_traffic_light_state = self.traffic_light_state

        self.update_traffic_light_lights(force=force_visual)

    def make_color_rgba(self, values: tuple[float, float, float, float]) -> ColorRGBA:
        color = ColorRGBA()
        color.r = float(values[0])
        color.g = float(values[1])
        color.b = float(values[2])
        color.a = float(values[3])
        return color

    def make_entity_state(self, name: str, x: float, y: float, z: float, yaw: float = 0.0) -> EntityState:
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        state = EntityState()
        state.name = name
        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)
        state.pose.orientation.x = qx
        state.pose.orientation.y = qy
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw
        state.reference_frame = "world"
        return state

    def traffic_light_scene_signature(self) -> tuple[tuple[str, float, float, float, float], ...]:
        signature = []
        for model_name in self.traffic_light_scene_model_names:
            x, y, z, yaw = self.traffic_light_scene_poses.get(
                model_name,
                self.traffic_light_scene_default_poses[model_name],
            )
            signature.append(
                (
                    model_name,
                    round(x, 4),
                    round(y, 4),
                    round(z, 4),
                    round(normalize_angle(yaw), 4),
                )
            )
        return tuple(signature)

    def traffic_light_indicator_pose(
        self,
        parent_model_name: str,
        local_z: float,
    ) -> tuple[float, float, float, float]:
        parent_x, parent_y, parent_z, parent_yaw = self.traffic_light_scene_poses.get(
            parent_model_name,
            self.traffic_light_scene_default_poses[parent_model_name],
        )
        cosine = math.cos(parent_yaw)
        sine = math.sin(parent_yaw)
        local_x = self.traffic_light_indicator_local_x_m
        local_y = self.traffic_light_indicator_local_y_m
        x = parent_x + local_x * cosine - local_y * sine
        y = parent_y + local_x * sine + local_y * cosine
        z = parent_z + local_z
        return x, y, z, parent_yaw

    def update_traffic_light_indicator_models(self, force: bool = False) -> None:
        scene_signature = self.traffic_light_scene_signature()
        if (
            not force
            and self.last_indicator_traffic_light_state == self.traffic_light_state
            and self.last_indicator_traffic_light_scene_signature == scene_signature
        ):
            return
        if not self.visual_state_client.service_is_ready():
            self.visual_state_client.wait_for_service(timeout_sec=0.0)
            if not self.indicator_wait_logged:
                self.get_logger().warn("Waiting for /set_entity_state before moving visible traffic light markers.")
                self.indicator_wait_logged = True
            return

        hidden_x, hidden_y, hidden_z = self.traffic_light_indicator_hidden_pose
        for state, model_poses in self.traffic_light_indicator_models.items():
            active = state == self.traffic_light_state
            for model_name, parent_model_name, local_z in model_poses:
                request = SetEntityState.Request()
                if active:
                    x, y, z, yaw = self.traffic_light_indicator_pose(parent_model_name, local_z)
                    request.state = self.make_entity_state(model_name, x, y, z, yaw)
                else:
                    request.state = self.make_entity_state(model_name, hidden_x, hidden_y, hidden_z)
                future = self.visual_state_client.call_async(request)
                future.add_done_callback(
                    lambda done_future, name=model_name: self.handle_indicator_response(done_future, name)
                )

        self.last_indicator_traffic_light_state = self.traffic_light_state
        self.last_indicator_traffic_light_scene_signature = scene_signature

    def handle_indicator_response(self, future, model_name: str) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warn(f"/set_entity_state failed for {model_name}: {exc}")
            return

        if response is not None and not response.success:
            status_message = getattr(response, "status_message", "entity update was rejected")
            self.get_logger().warn(f"/set_entity_state rejected {model_name}: {status_message}")

    def update_traffic_light_glow_poses(self, force: bool = False) -> None:
        scene_signature = self.traffic_light_scene_signature()
        if not force and self.last_glow_traffic_light_scene_signature == scene_signature:
            return
        if not self.visual_state_client.service_is_ready():
            self.visual_state_client.wait_for_service(timeout_sec=0.0)
            if not self.indicator_wait_logged:
                self.get_logger().warn("Waiting for /set_entity_state before moving traffic light glow markers.")
                self.indicator_wait_logged = True
            return

        for light_poses in TRAFFIC_LIGHT_GLOW_ENTITIES.values():
            for light_name, parent_model_name, local_z in light_poses:
                x, y, z, yaw = self.traffic_light_indicator_pose(parent_model_name, local_z)
                request = SetEntityState.Request()
                request.state = self.make_entity_state(light_name, x, y, z, yaw)
                future = self.visual_state_client.call_async(request)
                future.add_done_callback(
                    lambda done_future, name=light_name: self.handle_indicator_response(done_future, name)
                )

        self.last_glow_traffic_light_scene_signature = scene_signature

    def update_traffic_light_lights(self, force: bool = False) -> None:
        if not self.traffic_light_visual_update_enabled:
            return
        self.update_traffic_light_indicator_models(force=force)
        if TRAFFIC_LIGHT_GLOW_ENTITIES:
            self.update_traffic_light_glow_poses(force=force)
        if not TRAFFIC_LIGHT_GLOW_NAMES:
            self.last_visual_traffic_light_state = self.traffic_light_state
            return
        if not force and self.last_visual_traffic_light_state == self.traffic_light_state:
            return
        if not self.light_client.service_is_ready():
            self.light_client.wait_for_service(timeout_sec=0.0)
            if not self.light_wait_logged:
                self.get_logger().warn(
                    f"Waiting for {self.traffic_light_light_service}; traffic logic works, "
                    "but Gazebo glow lights will update after the service is ready."
                )
                self.light_wait_logged = True
            return

        attenuation_constant, attenuation_linear, attenuation_quadratic = TRAFFIC_LIGHT_GLOW_ATTENUATION
        for state, light_names in TRAFFIC_LIGHT_GLOW_NAMES.items():
            color_values = (
                TRAFFIC_LIGHT_GLOW_COLORS[state]
                if state == self.traffic_light_state
                else TRAFFIC_LIGHT_GLOW_OFF_COLOR
            )
            for light_name in light_names:
                request = SetLightProperties.Request()
                request.light_name = light_name
                request.diffuse = self.make_color_rgba(color_values)
                request.attenuation_constant = attenuation_constant
                request.attenuation_linear = attenuation_linear
                request.attenuation_quadratic = attenuation_quadratic
                future = self.light_client.call_async(request)
                future.add_done_callback(
                    lambda done_future, name=light_name: self.handle_light_response(done_future, name)
                )

        self.last_visual_traffic_light_state = self.traffic_light_state

    def handle_light_response(self, future, light_name: str) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warn(f"{self.traffic_light_light_service} failed for {light_name}: {exc}")
            return

        if response is not None and not response.success:
            self.get_logger().warn(
                f"{self.traffic_light_light_service} rejected {light_name}: {response.status_message}"
            )

    def load_camera_sign_templates(self) -> list[CameraSignTemplate]:
        templates: list[CameraSignTemplate] = []
        for name, kind, maneuver, path_parts in CAMERA_SIGN_TEXTURES:
            path = share_file(*path_parts)
            if path is None:
                self.get_logger().warn(f"camera sign template {name} is missing at {'/'.join(path_parts)}")
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                self.get_logger().warn(f"camera sign template {name} could not be read from {path}")
                continue
            feature, blue_ratio, red_ratio, white_ratio = self.camera_sign_feature(image)
            templates.append(
                CameraSignTemplate(
                    name,
                    kind,
                    maneuver,
                    feature,
                    blue_ratio,
                    red_ratio,
                    white_ratio,
                )
            )
        return templates

    def camera_sign_feature(self, image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        resized = cv2.resize(image, CAMERA_SIGN_FEATURE_SIZE, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array((85, 45, 45), dtype=np.uint8), np.array((135, 255, 255), dtype=np.uint8))
        red_low = cv2.inRange(hsv, np.array((0, 55, 55), dtype=np.uint8), np.array((12, 255, 255), dtype=np.uint8))
        red_high = cv2.inRange(hsv, np.array((168, 55, 55), dtype=np.uint8), np.array((179, 255, 255), dtype=np.uint8))
        red = cv2.bitwise_or(red_low, red_high)
        white = cv2.inRange(hsv, np.array((0, 0, 135), dtype=np.uint8), np.array((179, 90, 255), dtype=np.uint8))
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

    def camera_sign_template_score(
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
        if denominator <= 1e-6:
            feature_score = 0.0
        else:
            feature_score = max(0.0, min(1.0, numerator / denominator))

        color_error = (
            abs(blue_ratio - template.blue_ratio)
            + abs(red_ratio - template.red_ratio)
            + 0.5 * abs(white_ratio - template.white_ratio)
        )
        color_score = max(0.0, 1.0 - 2.5 * color_error)
        return 0.82 * feature_score + 0.18 * color_score

    def camera_sign_template_sanity_allows(
        self,
        template: CameraSignTemplate,
        blue_ratio: float,
        red_ratio: float,
        white_ratio: float,
        crop_width: int,
        crop_height: int,
    ) -> bool:
        aspect = crop_width / float(max(1, crop_height))
        if aspect < 0.35 or aspect > 3.0:
            return False

        if template.kind == "speed_limit_sign":
            return red_ratio >= 0.08 and white_ratio >= 0.12 and blue_ratio <= 0.16

        if template.kind in {"go_straight_sign", "turn_left_sign", "turn_right_sign", "parking_sign"}:
            if blue_ratio < 0.08 or red_ratio > 0.18:
                return False
            if template.kind == "parking_sign" and white_ratio < 0.06:
                return False
            return True

        return True

    def camera_sign_feature_score(self, candidate: np.ndarray, template: CameraSignTemplate) -> float:
        feature, blue_ratio, red_ratio, white_ratio = self.camera_sign_feature(candidate)
        return self.camera_sign_template_score(feature, blue_ratio, red_ratio, white_ratio, template)

    def handle_camera_image(self, msg: Image) -> None:
        if not self.traffic_light_camera_enabled:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Could not read traffic light camera frame: {exc}")
            return

        stamp_ns = self.get_clock().now().nanoseconds
        detection = self.detect_camera_traffic_light(frame)
        if detection is None:
            self.refresh_camera_traffic_light_timeout(stamp_ns)
        else:
            self.camera_traffic_light_state = detection.state
            self.camera_traffic_light_score = detection.score
            self.camera_traffic_light_area_px = detection.area_px
            self.camera_traffic_light_center = (detection.center_x, detection.center_y)
            self.camera_traffic_light_size = (detection.width_px, detection.height_px)
            self.camera_traffic_light_fill_ratio = detection.fill_ratio
            self.last_camera_detection_ns = stamp_ns

            if detection.state != self.last_camera_traffic_light_state:
                self.get_logger().info(
                    f"camera traffic light is {detection.state.upper()} "
                    f"score={detection.score:.1f} area={detection.area_px}px "
                    f"bbox={detection.width_px}x{detection.height_px}px "
                    f"fill={detection.fill_ratio:.2f} "
                    f"center=({detection.center_x},{detection.center_y})"
                )
                self.last_camera_traffic_light_state = detection.state

        self.update_traffic_light_control_from_camera(stamp_ns)
        self.publish_camera_traffic_light_state(stamp_ns)
        self.update_camera_instruction_from_frame(frame, stamp_ns)

    def camera_traffic_light_is_current(self, stamp_ns: int) -> bool:
        if self.camera_traffic_light_state is None:
            return False
        return stamp_ns - self.last_camera_detection_ns <= self.traffic_light_camera_timeout_ns

    def traffic_light_state_for_control(self, stamp_ns: int) -> str | None:
        if not self.camera_traffic_light_is_current(stamp_ns):
            return None
        return self.camera_traffic_light_state

    def detect_camera_traffic_light(self, frame: np.ndarray) -> TrafficLightCameraDetection | None:
        height, width = frame.shape[:2]
        roi_top = int(height * self.traffic_light_camera_roi_top_ratio)
        roi_bottom = int(height * self.traffic_light_camera_roi_bottom_ratio)
        if roi_bottom <= roi_top:
            return None

        roi = frame[roi_top:roi_bottom, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        max_area = max(
            self.traffic_light_camera_min_area_px,
            int(width * max(1, roi_bottom - roi_top) * self.traffic_light_camera_max_area_ratio),
        )
        kernel = np.ones((3, 3), np.uint8)
        best_detection: TrafficLightCameraDetection | None = None

        for state, ranges in TRAFFIC_LIGHT_CAMERA_COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(
                    mask,
                    cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)),
                )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            candidate = self.best_traffic_light_component(state, hsv, mask, roi_top, max_area)
            if candidate is None:
                continue
            if best_detection is None or candidate.score > best_detection.score:
                best_detection = candidate

        if best_detection is None or best_detection.score < self.traffic_light_camera_min_score:
            return None
        return best_detection

    def best_traffic_light_component(
        self,
        state: str,
        hsv: np.ndarray,
        mask: np.ndarray,
        roi_top: int,
        max_area: int,
    ) -> TrafficLightCameraDetection | None:
        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best: TrafficLightCameraDetection | None = None
        for label_index in range(1, component_count):
            w = int(stats[label_index, cv2.CC_STAT_WIDTH])
            h = int(stats[label_index, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_index, cv2.CC_STAT_AREA])
            if area < self.traffic_light_camera_min_area_px or area > max_area:
                continue
            aspect = max(w / h, h / w)
            if w <= 0 or h <= 0 or aspect > 4.0:
                continue
            if state == "red" and aspect > 3.5:
                continue
            fill_ratio = area / float(max(1, w * h))
            if fill_ratio < self.traffic_light_camera_min_fill_ratio:
                continue

            component_mask = labels == label_index
            mean_s = float(np.mean(hsv[:, :, 1][component_mask]))
            mean_v = float(np.mean(hsv[:, :, 2][component_mask]))
            score = area * (mean_s / 255.0) * (mean_v / 255.0)
            center_x = int(round(float(centroids[label_index][0])))
            center_y = int(round(float(centroids[label_index][1]) + roi_top))
            frame_width = hsv.shape[1]
            if center_x < int(frame_width * 0.04) or center_x > int(frame_width * 0.96):
                continue
            detection = TrafficLightCameraDetection(state, score, area, center_x, center_y, w, h, fill_ratio)
            if best is None or detection.score > best.score:
                best = detection

        return best

    def refresh_camera_traffic_light_timeout(self, stamp_ns: int) -> None:
        if self.camera_traffic_light_state is None:
            return
        if self.camera_traffic_light_is_current(stamp_ns):
            return

        self.get_logger().info(
            f"camera traffic light lost for {self.traffic_light_camera_timeout_ns * 1e-9:.1f}s."
        )
        self.camera_traffic_light_state = None
        self.camera_traffic_light_score = 0.0
        self.camera_traffic_light_area_px = 0
        self.camera_traffic_light_center = None
        self.camera_traffic_light_size = None
        self.camera_traffic_light_fill_ratio = 0.0
        self.last_camera_traffic_light_state = None
        if self.traffic_light_stop_latched:
            self.stop_for_traffic_light = True
        else:
            self.stop_for_traffic_light = False
        self.slow_for_traffic_light = False

    def traffic_light_control_target_payload(self) -> dict[str, float | str] | None:
        target = self.traffic_light_control_target
        if target is None:
            return None
        return {
            "name": target.name,
            "distance_m": target.distance_m,
            "forward_m": target.forward_m,
            "lateral_m": target.lateral_m,
            "angle_rad": target.angle_rad,
        }

    def publish_camera_traffic_light_state(self, stamp_ns: int) -> None:
        message = String()
        center_payload = None
        if self.camera_traffic_light_center is not None:
            center_payload = {
                "x": self.camera_traffic_light_center[0],
                "y": self.camera_traffic_light_center[1],
            }
        message.data = json.dumps(
            {
                "stamp_ns": stamp_ns,
                "state": self.camera_traffic_light_state,
                "control_state": self.traffic_light_state_for_control(stamp_ns),
                "visible": self.camera_traffic_light_is_current(stamp_ns),
                "score": self.camera_traffic_light_score,
                "area_px": self.camera_traffic_light_area_px,
                "size_px": self.camera_traffic_light_size,
                "fill_ratio": self.camera_traffic_light_fill_ratio,
                "center": center_payload,
                "stop": self.stop_for_traffic_light,
                "slow": self.slow_for_traffic_light,
                "stop_latched": self.traffic_light_stop_latched,
                "stop_latched_light": self.traffic_light_stop_latched_name,
                "route_segment": self.track.current_segment.name,
                "route_zone_allows_action": self.traffic_light_camera_route_zone_allows_action(),
                "control_target": self.traffic_light_control_target_payload(),
                "trigger_distance_m": self.traffic_light_trigger_distance_m,
                "stop_distance_m": self.traffic_light_stop_distance_m,
                "source": "camera",
            },
            sort_keys=True,
        )
        self.camera_traffic_light_pub.publish(message)

    def effective_speed(self) -> float:
        if self.track.stopped_at_terminal:
            return 0.0
        if self.stop_for_traffic_light:
            return 0.0
        speed = self.active_speed_mps
        if self.slow_for_traffic_light:
            speed = min(speed, self.speed_limit_mps)
        if self.track.current_segment.parking_route:
            return min(speed, self.speed_limit_mps)
        return speed

    def refresh_speed_limit_release(self) -> None:
        if self.active_speed_limit_name is None:
            return

        segment_changed = self.track.current_segment.name != self.speed_limit_release_segment_name
        passed_release_point = self.track.distance_on_segment >= self.speed_limit_release_distance_m
        if not segment_changed and not passed_release_point:
            return

        self.get_logger().info(
            f"camera speed limit sign {self.active_speed_limit_name} cleared; "
            f"speed restored to {self.speed:.2f}m/s "
            f"segment={self.track.current_segment.name} "
            f"distance={self.track.distance_on_segment:.3f}m"
        )
        self.active_speed_limit_name = None
        self.speed_limit_release_segment_name = None
        self.speed_limit_release_distance_m = 0.0
        self.active_speed_mps = self.speed

    def traffic_light_route_context_allows_action(self) -> bool:
        segment = self.track.current_segment
        return (
            not self.track.stopped_at_terminal
            and not segment.parking_route
        )

    def traffic_light_camera_route_zone_allows_action(self) -> bool:
        self.traffic_light_control_target = self.nearest_relevant_traffic_light_ahead()
        return self.traffic_light_control_target is not None

    def publish_route_trace(self, stamp_ns: int, force: bool = False) -> None:
        if not self.route_trace_enabled:
            return

        segment = self.track.current_segment
        signature = (segment.name, self.track.last_decision_text, self.track.stopped_at_terminal)
        periodic = stamp_ns - self.last_route_trace_ns >= self.route_trace_period_ns
        if not force and not periodic and signature == self.last_route_trace_signature:
            return

        x, y, yaw = self.track.sample()
        map_x, map_y = model_point_to_map(x, y, self.track.route_offset_x_m, self.track.route_offset_y_m)
        command_payload = None
        if self.last_command_sample is not None:
            command_map_x, command_map_y = model_point_to_map(
                self.last_command_sample.x,
                self.last_command_sample.y,
                self.track.route_offset_x_m,
                self.track.route_offset_y_m,
            )
            command_payload = {
                "x": self.last_command_sample.x,
                "y": self.last_command_sample.y,
                "yaw": self.last_command_sample.yaw,
                "map_x": command_map_x,
                "map_y": command_map_y,
                "speed_xy": self.last_command_sample.speed_xy,
                "yaw_rate": self.last_command_sample.yaw_rate,
            }
        actual_payload = None
        if self.last_actual_sample is not None:
            actual_map_x, actual_map_y = model_point_to_map(
                self.last_actual_sample.x,
                self.last_actual_sample.y,
                self.track.route_offset_x_m,
                self.track.route_offset_y_m,
            )
            actual_payload = {
                "x": self.last_actual_sample.x,
                "y": self.last_actual_sample.y,
                "yaw": self.last_actual_sample.yaw,
                "map_x": actual_map_x,
                "map_y": actual_map_y,
                "speed_xy": self.last_actual_sample.speed_xy,
                "yaw_rate": self.last_actual_sample.yaw_rate,
            }
        message = String()
        message.data = json.dumps(
            {
                "stamp_ns": stamp_ns,
                "segment": segment.name,
                "distance_on_segment_m": self.track.distance_on_segment,
                "segment_length_m": segment.length,
                "segment_progress": self.track.distance_on_segment / max(segment.length, 1e-9),
                "parking_route": segment.parking_route,
                "terminal_stop": segment.terminal_stop,
                "stopped_at_terminal": self.track.stopped_at_terminal,
                "decision": self.track.last_decision_text,
                "pending_maneuver": self.track.pending_maneuver,
                "pending_source": self.track.pending_maneuver_source,
                "camera_parking_armed": self.camera_parking_armed_name,
                "pose": {"x": x, "y": y, "yaw": yaw, "map_x": map_x, "map_y": map_y},
                "command_pose": command_payload,
                "actual_pose": actual_payload,
                "effective_speed_mps": self.effective_speed(),
                "active_speed_mps": self.active_speed_mps,
                "pending_speed_limit": self.pending_speed_limit_name,
                "active_speed_limit": self.active_speed_limit_name,
                "speed_limit_release_segment": self.speed_limit_release_segment_name,
                "speed_limit_release_distance_m": self.speed_limit_release_distance_m,
                "traffic_light_state": self.traffic_light_state,
                "camera_traffic_light_state": self.camera_traffic_light_state,
                "traffic_light_control_state": self.traffic_light_state_for_control(stamp_ns),
                "traffic_light_control_target": self.traffic_light_control_target_payload(),
                "traffic_light_trigger_distance_m": self.traffic_light_trigger_distance_m,
                "traffic_light_stop_distance_m": self.traffic_light_stop_distance_m,
                "camera_traffic_light_visible": self.camera_traffic_light_is_current(stamp_ns),
                "camera_traffic_light_score": self.camera_traffic_light_score,
                "stop_for_traffic_light": self.stop_for_traffic_light,
                "slow_for_traffic_light": self.slow_for_traffic_light,
                "traffic_light_stop_latched": self.traffic_light_stop_latched,
                "traffic_light_stop_latched_light": self.traffic_light_stop_latched_name,
                "traffic_light_stop_latched_elapsed_s": (
                    (stamp_ns - self.traffic_light_stop_latched_since_ns) * 1e-9
                    if self.traffic_light_stop_latched
                    else 0.0
                ),
                "tracked_traffic_light": self.tracked_traffic_light_name,
                "visible_instruction": self.last_visible_instruction_name,
                "applied_instruction": self.last_applied_instruction_name,
                "route_points_map": [
                    [round(px, 3), round(py, 3)]
                    for px, py in (
                        model_point_to_map(
                            point_x,
                            point_y,
                            self.track.route_offset_x_m,
                            self.track.route_offset_y_m,
                        )
                        for point_x, point_y in segment.points
                    )
                ],
            },
            sort_keys=True,
        )
        self.route_trace_pub.publish(message)
        self.last_route_trace_ns = stamp_ns
        self.last_route_trace_signature = signature

    def update_car(self) -> None:
        now = self.get_clock().now()
        self.refresh_traffic_light_state(now.nanoseconds)
        self.refresh_camera_traffic_light_timeout(now.nanoseconds)
        self.update_traffic_light_control_from_camera(now.nanoseconds)
        self.publish_camera_traffic_light_state(now.nanoseconds)

        dt = max(0.0, (now - self.last_update_time).nanoseconds * 1e-9)
        self.last_update_time = now
        if not self.drive_wheels:
            self.publish_stop_command()
            if not self.wheel_disabled_logged:
                self.get_logger().error(
                    "drive_wheels is false; refusing to guide the car with /set_entity_state. "
                    "Enable drive_wheels to use the diff-drive robot control loop."
                )
                self.wheel_disabled_logged = True
            return

        if self.skip_route_advance_once:
            progress_speed = 0.0
            self.skip_route_advance_once = False
        else:
            progress_speed = max(self.last_cmd_linear_mps, 0.0)
            if self.last_actual_sample is not None:
                actual_cap = max(0.0, self.last_actual_sample.speed_xy) + 0.02
                progress_speed = min(progress_speed, actual_cap, max(self.speed, self.speed_limit_mps))
        self.track.advance(dt * progress_speed)
        if self.last_actual_sample is not None and self.track.force_pending_maneuver_branch_at_endpoint(
            self.last_actual_sample.x,
            self.last_actual_sample.y,
            WHEEL_CONTROL_BRANCH_ENDPOINT_SWITCH_DISTANCE_M,
        ):
            self.last_cmd_linear_mps = 0.0
            self.publish_stop_command()
        self.refresh_speed_limit_release()
        self.start_armed_camera_parking_if_ready()
        if self.track.stopped_at_terminal and not self.parking_stop_logged:
            x, y, yaw = self.track.sample()
            self.get_logger().warn(
                f"parking complete; stopped at segment={self.track.current_segment.name} "
                f"pose=({x:+.3f},{y:+.3f}) yaw={yaw:+.3f}"
            )
            self.parking_stop_logged = True
            self.publish_route_trace(now.nanoseconds, force=True)
        if self.track.current_segment.name != self.last_segment_name:
            self.get_logger().info(
                f"route segment={self.track.current_segment.name}; previous={self.last_segment_name}; "
                f"distance={self.track.distance_on_segment:.3f}m; decision={self.track.last_decision_text}"
            )
            self.last_segment_name = self.track.current_segment.name
            self.last_applied_instruction_key = None
            self.publish_route_trace(now.nanoseconds, force=True)
        else:
            self.publish_route_trace(now.nanoseconds)
        self.publish_wheel_motion(now.nanoseconds)

    def traffic_light_camera_reaches_slow_trigger(self) -> bool:
        width_px, height_px = self.camera_traffic_light_size or (0, 0)
        return (
            self.camera_traffic_light_area_px >= self.traffic_light_camera_slow_min_area_px
            or height_px >= self.traffic_light_camera_slow_min_height_px
            or width_px >= self.traffic_light_camera_slow_min_height_px
        )

    def traffic_light_camera_reaches_stop_trigger(self) -> bool:
        width_px, height_px = self.camera_traffic_light_size or (0, 0)
        return (
            self.camera_traffic_light_area_px >= self.traffic_light_camera_stop_min_area_px
            or height_px >= self.traffic_light_camera_stop_min_height_px
            or width_px >= self.traffic_light_camera_stop_min_height_px
        )

    def nearest_relevant_traffic_light_ahead(
        self,
        max_distance_m: float | None = None,
    ) -> TrafficLightSceneTarget | None:
        if not self.traffic_light_route_context_allows_action():
            return None

        if self.last_actual_sample is not None:
            car_x = self.last_actual_sample.x
            car_y = self.last_actual_sample.y
            car_yaw = self.last_actual_sample.yaw
        else:
            car_x, car_y, route_yaw = self.track.sample()
            car_yaw = route_yaw + self.yaw_offset

        max_distance = (
            self.traffic_light_trigger_distance_m
            if max_distance_m is None
            else max(0.1, max_distance_m)
        )
        half_fov = self.traffic_light_scene_release_fov_rad * 0.5
        forward_x = math.cos(car_yaw)
        forward_y = math.sin(car_yaw)
        best_target: TrafficLightSceneTarget | None = None

        for model_name in self.traffic_light_scene_model_names:
            light_x, light_y, _, _ = self.traffic_light_scene_poses.get(
                model_name,
                self.traffic_light_scene_default_poses[model_name],
            )
            dx = light_x - car_x
            dy = light_y - car_y
            forward_distance = dx * forward_x + dy * forward_y
            if forward_distance <= TRAFFIC_LIGHT_ACQUIRE_MIN_FORWARD_M:
                continue

            distance = math.hypot(dx, dy)
            if distance > max_distance:
                continue

            lateral_distance = abs(-dx * forward_y + dy * forward_x)
            angle = math.atan2(lateral_distance, max(forward_distance, 1e-6))
            if angle > half_fov:
                continue

            target = TrafficLightSceneTarget(
                model_name,
                distance,
                forward_distance,
                lateral_distance,
                angle,
            )
            if best_target is None or target.distance_m < best_target.distance_m:
                best_target = target

        return best_target

    def traffic_light_scene_has_relevant_light_ahead(self) -> bool:
        return self.nearest_relevant_traffic_light_ahead() is not None

    def log_traffic_light_route_guard_block(self) -> None:
        signature = (self.track.current_segment.name, self.camera_traffic_light_state)
        if signature == self.last_traffic_light_route_guard_signature:
            return
        self.last_traffic_light_route_guard_signature = signature

        center = self.camera_traffic_light_center or (-1, -1)
        size = self.camera_traffic_light_size or (0, 0)
        nearby_target = self.nearest_relevant_traffic_light_ahead(TRAFFIC_LIGHT_CAPTURE_MAX_DISTANCE_M)
        if nearby_target is None:
            target_text = "-"
        else:
            target_text = (
                f"{nearby_target.name} "
                f"dist={nearby_target.distance_m:.2f}m "
                f"forward={nearby_target.forward_m:.2f}m "
                f"lateral={nearby_target.lateral_m:.2f}m"
            )
        self.get_logger().info(
            "camera traffic light ignored outside distance gate; "
            f"state={self.camera_traffic_light_state or '-'} "
            f"segment={self.track.current_segment.name} "
            f"gate<={self.traffic_light_trigger_distance_m:.2f}m "
            f"nearby_target={target_text} "
            f"area={self.camera_traffic_light_area_px}px "
            f"bbox={size[0]}x{size[1]}px "
            f"center=({center[0]},{center[1]})"
        )

    def latch_traffic_light_stop(self, stamp_ns: int) -> None:
        if self.traffic_light_stop_latched:
            return
        target_name = self.tracked_traffic_light_name or (
            self.traffic_light_control_target.name
            if self.traffic_light_control_target is not None
            else "camera_traffic_light"
        )
        self.traffic_light_stop_latched = True
        self.traffic_light_stop_latched_since_ns = stamp_ns
        self.traffic_light_stop_latched_name = target_name
        self.tracked_traffic_light_name = target_name
        target = self.traffic_light_control_target
        if target is None:
            target_text = ""
        else:
            target_text = (
                f"dist={target.distance_m:.2f}m "
                f"forward={target.forward_m:.2f}m "
                f"lateral={target.lateral_m:.2f}m "
            )
        center = self.camera_traffic_light_center or (-1, -1)
        size = self.camera_traffic_light_size or (0, 0)
        self.get_logger().warn(
            "red traffic light stop latched from camera; "
            f"target={target_name} "
            f"{target_text}"
            f"area={self.camera_traffic_light_area_px}px "
            f"bbox={size[0]}x{size[1]}px fill={self.camera_traffic_light_fill_ratio:.2f} "
            f"center=({center[0]},{center[1]}) "
            f"min_stop={self.traffic_light_min_stop_ns * 1e-9:.1f}s"
        )

    def release_traffic_light_stop_latch(self, stamp_ns: int, reason: str) -> None:
        elapsed_s = max(0.0, (stamp_ns - self.traffic_light_stop_latched_since_ns) * 1e-9)
        self.get_logger().info(
            f"released red traffic light stop latch; reason={reason} elapsed={elapsed_s:.1f}s "
            f"light={self.traffic_light_stop_latched_name or '-'}"
        )
        self.traffic_light_stop_latched = False
        self.traffic_light_stop_latched_since_ns = 0
        self.traffic_light_stop_latched_name = None
        self.tracked_traffic_light_name = None

    def update_latched_traffic_light_stop(self, stamp_ns: int) -> bool:
        if not self.traffic_light_stop_latched:
            return False

        elapsed_ns = stamp_ns - self.traffic_light_stop_latched_since_ns
        control_state = self.traffic_light_state_for_control(stamp_ns)
        if control_state == "green" and elapsed_ns >= self.traffic_light_min_stop_ns:
            self.release_traffic_light_stop_latch(stamp_ns, "camera_green_after_min_stop")
            return False

        camera_visible = self.camera_traffic_light_is_current(stamp_ns)
        self.traffic_light_control_target = self.nearest_relevant_traffic_light_ahead()
        if camera_visible and self.traffic_light_control_target is None and elapsed_ns >= self.traffic_light_min_stop_ns:
            self.release_traffic_light_stop_latch(stamp_ns, "latched_light_outside_distance_gate")
            return False

        if (
            not camera_visible
            and elapsed_ns >= self.traffic_light_min_stop_ns
            and not self.traffic_light_scene_has_relevant_light_ahead()
        ):
            self.release_traffic_light_stop_latch(stamp_ns, "camera_lost_and_scene_light_not_ahead")
            return False

        self.stop_for_traffic_light = True
        self.slow_for_traffic_light = False
        return True

    def update_traffic_light_control_from_camera(self, stamp_ns: int) -> None:
        if self.update_latched_traffic_light_stop(stamp_ns):
            return

        camera_visible = self.camera_traffic_light_is_current(stamp_ns)
        if not camera_visible:
            self.tracked_traffic_light_name = None
            self.traffic_light_control_target = None
            self.stop_for_traffic_light = False
            self.slow_for_traffic_light = False
            return

        control_state = self.traffic_light_state_for_control(stamp_ns)
        if not self.traffic_light_camera_route_zone_allows_action():
            red_would_stop = (
                control_state == "red" and self.traffic_light_camera_reaches_stop_trigger()
            )
            yellow_would_slow = (
                control_state == "yellow" and self.traffic_light_camera_reaches_slow_trigger()
            )
            if red_would_stop or yellow_would_slow:
                self.log_traffic_light_route_guard_block()
            self.tracked_traffic_light_name = None
            self.stop_for_traffic_light = False
            self.slow_for_traffic_light = False
            return

        self.last_traffic_light_route_guard_signature = None
        target = self.traffic_light_control_target
        self.tracked_traffic_light_name = target.name if target is not None else "camera_traffic_light"
        if control_state == "red":
            close_enough_to_stop = (
                target is not None and target.distance_m <= self.traffic_light_stop_distance_m
            )
            if close_enough_to_stop and self.traffic_light_camera_reaches_stop_trigger():
                self.latch_traffic_light_stop(stamp_ns)
                self.stop_for_traffic_light = True
                self.slow_for_traffic_light = False
                return

            self.stop_for_traffic_light = False
            self.slow_for_traffic_light = self.traffic_light_camera_reaches_slow_trigger()
            return

        self.stop_for_traffic_light = False
        self.slow_for_traffic_light = (
            control_state == "yellow" and self.traffic_light_camera_reaches_slow_trigger()
        )

    def detect_camera_instruction(self, frame: np.ndarray) -> CameraInstructionDetection | None:
        if not self.sign_detection_enabled or not self.camera_sign_templates:
            return None

        frame_height, frame_width = frame.shape[:2]
        roi_top = int(frame_height * self.camera_sign_roi_top_ratio)
        roi_bottom = int(frame_height * self.camera_sign_roi_bottom_ratio)
        if roi_bottom <= roi_top:
            return None

        prefer_parking = self.parking_camera_route_context_allows_arm()
        roi = frame[roi_top:roi_bottom, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array((85, 45, 40), dtype=np.uint8), np.array((135, 255, 255), dtype=np.uint8))
        red_low = cv2.inRange(hsv, np.array((0, 55, 55), dtype=np.uint8), np.array((12, 255, 255), dtype=np.uint8))
        red_high = cv2.inRange(hsv, np.array((168, 55, 55), dtype=np.uint8), np.array((179, 255, 255), dtype=np.uint8))
        color_mask = cv2.bitwise_or(blue, cv2.bitwise_or(red_low, red_high))
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
        best_detection: CameraInstructionDetection | None = None
        best_maneuver_detection: CameraInstructionDetection | None = None
        best_parking_detection: CameraInstructionDetection | None = None
        best_priority = 0.0
        best_maneuver_priority = 0.0
        best_parking_priority = 0.0
        for label_index in range(1, component_count):
            colored_area = int(stats[label_index, cv2.CC_STAT_AREA])
            if colored_area < self.camera_sign_min_colored_area_px:
                continue
            x = int(stats[label_index, cv2.CC_STAT_LEFT])
            y = int(stats[label_index, cv2.CC_STAT_TOP])
            w = int(stats[label_index, cv2.CC_STAT_WIDTH])
            h = int(stats[label_index, cv2.CC_STAT_HEIGHT])
            if w <= 0 or h < self.camera_sign_min_bbox_height_px:
                continue
            if max(w / h, h / w) > 5.0:
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
            classified = self.classify_camera_sign_crop(
                crop,
                prefer_kind="parking_sign" if prefer_parking else None,
            )
            if classified is None:
                continue
            template, score = classified
            if score < self.camera_sign_min_score:
                continue

            bbox_width = x1 - x0
            bbox_height = y1 - y0
            bbox_aspect = max(bbox_width / max(1, bbox_height), bbox_height / max(1, bbox_width))
            if template.kind == "speed_limit_sign" and bbox_aspect > 1.85:
                continue
            bbox_area = bbox_width * bbox_height
            priority = score + min(0.20, bbox_area / float(max(1, frame_width * frame_height)) * 5.0)
            center_x = int(round(float(centroids[label_index][0])))
            center_y = int(round(float(centroids[label_index][1]) + roi_top))
            detection = CameraInstructionDetection(
                template.name,
                template.kind,
                template.maneuver,
                score,
                bbox_area,
                center_x,
                center_y,
                bbox_width,
                bbox_height,
                frame_width,
                frame_height,
            )
            if (
                prefer_parking
                and detection.kind == "parking_sign"
                and detection.area_px >= self.camera_sign_parking_min_area_px
                and (best_parking_detection is None or priority > best_parking_priority)
            ):
                best_parking_detection = detection
                best_parking_priority = priority
            if (
                detection.maneuver in {"left", "right", "straight"}
                and self.camera_instruction_center_allows_maneuver(detection)
                and (best_maneuver_detection is None or priority > best_maneuver_priority)
            ):
                best_maneuver_detection = detection
                best_maneuver_priority = priority
            if best_detection is None or priority > best_priority:
                best_detection = detection
                best_priority = priority

        if best_parking_detection is not None:
            return best_parking_detection
        if (
            best_maneuver_detection is not None
            and self.maneuver_camera_route_context_allows_action(best_maneuver_detection.maneuver)
        ):
            return best_maneuver_detection
        return best_detection

    def classify_camera_sign_crop(
        self,
        crop: np.ndarray,
        prefer_kind: str | None = None,
    ) -> tuple[CameraSignTemplate, float] | None:
        if crop.size == 0:
            return None
        feature, blue_ratio, red_ratio, white_ratio = self.camera_sign_feature(crop)
        best_template: CameraSignTemplate | None = None
        best_score = 0.0
        preferred_template: CameraSignTemplate | None = None
        preferred_score = 0.0
        parking_template: CameraSignTemplate | None = None
        parking_score = 0.0
        scores_by_kind: dict[str, float] = {}
        templates_by_kind: dict[str, CameraSignTemplate] = {}
        crop_height, crop_width = crop.shape[:2]
        for template in self.camera_sign_templates:
            score = self.camera_sign_template_score(feature, blue_ratio, red_ratio, white_ratio, template)
            if not self.camera_sign_template_sanity_allows(
                template,
                blue_ratio,
                red_ratio,
                white_ratio,
                crop_width,
                crop_height,
            ):
                continue
            scores_by_kind[template.kind] = max(scores_by_kind.get(template.kind, 0.0), score)
            if template.kind not in templates_by_kind or score >= scores_by_kind[template.kind]:
                templates_by_kind[template.kind] = template
            if template.kind == "parking_sign":
                parking_template = template
                parking_score = score
            if template.kind == prefer_kind:
                preferred_template = template
                preferred_score = score
            if score > best_score:
                best_template = template
                best_score = score

        if best_template is None:
            return None
        if (
            preferred_template is not None
            and preferred_score >= self.camera_sign_min_score
            and preferred_score >= best_score - CAMERA_SIGN_CONTEXT_PREFERRED_MARGIN
        ):
            return preferred_template, preferred_score
        if (
            prefer_kind == "parking_sign"
            and parking_template is not None
            and parking_score >= max(self.camera_sign_min_score, CAMERA_SIGN_PARKING_CONFUSION_MIN_SCORE)
        ):
            return parking_template, parking_score
        turn_opposite_kind = CAMERA_SIGN_TURN_OPPOSITE_KINDS.get(best_template.kind)
        if turn_opposite_kind is not None:
            scene_kind = self.camera_scene_preferred_sign_kind(
                allowed_kinds=frozenset((best_template.kind, turn_opposite_kind))
            )
            if scene_kind is not None:
                scene_template = templates_by_kind.get(scene_kind)
                scene_score = scores_by_kind.get(scene_kind, 0.0)
                if (
                    scene_template is not None
                    and scene_score >= self.camera_sign_min_score
                    and scene_score >= best_score - CAMERA_SIGN_TURN_AMBIGUITY_MARGIN
                ):
                    return scene_template, scene_score

        scene_kind = self.camera_scene_preferred_sign_kind()
        if scene_kind is not None:
            scene_template = templates_by_kind.get(scene_kind)
            scene_score = scores_by_kind.get(scene_kind, 0.0)
            if (
                scene_template is not None
                and scene_score >= self.camera_sign_min_score
                and scene_score >= best_score - CAMERA_SIGN_TURN_AMBIGUITY_MARGIN
            ):
                return scene_template, scene_score
        if self.parking_sign_confusion_blocks_maneuver(
            best_template,
            best_score,
            parking_template,
            parking_score,
        ):
            return None
        if self.turn_sign_confusion_blocks_maneuver(best_template, best_score, scores_by_kind):
            return None
        return best_template, best_score

    def turn_sign_confusion_blocks_maneuver(
        self,
        best_template: CameraSignTemplate,
        best_score: float,
        scores_by_kind: dict[str, float],
    ) -> bool:
        opposite_kind = CAMERA_SIGN_TURN_OPPOSITE_KINDS.get(best_template.kind)
        if opposite_kind is None:
            return False

        opposite_score = scores_by_kind.get(opposite_kind, 0.0)
        if opposite_score < self.camera_sign_min_score:
            return False
        if opposite_score < best_score - CAMERA_SIGN_TURN_AMBIGUITY_MARGIN:
            return False

        segment = self.track.current_segment
        signature = (segment.name, best_template.kind, opposite_kind)
        if signature != self.last_camera_turn_confusion_guard_signature:
            self.get_logger().info(
                "camera turn sign ignored as left/right ambiguous; "
                f"segment={segment.name} "
                f"best={best_template.name}:{best_score:.2f} "
                f"opposite={opposite_kind}:{opposite_score:.2f}"
            )
            self.last_camera_turn_confusion_guard_signature = signature
        return True

    def parking_sign_confusion_blocks_maneuver(
        self,
        best_template: CameraSignTemplate,
        best_score: float,
        parking_template: CameraSignTemplate | None,
        parking_score: float,
    ) -> bool:
        if best_template.kind not in CAMERA_SIGN_PARKING_CONFUSION_BLOCK_KINDS:
            return False
        if parking_template is None or parking_score < CAMERA_SIGN_PARKING_CONFUSION_MIN_SCORE:
            return False
        if parking_score < best_score - CAMERA_SIGN_PARKING_CONFUSION_MARGIN:
            return False

        if not self.parking_scene_sign_faces_car():
            return False

        segment = self.track.current_segment
        signature = (segment.name, best_template.name)
        if signature != self.last_camera_parking_confusion_guard_signature:
            self.get_logger().info(
                "camera sign ignored as parking/turn confusion; "
                f"segment={segment.name} "
                f"best={best_template.name}:{best_score:.2f} "
                f"parking={parking_template.name}:{parking_score:.2f}"
            )
            self.last_camera_parking_confusion_guard_signature = signature
        return True

    def update_camera_instruction_from_frame(self, frame: np.ndarray, stamp_ns: int) -> None:
        detection = self.detect_camera_instruction(frame)
        if detection is None:
            if self.camera_instruction_detection is not None and stamp_ns - self.last_camera_instruction_ns > self.camera_sign_timeout_ns:
                self.camera_instruction_detection = None
                self.last_visible_instruction_name = None
                self.camera_instruction_stable_name = None
                self.camera_instruction_stable_count = 0
            self.publish_camera_sign_detection(stamp_ns)
            return

        self.camera_instruction_detection = detection
        self.last_camera_instruction_ns = stamp_ns
        self.last_visible_instruction_name = detection.name
        if detection.name == self.camera_instruction_stable_name:
            self.camera_instruction_stable_count += 1
        else:
            self.camera_instruction_stable_name = detection.name
            self.camera_instruction_stable_count = 1

        if detection.name != self.last_camera_instruction_log_name:
            self.get_logger().info(
                f"camera_sign={detection.name} source=camera kind={detection.kind} "
                f"maneuver={detection.maneuver or '-'} score={detection.score:.2f} "
                f"area={detection.area_px}px bbox={detection.width_px}x{detection.height_px}px "
                f"center=({detection.center_x},{detection.center_y})"
            )
            self.last_camera_instruction_log_name = detection.name

        required_stable_frames = self.camera_instruction_required_stable_frames(detection)
        stable = self.camera_instruction_stable_count >= required_stable_frames
        apply_key = self.camera_instruction_apply_key(detection)
        if stable and apply_key != self.last_applied_instruction_key:
            if self.apply_instruction(detection):
                self.last_applied_instruction_name = detection.name
                self.last_applied_instruction_key = apply_key
        elif stable:
            self.apply_continuous_instruction(detection)

        self.publish_camera_sign_detection(stamp_ns)

    def camera_instruction_required_stable_frames(self, detection: CameraInstructionDetection) -> int:
        if detection.maneuver in {"left", "right", "straight"}:
            return max(self.camera_sign_stable_frames, CAMERA_SIGN_MANEUVER_STABLE_FRAMES)
        return self.camera_sign_stable_frames

    def camera_instruction_apply_key(self, detection: CameraInstructionDetection) -> tuple[str, str]:
        if detection.maneuver is not None:
            return detection.name, self.track.current_segment.name
        return detection.name, ""

    def publish_camera_sign_detection(self, stamp_ns: int) -> None:
        detection = self.camera_instruction_detection
        payload = {
            "stamp_ns": stamp_ns,
            "visible": detection is not None and stamp_ns - self.last_camera_instruction_ns <= self.camera_sign_timeout_ns,
            "source": "camera",
            "stable_count": self.camera_instruction_stable_count,
            "last_applied": self.last_applied_instruction_name,
        }
        if detection is not None:
            required_stable_frames = self.camera_instruction_required_stable_frames(detection)
            payload.update(
                {
                    "name": detection.name,
                    "kind": detection.kind,
                    "maneuver": detection.maneuver,
                    "score": detection.score,
                    "area_px": detection.area_px,
                    "bbox_px": [detection.width_px, detection.height_px],
                    "center": [detection.center_x, detection.center_y],
                    "required_stable_frames": required_stable_frames,
                    "stable": self.camera_instruction_stable_count >= required_stable_frames,
                }
            )
        else:
            payload["required_stable_frames"] = self.camera_sign_stable_frames
            payload["stable"] = False
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self.camera_sign_pub.publish(message)

    def camera_instruction_ready_for_action(self, detection: CameraInstructionDetection) -> bool:
        if not self.camera_scene_sign_context_allows_detection(detection):
            return False
        if detection.kind == "parking_sign":
            return detection.area_px >= self.camera_sign_parking_min_area_px and self.parking_camera_route_context_allows_arm()
        if detection.kind == "speed_limit_sign":
            return detection.area_px >= self.camera_sign_action_min_area_px
        if detection.maneuver is not None:
            return (
                detection.area_px >= self.camera_sign_action_min_area_px
                and self.camera_instruction_center_allows_maneuver(detection)
                and self.maneuver_camera_route_context_allows_action(detection.maneuver)
            )
        return detection.area_px >= self.camera_sign_action_min_area_px

    def camera_instruction_center_allows_maneuver(self, detection: CameraInstructionDetection) -> bool:
        if detection.frame_width <= 0:
            return False
        center_ratio = detection.center_x / float(detection.frame_width)
        return CAMERA_SIGN_MANEUVER_CENTER_MIN_RATIO <= center_ratio <= CAMERA_SIGN_MANEUVER_CENTER_MAX_RATIO

    def camera_turn_left_triggers_parking(self, detection: CameraInstructionDetection) -> bool:
        return False

    def maneuver_camera_route_context_allows_action(self, maneuver: str | None = None) -> bool:
        if self.camera_parking_armed_name is not None or self.track.pending_maneuver == PARKING_MANEUVER:
            return False
        if self.track.stopped_at_terminal or self.track.current_segment.parking_route:
            return False
        if maneuver is None:
            return False
        if self.nearby_maneuver_projection(maneuver) is not None:
            return True
        current_route_matches, _ = self.track.current_route_matches_maneuver(maneuver)
        if current_route_matches:
            return True
        if self.scene_maneuver_branch_allows_action(maneuver):
            return True
        return self.track.current_segment_allows_maneuver_action()

    def parking_approach_maneuver_guard_blocks(self, maneuver: str | None) -> bool:
        return False

    def current_motion_for_maneuver(self) -> tuple[float, float, float]:
        if self.drive_wheels and self.last_actual_sample is not None:
            return self.last_actual_sample.x, self.last_actual_sample.y, self.last_actual_sample.yaw
        return self.track.sample()

    def nearby_maneuver_projection(
        self,
        maneuver: str,
        max_distance_m: float | None = None,
    ) -> tuple[RouteProjection, float] | None:
        x, y, yaw = self.current_motion_for_maneuver()
        return self.track.nearby_maneuver_projection(maneuver, x, y, yaw, max_distance_m)

    def parking_camera_route_context_allows_arm(self) -> bool:
        segment = self.track.current_segment
        if self.track.stopped_at_terminal or self.track.parking_started or segment.parking_route:
            return False
        parking_branch_distance = self.track.distance_to_next_parking_branch(self.sign_detection_max_distance_m)
        if segment.inner_route and parking_branch_distance is not None:
            return True
        if not self.parking_scene_sign_faces_car():
            return False
        return self.parking_auto_route_enabled

    def current_scene_context_pose(self) -> tuple[float, float, float]:
        if self.drive_wheels and self.last_actual_sample is not None:
            return self.last_actual_sample.x, self.last_actual_sample.y, self.last_actual_sample.yaw
        x, y, route_yaw = self.track.sample()
        return x, y, route_yaw + self.yaw_offset

    def scene_sign_faces_car(
        self,
        sign_pose: SceneSignPose,
        max_distance_m: float,
        min_face_dot: float,
        min_forward_m: float = 0.0,
    ) -> bool:
        car_x, car_y, car_yaw = self.current_scene_context_pose()
        car_forward_x = math.cos(car_yaw)
        car_forward_y = math.sin(car_yaw)
        car_left_x = -math.sin(car_yaw)
        car_left_y = math.cos(car_yaw)

        sign_dx = sign_pose.x - car_x
        sign_dy = sign_pose.y - car_y
        sign_distance = math.hypot(sign_dx, sign_dy)
        if sign_distance <= 1e-6 or sign_distance > max_distance_m:
            return False

        forward_m = sign_dx * car_forward_x + sign_dy * car_forward_y
        if forward_m < min_forward_m:
            return False
        lateral_m = sign_dx * car_left_x + sign_dy * car_left_y
        angle_rad = math.atan2(lateral_m, max(1e-6, forward_m))
        if abs(angle_rad) > self.sign_detection_fov_rad * 0.5:
            return False

        face_yaw = sign_pose.yaw + self.track.parking_sign_face_local_yaw_offset_rad
        face_x = math.cos(face_yaw)
        face_y = math.sin(face_yaw)
        to_car_x = (car_x - sign_pose.x) / sign_distance
        to_car_y = (car_y - sign_pose.y) / sign_distance
        return face_x * to_car_x + face_y * to_car_y >= min_face_dot

    def scene_sign_is_in_camera_zone(
        self,
        sign_pose: SceneSignPose,
        max_distance_m: float,
        min_forward_m: float = 0.0,
    ) -> bool:
        car_x, car_y, car_yaw = self.current_scene_context_pose()
        car_forward_x = math.cos(car_yaw)
        car_forward_y = math.sin(car_yaw)
        car_left_x = -math.sin(car_yaw)
        car_left_y = math.cos(car_yaw)

        sign_dx = sign_pose.x - car_x
        sign_dy = sign_pose.y - car_y
        sign_distance = math.hypot(sign_dx, sign_dy)
        if sign_distance <= 1e-6 or sign_distance > max_distance_m:
            return False

        forward_m = sign_dx * car_forward_x + sign_dy * car_forward_y
        if forward_m < min_forward_m:
            return False
        lateral_m = sign_dx * car_left_x + sign_dy * car_left_y
        angle_rad = math.atan2(lateral_m, max(1e-6, forward_m))
        return abs(angle_rad) <= self.sign_detection_fov_rad * 0.5

    def scene_sign_is_visible_for_camera_context(
        self,
        sign_pose: SceneSignPose,
        max_distance_m: float,
        min_forward_m: float = 0.0,
    ) -> bool:
        if sign_pose.kind in CAMERA_SIGN_MANEUVER_KINDS or sign_pose.kind == "parking_sign":
            return self.scene_sign_is_in_camera_zone(sign_pose, max_distance_m, min_forward_m)
        return self.scene_sign_faces_car(
            sign_pose,
            max_distance_m,
            self.track.parking_sign_face_min_dot,
            min_forward_m,
        )

    def parking_scene_sign_faces_car(self) -> bool:
        if not self.parking_sign_scene_poses:
            return False

        max_distance_m = max(self.track.parking_sign_face_max_distance_m, self.sign_detection_max_distance_m)
        return any(
            self.scene_sign_is_visible_for_camera_context(
                sign_pose,
                max_distance_m,
                self.sign_detection_min_forward_m,
            )
            for sign_pose in self.parking_sign_scene_poses.values()
        )

    def camera_scene_preferred_sign_kind(self, allowed_kinds: frozenset[str] | None = None) -> str | None:
        if not self.camera_sign_scene_poses:
            return None

        car_x, car_y, _ = self.current_scene_context_pose()
        best: tuple[float, str] | None = None
        for sign_pose in self.camera_sign_scene_poses.values():
            if sign_pose.kind is None:
                continue
            if allowed_kinds is not None and sign_pose.kind not in allowed_kinds:
                continue
            if not self.scene_sign_is_visible_for_camera_context(
                sign_pose,
                self.sign_detection_max_distance_m,
                self.sign_detection_min_forward_m,
            ):
                continue
            distance_m = math.hypot(sign_pose.x - car_x, sign_pose.y - car_y)
            if best is None or distance_m < best[0]:
                best = (distance_m, sign_pose.kind)

        return None if best is None else best[1]

    def nearest_facing_scene_sign(
        self,
        kind: str,
        max_distance_m: float | None = None,
    ) -> tuple[SceneSignPose, float] | None:
        if not self.camera_sign_scene_poses:
            return None

        if max_distance_m is None:
            max_distance_m = self.sign_detection_max_distance_m
        car_x, car_y, _ = self.current_scene_context_pose()
        best: tuple[SceneSignPose, float] | None = None
        for sign_pose in self.camera_sign_scene_poses.values():
            if sign_pose.kind != kind:
                continue
            if not self.scene_sign_is_visible_for_camera_context(
                sign_pose,
                max_distance_m,
                self.sign_detection_min_forward_m,
            ):
                continue
            distance_m = math.hypot(sign_pose.x - car_x, sign_pose.y - car_y)
            if best is None or distance_m < best[1]:
                best = (sign_pose, distance_m)
        return best

    def nearest_facing_scene_sign_for_detection(
        self,
        detection: CameraInstructionDetection,
        max_distance_m: float | None = None,
    ) -> tuple[SceneSignPose, float] | None:
        return self.nearest_facing_scene_sign(detection.kind, max_distance_m)

    def scene_maneuver_branch_allows_action(self, maneuver: str) -> bool:
        kind = CAMERA_SIGN_MANEUVER_KIND_BY_ACTION.get(maneuver)
        if kind is None:
            return False
        if self.nearest_facing_scene_sign(kind) is None:
            return False
        return self.track.distance_to_next_maneuver_branch(self.sign_detection_max_distance_m) is not None

    def camera_scene_sign_context_allows_detection(self, detection: CameraInstructionDetection) -> bool:
        if not self.camera_sign_scene_poses:
            return True

        facing_kinds: set[str] = set()
        for sign_pose in self.camera_sign_scene_poses.values():
            if not self.scene_sign_is_visible_for_camera_context(
                sign_pose,
                self.sign_detection_max_distance_m,
                self.sign_detection_min_forward_m,
            ):
                continue
            if sign_pose.kind is None:
                continue
            if sign_pose.kind == detection.kind:
                return True
            facing_kinds.add(sign_pose.kind)

        if not facing_kinds:
            return True

        if detection.kind not in CAMERA_SIGN_MANEUVER_KINDS:
            return True
        facing_maneuver_kinds = facing_kinds & CAMERA_SIGN_MANEUVER_KINDS
        if not facing_maneuver_kinds or detection.kind in facing_maneuver_kinds:
            return True

        scene_kinds = tuple(sorted(facing_maneuver_kinds))
        signature = (detection.kind, scene_kinds)
        if signature != self.last_camera_scene_context_guard_signature:
            self.get_logger().info(
                "camera sign ignored because scene sign context disagrees; "
                f"detected={detection.kind} scene_ahead={','.join(scene_kinds)}"
            )
            self.last_camera_scene_context_guard_signature = signature
        return False

    def parking_camera_route_context_allows_action(self) -> bool:
        if not self.parking_auto_route_enabled:
            return False
        if self.track.stopped_at_terminal or self.track.parking_started or self.track.current_segment.parking_route:
            return False
        if self.track.parking_semantic_route_enabled and self.track.current_segment.inner_route:
            return True
        if self.track.current_segment.name not in self.track.parking_entry_segments:
            return False
        current_x, current_y, _ = self.track.sample()
        entry_x, entry_y = map_point_to_model(
            *self.track.parking_orthogonal_tail_px[0],
            self.track.route_offset_x_m,
            self.track.route_offset_y_m,
        )
        return math.hypot(current_x - entry_x, current_y - entry_y) <= self.track.parking_entry_max_distance_m

    def apply_continuous_instruction(self, visible: CameraInstructionDetection) -> None:
        return

    def camera_pending_maneuver_conflict_blocks(self, visible: CameraInstructionDetection) -> bool:
        pending = self.track.pending_maneuver
        if pending not in {"left", "right", "straight"} or visible.maneuver not in {"left", "right", "straight"}:
            return False
        if pending == visible.maneuver:
            return True

        segment = self.track.current_segment
        source_name = self.track.pending_maneuver_source or "unknown"
        signature = (segment.name, pending, visible.maneuver, visible.name)
        if signature != self.last_camera_maneuver_conflict_guard_signature:
            self.get_logger().info(
                "camera maneuver ignored because another sign command is already queued; "
                f"pending='{pending}' source={source_name} "
                f"ignored='{visible.maneuver}' sign={visible.name} "
                f"next_branch={self.track.next_maneuver_branch_distance_text()}"
            )
            self.last_camera_maneuver_conflict_guard_signature = signature
        return True

    def apply_instruction(self, visible: CameraInstructionDetection) -> bool:
        if not self.camera_instruction_ready_for_action(visible):
            return False

        if visible.kind == "parking_sign":
            if self.track.parking_started or self.track.current_segment.parking_route or self.track.stopped_at_terminal:
                self.get_logger().info(f"parking instruction from camera sign {visible.name}; parking route already active.")
                return True
            parking_branch_distance = self.track.distance_to_next_parking_branch(self.sign_detection_max_distance_m)
            if parking_branch_distance is not None:
                self.camera_parking_armed_name = None
                self.track.set_pending_maneuver(PARKING_MANEUVER, visible.name)
                self.get_logger().info(
                    f"parking stub armed from camera sign {visible.name}; "
                    f"next_branch={parking_branch_distance:.2f}m"
                )
                return True
            if self.parking_camera_route_context_allows_action():
                return self.start_camera_parking_route(visible.name)

            self.camera_parking_armed_name = visible.name
            self.track.set_pending_maneuver(PARKING_MANEUVER, visible.name)
            self.get_logger().info(
                f"parking route armed from camera sign {visible.name}; "
                f"continuing on the outer route until the {self.track.parking_entry_label}."
            )
            return True

        if visible.maneuver is not None:
            if self.track.pending_maneuver == PARKING_MANEUVER:
                parking_branch_distance = self.track.distance_to_next_parking_branch(self.sign_detection_max_distance_m)
                if parking_branch_distance is not None:
                    signature = (
                        self.track.current_segment.name,
                        PARKING_MANEUVER,
                        visible.maneuver,
                        visible.name,
                    )
                    if signature != self.last_camera_maneuver_conflict_guard_signature:
                        self.get_logger().info(
                            "camera maneuver ignored because parking command is already queued; "
                            f"ignored='{visible.maneuver}' sign={visible.name} "
                            f"next_parking_branch={parking_branch_distance:.2f}m"
                        )
                        self.last_camera_maneuver_conflict_guard_signature = signature
                    return True
            x, y, yaw = self.current_motion_for_maneuver()
            maneuver_search_distance_m = self.track.nearby_maneuver_route_distance_m
            scene_sign = self.nearest_facing_scene_sign_for_detection(visible)
            current_route_matches, current_route_delta = self.track.current_route_matches_maneuver(visible.maneuver)
            if current_route_matches:
                delta_text = "-" if current_route_delta is None else f"{math.degrees(current_route_delta):+.1f}deg"
                self.get_logger().info(
                    f"camera maneuver '{visible.maneuver}' from camera sign {visible.name} "
                    f"matches current route; continuing segment={self.track.current_segment.name} "
                    f"delta={delta_text}"
                )
                return True
            if scene_sign is None:
                return False
            nearby = self.track.rejoin_nearby_maneuver(
                visible.maneuver,
                visible.name,
                x,
                y,
                yaw,
                maneuver_search_distance_m,
            )
            if nearby is not None:
                projection, delta = nearby
                self.last_segment_name = self.track.current_segment.name
                self.publish_route_trace(self.get_clock().now().nanoseconds, force=True)
                self.get_logger().info(
                    f"switched to nearest maneuver '{visible.maneuver}' from camera sign {visible.name}; "
                    f"segment={projection.segment.name}@{projection.distance_on_segment:.3f}m "
                    f"route_dist={projection.distance_m:.3f}m "
                    f"search_radius={maneuver_search_distance_m:.2f}m "
                    f"delta={math.degrees(delta):+.1f}deg"
                )
                return True

            if self.camera_pending_maneuver_conflict_blocks(visible):
                return True
            if self.track.current_segment_allows_maneuver_action():
                self.track.set_pending_maneuver(visible.maneuver, visible.name)
                self.get_logger().info(
                    f"queued maneuver '{visible.maneuver}' from camera sign {visible.name}; "
                    f"next_branch={self.track.next_maneuver_branch_distance_text()}"
                )
                return True
            scene_branch_distance = self.track.distance_to_next_maneuver_branch(self.sign_detection_max_distance_m)
            if scene_sign is not None and scene_branch_distance is not None:
                sign_pose, sign_distance_m = scene_sign
                self.track.set_pending_maneuver(visible.maneuver, sign_pose.name)
                self.get_logger().info(
                    f"queued scene maneuver '{visible.maneuver}' from camera sign {visible.name}; "
                    f"scene_sign={sign_pose.name} sign_dist={sign_distance_m:.3f}m "
                    f"next_branch={scene_branch_distance:.2f}m"
                )
                return True
            if scene_sign is not None:
                sign_pose, sign_distance_m = scene_sign
                if sign_distance_m <= self.track.maneuver_max_branch_distance_m:
                    scene_nearby = self.track.rejoin_nearby_maneuver(
                        visible.maneuver,
                        sign_pose.name,
                        sign_pose.x,
                        sign_pose.y,
                        yaw,
                        maneuver_search_distance_m,
                    )
                else:
                    scene_nearby = None
                if scene_nearby is not None:
                    projection, delta = scene_nearby
                    self.last_segment_name = self.track.current_segment.name
                    self.publish_route_trace(self.get_clock().now().nanoseconds, force=True)
                    self.get_logger().info(
                        f"switched to sign-local maneuver '{visible.maneuver}' from camera sign {visible.name}; "
                        f"scene_sign={sign_pose.name} sign_dist={sign_distance_m:.3f}m "
                        f"segment={projection.segment.name}@{projection.distance_on_segment:.3f}m "
                        f"route_dist={projection.distance_m:.3f}m "
                        f"search_radius={maneuver_search_distance_m:.2f}m "
                        f"delta={math.degrees(delta):+.1f}deg"
                    )
                    return True

            return False

        if visible.kind == "speed_limit_sign":
            if visible.name in self.applied_speed_limit_names:
                return True
            self.pending_speed_limit_name = visible.name
            self.active_speed_mps = min(self.speed, self.speed_limit_mps)
            self.active_speed_limit_name = visible.name
            self.speed_limit_release_segment_name = self.track.current_segment.name
            self.speed_limit_release_distance_m = self.track.distance_on_segment + SPEED_LIMIT_PASS_MARGIN_M
            self.applied_speed_limit_names.add(visible.name)
            self.pending_speed_limit_name = None
            self.get_logger().info(
                f"camera speed limit sign {visible.name} applied; "
                f"speed set to {self.active_speed_mps:.2f}m/s "
                f"until {self.speed_limit_release_segment_name}@"
                f"{self.speed_limit_release_distance_m:.3f}m "
                f"score={visible.score:.2f} area={visible.area_px}px"
            )
            return True

        return False

    def start_camera_parking_route(self, source_name: str) -> bool:
        if not self.parking_auto_route_enabled:
            self.get_logger().info(
                f"parking instruction from camera sign {source_name}; auto parking route is disabled."
            )
            self.camera_parking_armed_name = None
            self.track.pending_maneuver = None
            self.track.pending_maneuver_source = None
            return False
        if self.drive_wheels and self.last_actual_sample is not None:
            current_yaw = self.last_actual_sample.yaw
        else:
            _, _, current_yaw = self.track.sample()
        started = self.track.start_parking_route(source_name, current_yaw)
        if started:
            self.last_segment_name = self.track.current_segment.name
            self.parking_stop_logged = False
            self.camera_parking_armed_name = None
            self.get_logger().info(
                f"parking route started from camera sign {source_name}; "
                f"planned a dynamic path from the {self.track.parking_entry_label} into the parking bay."
            )
            return True

        if self.track.parking_started or self.track.current_segment.parking_route or self.track.stopped_at_terminal:
            self.get_logger().info(f"parking instruction from camera sign {source_name}; parking route already active.")
            self.camera_parking_armed_name = None
            self.track.pending_maneuver = None
            self.track.pending_maneuver_source = None
            return True

        self.get_logger().warn(f"parking instruction from camera sign {source_name}; no safe parking entry path found.")
        self.track.pending_maneuver = None
        self.track.pending_maneuver_source = None
        return False

    def start_armed_camera_parking_if_ready(self) -> None:
        if self.camera_parking_armed_name is None:
            return
        if not self.parking_auto_route_enabled:
            self.camera_parking_armed_name = None
            if self.track.pending_maneuver != PARKING_MANEUVER:
                self.track.pending_maneuver = None
                self.track.pending_maneuver_source = None
            return
        if self.track.parking_started or self.track.current_segment.parking_route or self.track.stopped_at_terminal:
            self.camera_parking_armed_name = None
            self.track.pending_maneuver = None
            self.track.pending_maneuver_source = None
            return
        if not self.parking_camera_route_context_allows_action():
            return

        source_name = self.camera_parking_armed_name
        if not self.start_camera_parking_route(source_name):
            self.camera_parking_armed_name = None

    def handle_model_states(self, msg: ModelStates) -> None:
        name_to_index = {name: index for index, name in enumerate(msg.name)}
        camera_sign_scene_poses: dict[str, SceneSignPose] = {}
        parking_sign_scene_poses: dict[str, SceneSignPose] = {}
        for model_name, index in name_to_index.items():
            scene_identity = camera_sign_scene_identity_from_model_name(model_name)
            if scene_identity is None and not is_parking_sign_model_name(
                model_name,
                self.track.parking_sign_model_name_tokens,
            ):
                continue
            pose = msg.pose[index]
            _, _, yaw = quaternion_to_rpy(pose.orientation)
            kind, maneuver = scene_identity if scene_identity is not None else ("parking_sign", PARKING_MANEUVER)
            scene_pose = SceneSignPose(
                model_name,
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
                yaw,
                kind,
                maneuver,
            )
            camera_sign_scene_poses[model_name] = scene_pose
            if kind == "parking_sign":
                parking_sign_scene_poses[model_name] = scene_pose
        self.camera_sign_scene_poses = camera_sign_scene_poses
        self.parking_sign_scene_poses = parking_sign_scene_poses

        for model_name in self.traffic_light_scene_model_names:
            index = name_to_index.get(model_name)
            if index is None:
                continue
            pose = msg.pose[index]
            _, _, yaw = quaternion_to_rpy(pose.orientation)
            self.traffic_light_scene_poses[model_name] = (
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
                yaw,
            )

        try:
            index = name_to_index[self.entity_name]
        except KeyError:
            if not self.model_missing_logged:
                self.get_logger().warn(f"/model_states does not contain {self.entity_name} yet.")
                self.model_missing_logged = True
            return

        self.model_missing_logged = False
        pose = msg.pose[index]
        twist = msg.twist[index]
        roll, pitch, yaw = quaternion_to_rpy(pose.orientation)
        sample = MotionSample(
            self.get_clock().now().nanoseconds,
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
            roll,
            pitch,
            yaw,
            math.hypot(float(twist.linear.x), float(twist.linear.y)),
            float(twist.angular.z),
        )
        if self.drive_wheels:
            if self.reset_initial_pose_if_needed(sample):
                return
            if not self.guided_pose_synced:
                self.sync_wheel_route_start(sample)
            self.sync_route_progress_to_actual(sample)
            sample = self.rejoin_route_if_needed(sample) or sample
            self.inspect_actual_footprint(sample)
            self.inspect_actual_motion(sample)
            self.last_actual_sample = sample
            self.last_actual_command_sample = self.last_command_sample
            return

        if not self.guided_pose_synced:
            self.guided_pose_synced = True
            self.last_actual_sample = None
            self.last_actual_command_sample = None
            self.last_actual_xy_step = None

        self.inspect_actual_footprint(sample)
        self.inspect_actual_motion(sample)
        self.last_actual_sample = sample
        self.last_actual_command_sample = self.last_command_sample

    def reset_initial_pose_if_needed(self, sample: MotionSample) -> bool:
        if not self.initial_pose_reset_enabled:
            self.initial_pose_reset_done = True
            return False
        if self.initial_pose_reset_done:
            return False

        if self.initial_pose_reset_pending:
            if (
                self.initial_pose_reset_future is not None
                and not self.initial_pose_reset_future.done()
            ):
                self.publish_stop_command()
                return True
            if sample.stamp_ns < self.initial_pose_reset_settle_until_ns:
                self.publish_stop_command()
                return True
            self.initial_pose_reset_done = True
            self.initial_pose_reset_pending = False
            self.initial_pose_reset_future = None
            self.guided_pose_synced = False
            self.last_actual_sample = None
            self.last_actual_command_sample = None
            self.last_actual_xy_step = None
            return False

        route_x, route_y, route_yaw = self.track.sample()
        target_yaw = normalize_angle(route_yaw + self.yaw_offset)
        position_error = math.hypot(sample.x - route_x, sample.y - route_y)
        yaw_error = abs(normalize_angle(sample.yaw - target_yaw))
        if (
            position_error <= self.initial_pose_reset_position_tolerance_m
            and yaw_error <= self.initial_pose_reset_yaw_tolerance_rad
        ):
            self.initial_pose_reset_done = True
            return False

        if not self.route_rejoin_state_client.service_is_ready():
            self.route_rejoin_state_client.wait_for_service(timeout_sec=0.0)
            if not self.initial_pose_reset_wait_logged:
                self.get_logger().warn("Waiting for /set_entity_state before resetting the initial car pose.")
                self.initial_pose_reset_wait_logged = True
            self.publish_stop_command()
            return True

        state = self.make_entity_state(self.entity_name, route_x, route_y, self.z, target_yaw)
        request = SetEntityState.Request()
        request.state = state
        self.initial_pose_reset_future = self.route_rejoin_state_client.call_async(request)
        self.initial_pose_reset_future.add_done_callback(self.handle_initial_pose_reset_response)
        self.initial_pose_reset_pending = True
        self.initial_pose_reset_settle_until_ns = sample.stamp_ns + int(0.25 * 1e9)
        self.track.distance_since_branch = MIN_BRANCH_DISTANCE_M
        self.last_segment_name = self.track.current_segment.name
        self.publish_stop_command()
        self.last_actual_sample = None
        self.last_actual_command_sample = None
        self.last_actual_xy_step = None
        self.last_command_sample = MotionSample(
            sample.stamp_ns,
            route_x,
            route_y,
            self.z,
            0.0,
            0.0,
            target_yaw,
            0.0,
            0.0,
        )
        self.get_logger().info(
            "initial pose reset to configured route start; "
            f"actual=({sample.x:+.3f},{sample.y:+.3f}) yaw={sample.yaw:+.3f} "
            f"target=({route_x:+.3f},{route_y:+.3f}) yaw={target_yaw:+.3f} "
            f"error={position_error:.3f}m/{yaw_error:.3f}rad "
            f"segment={self.track.current_segment.name}"
        )
        return True

    def handle_initial_pose_reset_response(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warn(f"initial /set_entity_state failed: {exc}")
            self.initial_pose_reset_done = True
            self.initial_pose_reset_pending = False
            self.initial_pose_reset_future = None
            return

        if response is not None and not response.success:
            status_message = getattr(response, "status_message", "entity update was rejected")
            self.get_logger().warn(f"initial /set_entity_state rejected pose reset: {status_message}")

    def sync_wheel_route_start(self, sample: MotionSample) -> None:
        projection = self.track.nearest_pose_projection(sample.x, sample.y, sample.yaw, outer_only=True)
        startup_sync_max_distance_m = max(
            self.route_rejoin_max_route_distance_m,
            ROUTE_START_SYNC_MAX_ROUTE_DISTANCE_M,
        )
        if projection.distance_m <= startup_sync_max_distance_m:
            self.track.current_segment = projection.segment
            self.track.distance_on_segment = min(
                max(projection.distance_on_segment, 0.0),
                projection.segment.length,
            )
            self.track.distance_since_branch = MIN_BRANCH_DISTANCE_M
            self.last_segment_name = self.track.current_segment.name
            self.get_logger().info(
                "wheel controller synced route progress from initial model pose; "
                f"segment={projection.segment.name}@{projection.distance_on_segment:.3f}m "
                f"route_dist={projection.distance_m:.3f}m "
                f"startup_gate={startup_sync_max_distance_m:.3f}m"
            )
        else:
            self.get_logger().warn(
                "initial model pose is not close to the planned route; "
                f"route_dist={projection.distance_m:.3f}m. "
                "The wheel controller will drive toward the configured route start without teleporting."
            )
        self.guided_pose_synced = True
        self.skip_route_advance_once = True
        self.last_actual_sample = None
        self.last_actual_command_sample = None
        self.last_actual_xy_step = None

    def sync_route_progress_to_actual(self, sample: MotionSample) -> bool:
        if (
            not self.route_progress_sync_enabled
            or self.last_command_sample is None
            or self.track.stopped_at_terminal
        ):
            return False
        if sample.stamp_ns - self.last_route_progress_sync_ns < self.route_progress_sync_min_interval_ns:
            return False

        command_error = math.hypot(sample.x - self.last_command_sample.x, sample.y - self.last_command_sample.y)
        if command_error < self.route_progress_sync_command_error_m:
            return False

        projection = self.track.current_segment.project(sample.x, sample.y)
        yaw_error = abs(normalize_angle(projection.yaw + self.yaw_offset - sample.yaw))
        if (
            projection.distance_m > self.route_progress_sync_max_route_distance_m
            or yaw_error > self.route_progress_sync_max_yaw_error_rad
        ):
            return False

        old_segment_name = self.track.current_segment.name
        old_distance = self.track.distance_on_segment
        self.track.distance_on_segment = min(
            max(projection.distance_on_segment, 0.0),
            projection.segment.length,
        )
        self.track.stopped_at_terminal = False
        if projection.segment.parking_route:
            self.track.parking_started = True
        self.track.last_decision_text = f"route/progress_sync: {projection.segment.name}"

        route_x, route_y, route_yaw = self.track.sample()
        route_yaw = normalize_angle(route_yaw + self.yaw_offset)
        self.last_command_sample = MotionSample(
            sample.stamp_ns,
            route_x,
            route_y,
            self.z,
            0.0,
            0.0,
            route_yaw,
            self.last_cmd_linear_mps,
            self.last_cmd_angular_rps,
        )
        self.last_route_progress_sync_ns = sample.stamp_ns

        if sample.stamp_ns - self.last_route_progress_sync_log_ns >= self.route_progress_sync_log_period_ns:
            self.get_logger().info(
                "ROUTE_PROGRESS_SYNC "
                f"actual=({sample.x:+.3f},{sample.y:+.3f}) "
                f"cmd_error={command_error:.3f}m route_dist={projection.distance_m:.3f}m "
                f"yaw_error={yaw_error:.3f}rad "
                f"segment={old_segment_name}@{old_distance:.3f}m"
                f"->{projection.segment.name}@{projection.distance_on_segment:.3f}m"
            )
            self.last_route_progress_sync_log_ns = sample.stamp_ns
        return True

    def rejoin_route_if_needed(self, sample: MotionSample) -> MotionSample | None:
        if not self.route_rejoin_enabled or self.last_command_sample is None:
            return None
        if sample.stamp_ns - self.last_route_rejoin_ns < self.route_rejoin_cooldown_ns:
            return None
        if self.route_rejoin_future is not None and not self.route_rejoin_future.done():
            return None

        command_error = math.hypot(sample.x - self.last_command_sample.x, sample.y - self.last_command_sample.y)
        if command_error < self.route_rejoin_command_error_m:
            return None
        if self.track.stopped_at_terminal:
            return None

        if not self.route_rejoin_state_client.service_is_ready():
            self.route_rejoin_state_client.wait_for_service(timeout_sec=0.0)
            if not self.route_rejoin_wait_logged:
                self.get_logger().warn(
                    "Waiting for /set_entity_state before route rejoin recovery can snap the car."
                )
                self.route_rejoin_wait_logged = True
            return None

        locked_to_parking = self.track.parking_started or self.track.current_segment.parking_route
        if locked_to_parking:
            projection = self.track.current_segment.project(sample.x, sample.y)
            if projection.distance_m > self.route_rejoin_max_route_distance_m:
                return None
            projection = self.track.rejoin_current_segment(sample.x, sample.y)
        else:
            rejoin_outer_only = self.route_rejoin_outer_only and not self.track.current_segment.inner_route
            projection = self.track.nearest_projection(sample.x, sample.y, rejoin_outer_only)
            if projection.distance_m > self.route_rejoin_max_route_distance_m:
                return None
            projection = self.track.rejoin_at_nearest(sample.x, sample.y, rejoin_outer_only)

        route_x, route_y, route_yaw = self.track.sample()
        yaw = normalize_angle(route_yaw + self.yaw_offset)
        state = self.make_entity_state(self.entity_name, route_x, route_y, self.z, yaw)
        request = SetEntityState.Request()
        request.state = state
        self.route_rejoin_future = self.route_rejoin_state_client.call_async(request)
        self.route_rejoin_future.add_done_callback(self.handle_route_rejoin_response)

        self.publish_stop_command()
        self.last_actual_sample = None
        self.last_actual_command_sample = None
        self.last_actual_xy_step = None
        self.last_segment_name = self.track.current_segment.name
        self.last_route_rejoin_ns = sample.stamp_ns

        rejoined_sample = MotionSample(
            sample.stamp_ns,
            route_x,
            route_y,
            self.z,
            0.0,
            0.0,
            yaw,
            0.0,
            0.0,
        )
        self.last_command_sample = rejoined_sample
        self.last_cmd_linear_mps = 0.0
        self.last_cmd_angular_rps = 0.0
        self.publish_route_trace(sample.stamp_ns, force=True)

        self.get_logger().info(
            f"ROUTE_REJOIN teleport actual=({sample.x:+.3f},{sample.y:+.3f}) "
            f"cmd_error={command_error:.3f}m nearest=({route_x:+.3f},{route_y:+.3f}) "
            f"route_dist={projection.distance_m:.3f}m "
            f"segment={projection.segment.name}@{projection.distance_on_segment:.3f}m "
            f"parking_locked={locked_to_parking} yaw={yaw:+.3f}"
        )
        return rejoined_sample

    def handle_route_rejoin_response(self, future) -> None:
        if future is self.route_rejoin_future:
            self.route_rejoin_future = None
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warn(f"/set_entity_state failed during route rejoin recovery: {exc}")
            return

        if response is not None and not response.success:
            status_message = getattr(response, "status_message", "entity update was rejected")
            self.get_logger().warn(
                f"/set_entity_state rejected route rejoin recovery: {status_message}"
            )

    def inspect_actual_footprint(self, sample: MotionSample) -> None:
        check = check_road_footprint(
            self.track.clearance,
            sample.x,
            sample.y,
            sample.yaw,
            self.footprint_front_m,
            self.footprint_rear_m,
            self.footprint_half_width_m,
            self.track.route_offset_x_m,
            self.track.route_offset_y_m,
        )
        self.last_actual_footprint_check = check
        if check.on_road or sample.stamp_ns - self.last_actual_footprint_log_ns < self.jitter_log_period_ns:
            return

        map_x, map_y = model_point_to_map(
            sample.x,
            sample.y,
            self.track.route_offset_x_m,
            self.track.route_offset_y_m,
        )
        command_error, command_yaw_error = self.command_error(sample)
        self.get_logger().warn(
            f"GAZEBO_FOOTPRINT_OFFROAD actual pose=({sample.x:+.4f},{sample.y:+.4f}) "
            f"yaw={sample.yaw:+.4f} origin_pixel=({map_x:.1f},{map_y:.1f}) "
            f"min_clearance={check.min_clearance_px:.2f}px "
            f"cmd_error={command_error:.4f}m/{command_yaw_error:.3f}rad"
        )
        self.last_actual_footprint_log_ns = sample.stamp_ns

    def inspect_actual_motion(self, sample: MotionSample) -> None:
        previous = self.last_actual_sample
        previous_command = self.last_actual_command_sample
        command = self.last_command_sample
        dt = 0.0
        xy_step = 0.0
        command_xy_step = 0.0
        z_step = 0.0
        yaw_step = 0.0
        command_yaw_step = 0.0
        position_residual = 0.0
        xy_step_delta = 0.0
        yaw_step_error = 0.0
        speed_delta = 0.0
        yaw_rate_delta = 0.0
        reasons: list[str] = []

        if previous is not None:
            dt = max(0.0, (sample.stamp_ns - previous.stamp_ns) * 1e-9)
            xy_step = math.hypot(sample.x - previous.x, sample.y - previous.y)
            z_step = abs(sample.z - previous.z)
            yaw_step = abs(normalize_angle(sample.yaw - previous.yaw))
            speed_delta = abs(sample.speed_xy - previous.speed_xy)
            yaw_rate_delta = abs(sample.yaw_rate - previous.yaw_rate)
            if command is not None and previous_command is not None:
                command_xy_step = math.hypot(command.x - previous_command.x, command.y - previous_command.y)
                command_yaw_step = abs(normalize_angle(command.yaw - previous_command.yaw))
                position_residual = abs(xy_step - command_xy_step)
                yaw_step_error = abs(yaw_step - command_yaw_step)
                if yaw_step_error > self.jitter_yaw_step_rad:
                    reasons.append(f"yaw_step_error={yaw_step_error:.3f}rad")
            if self.last_actual_xy_step is not None:
                xy_step_delta = abs(xy_step - self.last_actual_xy_step)
                if xy_step_delta > self.jitter_xy_step_delta_m:
                    reasons.append(f"xy_step_delta={xy_step_delta:.4f}m")
            if z_step > self.jitter_z_delta_m:
                reasons.append(f"z_delta={z_step:.4f}m")
            if speed_delta > self.jitter_speed_delta_mps:
                reasons.append(f"speed_delta={speed_delta:.3f}m/s")
            if yaw_rate_delta > self.jitter_yaw_rate_delta_rps:
                reasons.append(f"yaw_rate_delta={yaw_rate_delta:.3f}rad/s")

        tilt = max(abs(sample.roll), abs(sample.pitch))
        if tilt > self.jitter_tilt_rad:
            reasons.append(f"tilt={tilt:.4f}rad")

        command_error, command_yaw_error = self.command_error(sample)
        if command_error > self.jitter_command_error_m:
            reasons.append(f"cmd_xy_error={command_error:.4f}m")

        if reasons:
            self.log_jitter(
                sample,
                dt,
                xy_step,
                command_xy_step,
                z_step,
                yaw_step,
                command_yaw_step,
                position_residual,
                yaw_step_error,
                speed_delta,
                yaw_rate_delta,
                command_error,
                command_yaw_error,
                reasons,
            )
        self.log_telemetry(
            sample,
            dt,
            xy_step,
            command_xy_step,
            z_step,
            yaw_step,
            command_yaw_step,
            position_residual,
            yaw_step_error,
            command_error,
            command_yaw_error,
        )
        if previous is not None:
            self.last_actual_xy_step = xy_step

    def command_error(self, sample: MotionSample) -> tuple[float, float]:
        if self.last_command_sample is None:
            return 0.0, 0.0
        return (
            math.hypot(sample.x - self.last_command_sample.x, sample.y - self.last_command_sample.y),
            abs(normalize_angle(sample.yaw - self.last_command_sample.yaw)),
        )

    def log_telemetry(
        self,
        sample: MotionSample,
        dt: float,
        xy_step: float,
        command_xy_step: float,
        z_step: float,
        yaw_step: float,
        command_yaw_step: float,
        position_residual: float,
        yaw_step_error: float,
        command_error: float,
        command_yaw_error: float,
    ) -> None:
        if sample.stamp_ns - self.last_telemetry_log_ns < self.telemetry_period_ns:
            return

        command_text = "cmd=none"
        if self.last_command_sample is not None:
            command_text = (
                f"cmd pose=({self.last_command_sample.x:+.4f},{self.last_command_sample.y:+.4f},"
                f"{self.last_command_sample.z:+.4f}) yaw={self.last_command_sample.yaw:+.4f} "
                f"cmd_vel=({self.last_cmd_linear_mps:.3f}m/s,{self.last_cmd_angular_rps:+.3f}rad/s)"
            )
        footprint_text = "footprint=unknown"
        if self.last_actual_footprint_check is not None:
            footprint_state = "road" if self.last_actual_footprint_check.on_road else "OFFROAD"
            footprint_text = (
                f"footprint={footprint_state}:{self.last_actual_footprint_check.min_clearance_px:.2f}px"
            )
        if self.motion_log_verbose:
            self.get_logger().info(
                f"MOTION actual pose=({sample.x:+.4f},{sample.y:+.4f},{sample.z:+.4f}) "
                f"rpy=({sample.roll:+.4f},{sample.pitch:+.4f},{sample.yaw:+.4f}) "
                f"speed={sample.speed_xy:.4f}m/s yaw_rate={sample.yaw_rate:+.3f}rad/s; "
                f"step dt={dt:.3f}s xy={xy_step:.4f}m z={z_step:.4f}m yaw={yaw_step:.4f}rad "
                f"cmd_xy={command_xy_step:.4f}m cmd_yaw={command_yaw_step:.4f}rad "
                f"step_error xy={position_residual:.4f}m yaw={yaw_step_error:.4f}rad; {command_text}; "
                f"tracking_error xy={command_error:.4f}m yaw={command_yaw_error:.4f}rad; "
                f"{footprint_text}; "
                f"route={self.track.current_segment.name}@{self.track.distance_on_segment:.3f}m; "
                f"cmd_vel sent={self.sent_cmd_count}"
            )
        else:
            self.get_logger().info(
                f"MOTION pos=({sample.x:+.3f},{sample.y:+.3f}) yaw={sample.yaw:+.3f} "
                f"v={sample.speed_xy:.3f} step_err={position_residual:.4f}m/{yaw_step_error:.3f}rad "
                f"track_err={command_error:.4f}m {footprint_text} route={self.track.current_segment.name} "
                f"cmd_vel=({self.last_cmd_linear_mps:.3f},{self.last_cmd_angular_rps:+.3f})"
            )
        self.last_telemetry_log_ns = sample.stamp_ns

    def log_jitter(
        self,
        sample: MotionSample,
        dt: float,
        xy_step: float,
        command_xy_step: float,
        z_step: float,
        yaw_step: float,
        command_yaw_step: float,
        position_residual: float,
        yaw_step_error: float,
        speed_delta: float,
        yaw_rate_delta: float,
        command_error: float,
        command_yaw_error: float,
        reasons: list[str],
    ) -> None:
        if sample.stamp_ns - self.last_jitter_log_ns < self.jitter_log_period_ns:
            self.suppressed_jitter_count += 1
            return

        event_count = self.suppressed_jitter_count + 1
        if self.motion_log_verbose:
            self.get_logger().warn(
                f"JITTER events={event_count} reasons={','.join(reasons)}; "
                f"actual pose=({sample.x:+.5f},{sample.y:+.5f},{sample.z:+.5f}) "
                f"rpy=({sample.roll:+.5f},{sample.pitch:+.5f},{sample.yaw:+.5f}) "
                f"speed={sample.speed_xy:.4f}m/s yaw_rate={sample.yaw_rate:+.3f}rad/s; "
                f"step dt={dt:.3f}s xy={xy_step:.5f}m z={z_step:.5f}m yaw={yaw_step:.5f}rad "
                f"cmd_xy={command_xy_step:.5f}m cmd_yaw={command_yaw_step:.5f}rad "
                f"error xy={position_residual:.5f}m yaw={yaw_step_error:.5f}rad "
                f"speed_delta={speed_delta:.4f}m/s "
                f"yaw_rate_delta={yaw_rate_delta:.3f}rad/s; "
                f"cmd_error xy={command_error:.5f}m yaw={command_yaw_error:.5f}rad; "
                f"route={self.track.current_segment.name}@{self.track.distance_on_segment:.3f}m"
            )
        else:
            self.get_logger().warn(
                f"JITTER events={event_count} reasons={','.join(reasons)} "
                f"pos=({sample.x:+.3f},{sample.y:+.3f}) "
                f"step={xy_step:.4f}/{command_xy_step:.4f}m "
                f"yaw={yaw_step:.3f}/{command_yaw_step:.3f}rad "
                f"route={self.track.current_segment.name}"
            )
        self.last_jitter_log_ns = sample.stamp_ns
        self.suppressed_jitter_count = 0


def main() -> None:
    rclpy.init()
    node = GazeboTrackCar()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
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
