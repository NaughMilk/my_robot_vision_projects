from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


YELLOW_BGR = (0, 220, 255)
FLOOR_BGR = (28, 31, 36)
GRID_BGR = (42, 47, 54)
ACCENT_BGR = (255, 200, 90)
TEXT_BGR = (235, 240, 245)
MAP_VISUAL_NAME = "autonomous_map_visual.png"


@dataclass
class RobotPose:
    x: float
    y: float
    heading: float
    linear: float = 0.0
    angular: float = 0.0


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def heading_vectors(angle: float) -> tuple[np.ndarray, np.ndarray]:
    forward = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
    right = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float32)
    return forward, right


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def share_file(*parts: str) -> Path | None:
    try:
        from ament_index_python.packages import get_package_share_directory

        share_path = Path(get_package_share_directory("my_robot_vision")).joinpath(*parts)
        if share_path.exists():
            return share_path
    except Exception:
        pass

    source_path = package_root().joinpath(*parts)
    if source_path.exists():
        return source_path
    return None


def autonomous_map_visual_path() -> Path | None:
    return share_file("maps", MAP_VISUAL_NAME)


def autonomous_map_available() -> bool:
    return autonomous_map_visual_path() is not None


def build_generated_track_points(width: int = 2000, height: int = 1200) -> np.ndarray:
    xs = np.linspace(160, width - 180, 280)
    ys = (
        height * 0.52
        + 165.0 * np.sin(xs / 180.0)
        + 60.0 * np.sin(xs / 58.0)
        - 35.0 * np.cos(xs / 92.0)
    )
    ys = np.clip(ys, 180, height - 180)
    return np.round(np.column_stack([xs, ys])).astype(np.int32)


def autonomous_map_layout(width: int = 2000, height: int = 1200) -> tuple[tuple[int, int, int], tuple[int, int, int, int], int]:
    map_size = int(min(height * 0.82, width * 0.48))
    x0 = (width - map_size) // 2
    y0 = (height - map_size) // 2

    route_margin = int(map_size * 0.14)
    route_left = x0 + route_margin
    route_top = y0 + route_margin
    route_right = x0 + map_size - route_margin
    route_bottom = y0 + map_size - route_margin
    line_width = max(40, int(map_size * 0.05))

    return (x0, y0, map_size), (route_left, route_top, route_right, route_bottom), line_width


def rounded_rect_track_points(
    left: int,
    top: int,
    right: int,
    bottom: int,
    radius: int,
    samples_per_arc: int = 40,
    samples_per_edge: int = 44,
) -> np.ndarray:
    points: list[tuple[float, float]] = []

    def append_edge(start: tuple[float, float], end: tuple[float, float]) -> None:
        xs = np.linspace(start[0], end[0], samples_per_edge, endpoint=False)
        ys = np.linspace(start[1], end[1], samples_per_edge, endpoint=False)
        points.extend(zip(xs, ys))

    def append_arc(center: tuple[float, float], start_deg: float, end_deg: float) -> None:
        angles = np.linspace(math.radians(start_deg), math.radians(end_deg), samples_per_arc, endpoint=False)
        for angle in angles:
            points.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))

    append_edge((left + radius, top), (right - radius, top))
    append_arc((right - radius, top + radius), -90.0, 0.0)
    append_edge((right, top + radius), (right, bottom - radius))
    append_arc((right - radius, bottom - radius), 0.0, 90.0)
    append_edge((right - radius, bottom), (left + radius, bottom))
    append_arc((left + radius, bottom - radius), 90.0, 180.0)
    append_edge((left, bottom - radius), (left, top + radius))
    append_arc((left + radius, top + radius), 180.0, 270.0)
    points.append((left + radius, top))

    return np.round(np.array(points, dtype=np.float32)).astype(np.int32)


def build_autonomous_map_track_points(width: int = 2000, height: int = 1200) -> np.ndarray:
    _, (left, top, right, bottom), line_width = autonomous_map_layout(width, height)
    radius = max(line_width + 12, int((right - left) * 0.14))
    return rounded_rect_track_points(left, top, right, bottom, radius)


def build_track_points(width: int = 2000, height: int = 1200) -> np.ndarray:
    if autonomous_map_available():
        return build_autonomous_map_track_points(width, height)
    return build_generated_track_points(width, height)


def draw_static_world(
    track_points: np.ndarray,
    width: int = 2000,
    height: int = 1200,
    line_width: int = 110,
) -> np.ndarray:
    world = np.full((height, width, 3), FLOOR_BGR, dtype=np.uint8)

    for x in range(0, width, 80):
        cv2.line(world, (x, 0), (x, height), GRID_BGR, 1, cv2.LINE_AA)
    for y in range(0, height, 80):
        cv2.line(world, (0, y), (width, y), GRID_BGR, 1, cv2.LINE_AA)

    if autonomous_map_available():
        (x0, y0, map_size), _, map_line_width = autonomous_map_layout(width, height)
        map_visual = cv2.imread(str(autonomous_map_visual_path()))
        if map_visual is not None:
            map_visual = cv2.resize(map_visual, (map_size, map_size), interpolation=cv2.INTER_NEAREST)
            world[y0 : y0 + map_size, x0 : x0 + map_size] = map_visual
            line_width = map_line_width

    cv2.polylines(world, [track_points], False, (0, 150, 180), line_width + 18, cv2.LINE_AA)
    cv2.polylines(world, [track_points], False, YELLOW_BGR, line_width, cv2.LINE_AA)
    return world


def make_start_pose(track_points: np.ndarray, line_width: int | None = None) -> RobotPose:
    if line_width is None and autonomous_map_available():
        _, _, line_width = autonomous_map_layout()
    if line_width is None:
        line_width = 110

    track_f = track_points.astype(np.float32)
    start_point = track_f[8]
    next_point = track_f[14]
    heading = math.atan2(float(next_point[1] - start_point[1]), float(next_point[0] - start_point[0]))
    _, right = heading_vectors(heading)
    start_xy = start_point
    return RobotPose(float(start_xy[0]), float(start_xy[1]), heading)


def camera_trapezoid(pose: RobotPose) -> np.ndarray:
    forward, right = heading_vectors(pose.heading)
    robot_xy = np.array([pose.x, pose.y], dtype=np.float32)
    origin = robot_xy + forward * 26.0

    near_center = origin + forward * 48.0
    far_center = origin + forward * 330.0
    near_half_width = 125.0
    far_half_width = 245.0

    return np.array(
        [
            near_center - right * near_half_width,
            near_center + right * near_half_width,
            far_center + right * far_half_width,
            far_center - right * far_half_width,
        ],
        dtype=np.float32,
    )


def render_camera(
    world: np.ndarray,
    pose: RobotPose,
    frame_width: int = 420,
    frame_height: int = 260,
) -> tuple[np.ndarray, np.ndarray]:
    src = camera_trapezoid(pose)
    dst = np.array(
        [
            [0, frame_height - 1],
            [frame_width - 1, frame_height - 1],
            [frame_width - 1, 0],
            [0, 0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    frame = cv2.warpPerspective(
        world,
        matrix,
        (frame_width, frame_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=FLOOR_BGR,
    )
    return frame, src


def update_pose(
    pose: RobotPose,
    linear_cmd: float,
    angular_cmd: float,
    dt: float,
    speed_scale: float = 135.0,
) -> None:
    pose.linear += 0.28 * (linear_cmd - pose.linear)
    pose.angular += 0.36 * (angular_cmd - pose.angular)
    pose.heading += pose.angular * dt

    forward, _ = heading_vectors(pose.heading)
    pose.x += float(forward[0] * pose.linear * speed_scale * dt)
    pose.y += float(forward[1] * pose.linear * speed_scale * dt)


def nearest_track_point(track_points: np.ndarray, pose: RobotPose) -> tuple[int, np.ndarray, float]:
    track_f = track_points.astype(np.float32)
    pose_xy = np.array([pose.x, pose.y], dtype=np.float32)
    delta = track_f - pose_xy
    d2 = np.sum(delta * delta, axis=1)
    idx = int(np.argmin(d2))
    return idx, track_f[idx], float(math.sqrt(float(d2[idx])))


def draw_robot(
    image: np.ndarray,
    pose: RobotPose,
    color: tuple[int, int, int] = (80, 250, 120),
) -> None:
    forward, right = heading_vectors(pose.heading)
    center = np.array([pose.x, pose.y], dtype=np.float32)
    nose = center + forward * 30.0
    tail_left = center - forward * 20.0 - right * 14.0
    tail_right = center - forward * 20.0 + right * 14.0
    robot_shape = np.round(np.array([nose, tail_left, tail_right])).astype(np.int32)
    cv2.fillConvexPoly(image, robot_shape, color, cv2.LINE_AA)
    cv2.circle(image, (int(pose.x), int(pose.y)), 5, (255, 255, 255), -1)
