from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from my_robot_vision.gazebo_track_car import (  # noqa: E402
    DEFAULT_CLEARANCE_PX,
    MAP_TEXTURE_SIZE,
    MODEL_SIZE_M,
    DirectedTrack,
    load_map_image,
    model_point_to_map,
)


DEBUG_PATH = "/tmp/gazebo_straight_jitter_debug.png"


@dataclass
class StraightReport:
    segment: str
    samples: int
    length_m: float
    residual_range_px: float
    max_jump_px: float
    anomaly_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find small lateral jumps on straight Gazebo route segments."
    )
    parser.add_argument("--step-m", type=float, default=0.01)
    parser.add_argument("--min-length-m", type=float, default=0.45)
    parser.add_argument("--turn-trim-m", type=float, default=0.12)
    parser.add_argument("--jump-threshold-px", type=float, default=0.65)
    parser.add_argument("--range-threshold-px", type=float, default=1.25)
    parser.add_argument("--clearance-px", type=int, default=DEFAULT_CLEARANCE_PX)
    direction_group = parser.add_mutually_exclusive_group()
    direction_group.add_argument("--reverse-direction", dest="reverse_direction", action="store_true")
    direction_group.add_argument("--no-reverse-direction", dest="reverse_direction", action="store_false")
    parser.set_defaults(reverse_direction=True)
    parser.add_argument("--debug-path", default=DEBUG_PATH)
    return parser.parse_args()


def segment_is_straight(segment_name: str, points: np.ndarray, min_length_m: float) -> bool:
    if "corner" in segment_name or len(points) < 4:
        return False

    delta = points[-1] - points[0]
    length = float(np.linalg.norm(delta))
    if length < min_length_m:
        return False

    tangent = delta / max(length, 1e-9)
    axis_alignment = max(abs(float(tangent[0])), abs(float(tangent[1])))
    return axis_alignment > 0.985


def lateral_residual_px(distances: np.ndarray, points: np.ndarray) -> np.ndarray:
    delta = points[-1] - points[0]
    length = float(np.linalg.norm(delta))
    tangent = delta / max(length, 1e-9)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    lateral_m = (points - points[0]) @ normal
    if len(distances) >= 2:
        slope, intercept = np.polyfit(distances, lateral_m, 1)
        lateral_m = lateral_m - (slope * distances + intercept)
    return lateral_m * MAP_TEXTURE_SIZE / MODEL_SIZE_M


def analyze_segment(
    segment_name: str,
    samples: list[tuple[float, float, float]],
    args: argparse.Namespace,
) -> tuple[StraightReport | None, list[tuple[float, float, float]]]:
    if len(samples) < 4:
        return None, []

    distances = np.array([sample[0] for sample in samples], dtype=np.float64)
    points = np.array([[sample[1], sample[2]] for sample in samples], dtype=np.float64)
    if not segment_is_straight(segment_name, points, args.min_length_m):
        return None, []

    trim_mask = (distances >= args.turn_trim_m) & (distances <= distances[-1] - args.turn_trim_m)
    if np.count_nonzero(trim_mask) < 4:
        return None, []

    trimmed_distances = distances[trim_mask]
    trimmed_points = points[trim_mask]
    residual = lateral_residual_px(trimmed_distances, trimmed_points)
    jumps = np.abs(np.diff(residual))
    residual_range = float(np.max(residual) - np.min(residual))
    max_jump = float(np.max(jumps)) if len(jumps) else 0.0
    anomaly_indexes = np.flatnonzero(
        (np.abs(residual) > args.range_threshold_px)
        | (np.r_[False, jumps > args.jump_threshold_px])
    )
    anomaly_points = [
        (float(trimmed_distances[index]), float(trimmed_points[index, 0]), float(trimmed_points[index, 1]))
        for index in anomaly_indexes
    ]
    report = StraightReport(
        segment=segment_name,
        samples=int(np.count_nonzero(trim_mask)),
        length_m=float(distances[-1]),
        residual_range_px=residual_range,
        max_jump_px=max_jump,
        anomaly_count=len(anomaly_points),
    )
    return report, anomaly_points


def draw_debug(
    image: np.ndarray,
    route_points: list[tuple[float, float]],
    anomaly_points: list[tuple[float, float]],
    path: str,
) -> None:
    overlay = image.copy()
    if route_points:
        route_px = np.array(
            [model_point_to_map(x, y) for x, y in route_points],
            dtype=np.float32,
        )
        cv2.polylines(overlay, [np.round(route_px).astype(np.int32)], False, (255, 80, 0), 1, cv2.LINE_AA)

    for x, y in anomaly_points:
        px, py = model_point_to_map(x, y)
        cv2.circle(overlay, (int(round(px)), int(round(py))), 2, (0, 0, 255), -1, cv2.LINE_AA)

    cv2.imwrite(
        path,
        cv2.resize(overlay, (int(MAP_TEXTURE_SIZE) * 4, int(MAP_TEXTURE_SIZE) * 4), interpolation=cv2.INTER_NEAREST),
    )


def main() -> int:
    args = parse_args()
    track = DirectedTrack(
        reverse_direction=args.reverse_direction,
        branch_probability=0.0,
        route_seed=7,
        clearance_px=args.clearance_px,
    )

    by_segment: dict[str, list[tuple[float, float, float]]] = {}
    route_points: list[tuple[float, float]] = []
    for segment_name, distance, x, y, _ in track.outer_lap_samples(args.step_m):
        by_segment.setdefault(segment_name, []).append((distance, x, y))
        route_points.append((x, y))

    reports: list[StraightReport] = []
    anomaly_points: list[tuple[float, float]] = []
    for segment_name, samples in by_segment.items():
        report, anomalies = analyze_segment(segment_name, samples, args)
        if report is None:
            continue
        reports.append(report)
        anomaly_points.extend((x, y) for _, x, y in anomalies)

    reports.sort(key=lambda item: (item.anomaly_count == 0, -item.max_jump_px, -item.residual_range_px))
    print(
        f"straight_segments={len(reports)} step={args.step_m:.3f}m "
        f"jump_threshold={args.jump_threshold_px:.2f}px range_threshold={args.range_threshold_px:.2f}px"
    )
    for report in reports:
        status = "JITTER" if report.anomaly_count else "ok"
        print(
            f"{status:6s} {report.segment:42s} "
            f"n={report.samples:4d} len={report.length_m:.3f}m "
            f"range={report.residual_range_px:.3f}px "
            f"max_jump={report.max_jump_px:.3f}px "
            f"events={report.anomaly_count}"
        )

    draw_debug(load_map_image(), route_points, anomaly_points, args.debug_path)
    print(f"debug={args.debug_path}")
    return 1 if any(report.anomaly_count for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
