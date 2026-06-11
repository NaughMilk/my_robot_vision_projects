#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from my_robot_vision.gazebo_track_car import (
    DEFAULT_CLEARANCE_PX,
    DirectedTrack,
    load_map_image,
    model_point_to_map,
)


POS_RE = re.compile(
    r"pos=\((?P<x>[+-]?\d+(?:\.\d+)?),(?P<y>[+-]?\d+(?:\.\d+)?)\).*?route=(?P<route>\S+)"
)

ROUTE_PREVIEW_COLORS = {
    "magenta": (255, 0, 255),
    "cyan": (255, 255, 0),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a lightweight route preview on the current map texture.")
    parser.add_argument("--route-config", required=True, type=Path)
    parser.add_argument("--map-image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace-log", type=Path)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--trace-all", action="store_true", help="Use MOTION and JITTER samples. Default uses MOTION only.")
    parser.add_argument("--clean", action="store_true", help="Hide large labels and keep only route lines/markers.")
    parser.add_argument("--show-safe-edges", action="store_true", help="Overlay safe-mask edges for route planner debugging.")
    parser.add_argument("--safe-edges-only", action="store_true", help="Render only the safe-mask edge lines on a black background.")
    parser.add_argument("--show-safe-centerline", action="store_true", help="Overlay the merged safe-mask centerline.")
    parser.add_argument("--safe-centerline-only", action="store_true", help="Render a one-line skeleton of the safe-mask area.")
    parser.add_argument("--show-flat-guide", action="store_true", help="Overlay a rectified straight guide for the map lanes.")
    parser.add_argument("--flat-guide-only", action="store_true", help="Render only the rectified straight guide on a black background.")
    parser.add_argument("--flat-guide-map-only", action="store_true", help="Render the rectified straight guide on the map without route colors.")
    parser.add_argument(
        "--force-route-color",
        choices=sorted(ROUTE_PREVIEW_COLORS),
        help="Override route preview color without changing route semantics in the config.",
    )
    parser.add_argument(
        "--all-magenta",
        action="store_true",
        help="Shorthand for --force-route-color magenta so every route segment renders magenta.",
    )
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--reverse-direction", action="store_true")
    parser.add_argument(
        "--visual-frame",
        action="store_true",
        help="Project model poses through the Gazebo map visual offset. Default keeps the original 2D route/code view.",
    )
    parser.add_argument("--route-seed", type=int, default=11)
    parser.add_argument("--clearance-px", type=int, default=DEFAULT_CLEARANCE_PX)
    return parser.parse_args()


def scaled_points(points: list[tuple[float, float]], scale: int) -> np.ndarray:
    return np.array([[int(round(x * scale)), int(round(y * scale))] for x, y in points], dtype=np.int32)


def segment_pixels(
    track: DirectedTrack,
    segment_name: str,
    visual_frame: bool = False,
) -> list[tuple[float, float]]:
    segment = track.segments[segment_name]
    if visual_frame:
        return [model_point_to_map(x, y) for x, y in segment.points]
    return [
        model_point_to_map(x, y, track.route_offset_x_m, track.route_offset_y_m)
        for x, y in segment.points
    ]


def draw_polyline(
    image: np.ndarray,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    scale: int,
    thickness: int,
    alpha: float = 1.0,
) -> None:
    if len(points) < 2:
        return
    pts = scaled_points(points, scale)
    if alpha >= 0.99:
        cv2.polylines(image, [pts], False, color, thickness, cv2.LINE_AA)
        return
    overlay = image.copy()
    cv2.polylines(overlay, [pts], False, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def draw_arrows(
    image: np.ndarray,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    scale: int,
    every_px: float = 90.0,
) -> None:
    if len(points) < 2:
        return
    next_mark = every_px
    walked = 0.0
    for start, end in zip(points, points[1:]):
        sx, sy = start
        ex, ey = end
        length = float(np.hypot(ex - sx, ey - sy))
        if length < 1e-6:
            continue
        while walked + length >= next_mark:
            ratio = (next_mark - walked) / length
            x = sx + (ex - sx) * ratio
            y = sy + (ey - sy) * ratio
            back = 8.0 / length
            bx = x - (ex - sx) * back
            by = y - (ey - sy) * back
            cv2.arrowedLine(
                image,
                (int(round(bx * scale)), int(round(by * scale))),
                (int(round(x * scale)), int(round(y * scale))),
                color,
                max(1, int(round(2 * scale))),
                cv2.LINE_AA,
                tipLength=0.55,
            )
            next_mark += every_px
        walked += length


def draw_marker(
    image: np.ndarray,
    point: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
    scale: int,
) -> None:
    x, y = int(round(point[0] * scale)), int(round(point[1] * scale))
    cv2.circle(image, (x, y), max(5, int(4 * scale)), color, -1, cv2.LINE_AA)
    cv2.circle(image, (x, y), max(6, int(5 * scale)), (0, 0, 0), max(1, scale), cv2.LINE_AA)
    cv2.putText(
        image,
        text,
        (x + 8 * scale, y - 6 * scale),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34 * scale,
        (20, 20, 20),
        max(1, int(3 * scale)),
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x + 8 * scale, y - 6 * scale),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34 * scale,
        color,
        max(1, int(1 * scale)),
        cv2.LINE_AA,
    )


def draw_dot(image: np.ndarray, point: tuple[float, float], color: tuple[int, int, int], scale: int) -> None:
    x, y = int(round(point[0] * scale)), int(round(point[1] * scale))
    cv2.circle(image, (x, y), max(4, int(4 * scale)), (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image, (x, y), max(3, int(3 * scale)), color, -1, cv2.LINE_AA)


def flat_guide_polylines() -> list[list[tuple[float, float]]]:
    return [
        [
            (50.0, 35.0),
            (205.0, 35.0),
            (221.0, 51.0),
            (221.0, 194.0),
            (205.0, 202.0),
            (50.0, 202.0),
            (35.0, 194.0),
            (35.0, 51.0),
            (50.0, 35.0),
        ],
        [(35.0, 116.0), (221.0, 116.0)],
        [(107.0, 35.0), (107.0, 116.0)],
        [(150.0, 116.0), (150.0, 202.0)],
        [(90.0, 116.0), (90.0, 140.0)],
    ]


def draw_flat_guide(image: np.ndarray, scale: int, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    for points in flat_guide_polylines():
        draw_polyline(image, points, color, scale, max(2, int(round(2 * scale))), alpha=1.0)


def resolve_forced_route_color(args: argparse.Namespace) -> Optional[tuple[int, int, int]]:
    if args.force_route_color:
        return ROUTE_PREVIEW_COLORS[args.force_route_color]
    if args.all_magenta:
        return ROUTE_PREVIEW_COLORS["magenta"]
    return None


def draw_legend(
    image: np.ndarray,
    scale: int,
    trace: bool,
    forced_route_color: Optional[tuple[int, int, int]] = None,
) -> None:
    x0, y0 = 8 * scale, 10 * scale
    if forced_route_color is None:
        rows = [
            ((255, 255, 0), "outer route"),
            ((255, 0, 255), "middle route"),
        ]
    else:
        rows = [
            (forced_route_color, "route preview"),
        ]
    if trace:
        rows.append(((0, 255, 255), "actual trace"))
    for index, (color, text) in enumerate(rows):
        y = y0 + index * 16 * scale
        cv2.line(image, (x0, y), (x0 + 22 * scale, y), color, max(2, int(3 * scale)), cv2.LINE_AA)
        cv2.putText(
            image,
            text,
            (x0 + 28 * scale, y + 4 * scale),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34 * scale,
            (245, 245, 245),
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            (x0 + 28 * scale, y + 4 * scale),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34 * scale,
            (20, 20, 20),
            max(1, int(1 * scale)),
            cv2.LINE_AA,
        )


def load_trace_points(
    path: Path,
    motion_only: bool,
    visual_frame: bool = False,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if motion_only and "MOTION" not in line:
                continue
            match = POS_RE.search(line)
            if not match:
                continue
            x = float(match.group("x"))
            y = float(match.group("y"))
            if visual_frame:
                points.append(model_point_to_map(x, y))
            else:
                points.append(model_point_to_map(x, y, 0.0, 0.0))
    return points


def render_base(
    track: DirectedTrack,
    scale: int,
    clean: bool,
    show_safe_edges: bool,
    show_safe_centerline: bool,
    show_flat_guide: bool,
    forced_route_color: Optional[tuple[int, int, int]] = None,
    visual_frame: bool = False,
) -> np.ndarray:
    base = load_map_image()
    image = cv2.resize(base, (base.shape[1] * scale, base.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

    if show_safe_edges:
        safe_edges = cv2.Canny(track.safe_mask.astype(np.uint8) * 255, 50, 120)
        safe_edges = cv2.resize(safe_edges, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        image[safe_edges > 0] = (255, 255, 255)
    if show_safe_centerline:
        centerline = safe_centerline_mask(track.safe_mask, scale)
        image[centerline > 0] = (255, 255, 255)
    if show_flat_guide:
        draw_flat_guide(image, scale)

    ordered_names = sorted(track.segments)
    for name in ordered_names:
        segment = track.segments[name]
        points = segment_pixels(track, name, visual_frame)
        if forced_route_color is None:
            color = (255, 0, 255) if segment.inner_route else (255, 255, 0)
            thickness = max(2, int(round((3 if segment.inner_route else 2) * scale)))
            alpha = 0.86 if segment.inner_route else 0.62
            arrow_spacing_px = 58.0 if segment.inner_route else 95.0
        else:
            color = forced_route_color
            thickness = max(2, int(round(3 * scale)))
            alpha = 0.86
            arrow_spacing_px = 70.0
        draw_polyline(image, points, color, scale, thickness, alpha=alpha)
        draw_arrows(image, points, color, scale, every_px=arrow_spacing_px)

    start_segment = track.current_segment
    start_point = segment_pixels(track, start_segment.name, visual_frame)[0]
    if clean:
        draw_dot(image, start_point, (0, 255, 0), scale)
    else:
        draw_marker(image, start_point, "START", (0, 255, 0), scale)
        draw_legend(image, scale, trace=False, forced_route_color=forced_route_color)
    return image


def render_safe_edges_only(track: DirectedTrack, scale: int) -> np.ndarray:
    safe_edges = cv2.Canny(track.safe_mask.astype(np.uint8) * 255, 50, 120)
    safe_edges = cv2.resize(
        safe_edges,
        (safe_edges.shape[1] * scale, safe_edges.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    image = np.zeros((safe_edges.shape[0], safe_edges.shape[1], 3), dtype=np.uint8)
    image[safe_edges > 0] = (255, 255, 255)
    return image


def skeletonize(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(mask_u8)

    skeleton = np.zeros_like(mask_u8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(mask_u8, opened)
        eroded = cv2.erode(mask_u8, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        mask_u8 = eroded
        if cv2.countNonZero(mask_u8) == 0:
            break
    return skeleton


def safe_centerline_mask(safe_mask: np.ndarray, scale: int) -> np.ndarray:
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    merged_safe = cv2.morphologyEx(safe_mask.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel) > 0
    centerline = skeletonize(merged_safe)
    return cv2.resize(
        centerline,
        (centerline.shape[1] * scale, centerline.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )


def render_safe_centerline_only(track: DirectedTrack, scale: int) -> np.ndarray:
    centerline = safe_centerline_mask(track.safe_mask, scale)
    image = np.zeros((centerline.shape[0], centerline.shape[1], 3), dtype=np.uint8)
    image[centerline > 0] = (255, 255, 255)
    return image


def render_flat_guide_only(scale: int) -> np.ndarray:
    base_size = int(256 * scale)
    image = np.zeros((base_size, base_size, 3), dtype=np.uint8)
    draw_flat_guide(image, scale)
    return image


def render_flat_guide_map_only(scale: int) -> np.ndarray:
    base = load_map_image()
    image = cv2.resize(base, (base.shape[1] * scale, base.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
    draw_flat_guide(image, scale)
    return image


def main() -> None:
    args = parse_args()
    forced_route_color = resolve_forced_route_color(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.trace_output is not None:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)

    os.environ["MY_ROBOT_ROUTE_GRAPH_CONFIG"] = str(args.route_config.resolve())
    os.environ["MY_ROBOT_MAP_IMAGE"] = str(args.map_image.resolve())

    track = DirectedTrack(
        args.reverse_direction,
        branch_probability=0.0,
        route_seed=args.route_seed,
        route_explore_enabled=True,
        clearance_px=args.clearance_px,
    )

    if args.flat_guide_map_only:
        plan_image = render_flat_guide_map_only(args.scale)
    elif args.flat_guide_only:
        plan_image = render_flat_guide_only(args.scale)
    elif args.safe_centerline_only:
        plan_image = render_safe_centerline_only(track, args.scale)
    elif args.safe_edges_only:
        plan_image = render_safe_edges_only(track, args.scale)
    else:
        plan_image = render_base(
            track,
            args.scale,
            args.clean,
            args.show_safe_edges,
            args.show_safe_centerline,
            args.show_flat_guide,
            forced_route_color,
            args.visual_frame,
        )
    cv2.imwrite(str(args.output), plan_image)

    if args.trace_log and args.trace_output:
        trace_image = plan_image.copy()
        trace_points = load_trace_points(
            args.trace_log,
            motion_only=not args.trace_all,
            visual_frame=args.visual_frame,
        )
        draw_polyline(trace_image, trace_points, (0, 255, 255), args.scale, max(2, int(3 * args.scale)), alpha=0.95)
        if trace_points:
            if args.clean:
                draw_dot(trace_image, trace_points[0], (0, 220, 0), args.scale)
                draw_dot(trace_image, trace_points[-1], (0, 128, 255), args.scale)
            else:
                draw_marker(trace_image, trace_points[0], "TRACE START", (0, 220, 0), args.scale)
                draw_marker(trace_image, trace_points[-1], "TRACE END", (0, 128, 255), args.scale)
                draw_legend(trace_image, args.scale, trace=True, forced_route_color=forced_route_color)
        cv2.imwrite(str(args.trace_output), trace_image)

    print(f"Wrote {args.output}")
    if args.trace_log and args.trace_output:
        print(f"Wrote {args.trace_output}")


if __name__ == "__main__":
    main()
