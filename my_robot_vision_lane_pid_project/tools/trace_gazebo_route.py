#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from my_robot_vision.gazebo_track_car import MAP_TEXTURE_SIZE, load_map_image, normalize_angle  # noqa: E402


class GazeboRouteTrace(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("trace_gazebo_route")
        self.args = args
        self.rows: list[dict[str, Any]] = []
        self.wall_start = time.monotonic()
        self.last_message_wall = 0.0
        self.last_trace: dict[str, Any] | None = None
        self.last_print_wall = 0.0
        self.last_route_points_map: list[list[float]] = []
        self.done = False
        self.create_subscription(String, args.topic, self.handle_trace, 10)

    def handle_trace(self, msg: String) -> None:
        now_wall = time.monotonic()
        self.last_message_wall = now_wall
        try:
            trace = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            print(f"bad trace JSON: {exc}: {msg.data[:160]}")
            return

        pose = trace.get("pose", {})
        actual_pose = trace.get("actual_pose") or {}
        previous = self.last_trace
        yaw = float(pose.get("yaw", 0.0))
        actual_yaw = float(actual_pose.get("yaw", yaw))
        yaw_jump = 0.0
        actual_yaw_jump = 0.0
        segment_changed = False
        decision_changed = False
        stopped_changed = False
        if previous is not None:
            previous_pose = previous.get("pose", {})
            previous_actual_pose = previous.get("actual_pose") or {}
            yaw_jump = normalize_angle(yaw - float(previous_pose.get("yaw", yaw)))
            actual_yaw_jump = normalize_angle(
                actual_yaw - float(previous_actual_pose.get("yaw", actual_yaw))
            )
            segment_changed = trace.get("segment") != previous.get("segment")
            decision_changed = trace.get("decision") != previous.get("decision")
            stopped_changed = trace.get("stopped_at_terminal") != previous.get("stopped_at_terminal")

        yaw_jump_deg = abs(math.degrees(yaw_jump))
        actual_yaw_jump_deg = abs(math.degrees(actual_yaw_jump))
        yaw_jump_event = yaw_jump_deg >= self.args.yaw_jump_deg or actual_yaw_jump_deg >= self.args.yaw_jump_deg
        parking_event = bool(trace.get("parking_route")) or "parking" in str(trace.get("decision", ""))
        should_print = (
            segment_changed
            or decision_changed
            or stopped_changed
            or yaw_jump_event
            or (parking_event and now_wall - self.last_print_wall >= self.args.parking_print_period)
            or now_wall - self.last_print_wall >= self.args.print_period
        )

        row = self.flatten_trace(trace, now_wall - self.wall_start, yaw_jump, actual_yaw_jump)
        self.rows.append(row)
        self.last_trace = trace
        route_points = trace.get("route_points_map") or []
        if route_points:
            self.last_route_points_map = route_points

        if should_print:
            flags: list[str] = []
            if segment_changed:
                flags.append("SEG")
            if decision_changed:
                flags.append("DECISION")
            if stopped_changed:
                flags.append("STOP")
            if yaw_jump_event:
                flags.append("YAW_JUMP")
            if parking_event:
                flags.append("PARK")
            flag_text = ",".join(flags) if flags else "TRACE"
            print(
                f"t={row['t_wall']:.2f}s {flag_text:18s} "
                f"seg={row['segment']} @{row['distance_on_segment_m']:.3f}/{row['segment_length_m']:.3f}m "
                f"progress={row['segment_progress']:.2f} "
                f"map=({row['map_x']:.1f},{row['map_y']:.1f}) "
                f"route_yaw={math.degrees(row['yaw']):+7.1f}deg "
                f"actual_yaw={math.degrees(row['actual_yaw']):+7.1f}deg "
                f"dyaw={math.degrees(row['yaw_jump']):+6.1f}/{math.degrees(row['actual_yaw_jump']):+6.1f}deg "
                f"v={row['effective_speed_mps']:.3f} "
                f"decision={row['decision']} "
                f"visible={row['visible_instruction'] or '-'}"
            )
            if parking_event and route_points:
                print(f"  route_points_map={route_points}")
            self.last_print_wall = now_wall

        if now_wall - self.wall_start >= self.args.duration:
            self.done = True

    def flatten_trace(
        self,
        trace: dict[str, Any],
        t_wall: float,
        yaw_jump: float,
        actual_yaw_jump: float,
    ) -> dict[str, Any]:
        pose = trace.get("pose", {})
        command_pose = trace.get("command_pose") or {}
        actual_pose = trace.get("actual_pose") or {}
        return {
            "t_wall": t_wall,
            "stamp_ns": trace.get("stamp_ns", 0),
            "segment": trace.get("segment", ""),
            "distance_on_segment_m": float(trace.get("distance_on_segment_m", 0.0)),
            "segment_length_m": float(trace.get("segment_length_m", 0.0)),
            "segment_progress": float(trace.get("segment_progress", 0.0)),
            "parking_route": bool(trace.get("parking_route", False)),
            "terminal_stop": bool(trace.get("terminal_stop", False)),
            "stopped_at_terminal": bool(trace.get("stopped_at_terminal", False)),
            "decision": trace.get("decision", ""),
            "pending_maneuver": trace.get("pending_maneuver") or "",
            "pending_source": trace.get("pending_source") or "",
            "visible_instruction": trace.get("visible_instruction") or "",
            "applied_instruction": trace.get("applied_instruction") or "",
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "yaw": float(pose.get("yaw", 0.0)),
            "yaw_jump": yaw_jump,
            "map_x": float(pose.get("map_x", 0.0)),
            "map_y": float(pose.get("map_y", 0.0)),
            "command_yaw": float(command_pose.get("yaw", pose.get("yaw", 0.0))),
            "command_map_x": float(command_pose.get("map_x", pose.get("map_x", 0.0))),
            "command_map_y": float(command_pose.get("map_y", pose.get("map_y", 0.0))),
            "actual_yaw": float(actual_pose.get("yaw", pose.get("yaw", 0.0))),
            "actual_yaw_jump": actual_yaw_jump,
            "actual_map_x": float(actual_pose.get("map_x", pose.get("map_x", 0.0))),
            "actual_map_y": float(actual_pose.get("map_y", pose.get("map_y", 0.0))),
            "effective_speed_mps": float(trace.get("effective_speed_mps", 0.0)),
            "active_speed_mps": float(trace.get("active_speed_mps", 0.0)),
        }

    def write_csv(self) -> None:
        if not self.rows or not self.args.csv:
            return
        path = Path(self.args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"wrote {len(self.rows)} trace rows to {path}")

    def write_debug_image(self) -> None:
        if not self.last_route_points_map or not self.args.debug_image:
            return
        image = load_map_image()
        points = np.round(np.array(self.last_route_points_map, dtype=np.float32)).astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(image, [points], False, (255, 0, 255), 1, cv2.LINE_AA)
        for point in points:
            cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1, cv2.LINE_AA)

        path = Path(self.args.debug_image)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(path),
            cv2.resize(
                image,
                (int(MAP_TEXTURE_SIZE) * 4, int(MAP_TEXTURE_SIZE) * 4),
                interpolation=cv2.INTER_NEAREST,
            ),
        )
        print(f"wrote route overlay to {path}")

    def print_summary(self) -> None:
        if not self.rows:
            print("no route trace messages captured")
            return
        yaw_jumps = [
            row
            for row in self.rows
            if abs(math.degrees(float(row["yaw_jump"]))) >= self.args.yaw_jump_deg
            or abs(math.degrees(float(row["actual_yaw_jump"]))) >= self.args.yaw_jump_deg
        ]
        parking_rows = [row for row in self.rows if bool(row["parking_route"]) or "parking" in str(row["decision"])]
        print(
            f"summary samples={len(self.rows)} parking_samples={len(parking_rows)} "
            f"yaw_jumps>={self.args.yaw_jump_deg:.1f}deg={len(yaw_jumps)} "
            f"last_segment={self.rows[-1]['segment']} "
            f"last_map=({float(self.rows[-1]['map_x']):.1f},{float(self.rows[-1]['map_y']):.1f}) "
            f"last_route_yaw={math.degrees(float(self.rows[-1]['yaw'])):+.1f}deg "
            f"last_actual_yaw={math.degrees(float(self.rows[-1]['actual_yaw'])):+.1f}deg "
            f"stopped={bool(self.rows[-1]['stopped_at_terminal'])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace live gazebo_track_car route decisions and yaw changes.")
    parser.add_argument("--topic", default="/gazebo_track_car/route_trace")
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--no-message-timeout", type=float, default=8.0)
    parser.add_argument("--print-period", type=float, default=1.0)
    parser.add_argument("--parking-print-period", type=float, default=0.15)
    parser.add_argument("--yaw-jump-deg", type=float, default=35.0)
    parser.add_argument("--csv", default="/tmp/gazebo_route_trace.csv")
    parser.add_argument("--debug-image", default="/tmp/gazebo_route_trace.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = GazeboRouteTrace(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.last_message_wall == 0.0 and time.monotonic() - node.wall_start >= args.no_message_timeout:
                print(f"no {args.topic} messages after {args.no_message_timeout:.1f}s")
                break
    finally:
        node.print_summary()
        node.write_csv()
        node.write_debug_image()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
