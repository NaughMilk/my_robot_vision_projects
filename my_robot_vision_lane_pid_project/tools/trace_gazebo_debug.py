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

import rclpy
from gazebo_msgs.msg import ModelStates
from rclpy.node import Node
from std_msgs.msg import String

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from my_robot_vision.gazebo_track_car import model_point_to_map, quaternion_to_rpy  # noqa: E402


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any) -> bool:
    return bool(value)


class GazeboDebugTrace(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("trace_gazebo_debug")
        self.args = args
        self.wall_start = time.monotonic()
        self.last_any_wall = 0.0
        self.last_pose_wall = 0.0
        self.done = False

        self.sign_rows: list[dict[str, Any]] = []
        self.route_rows: list[dict[str, Any]] = []
        self.pose_rows: list[dict[str, Any]] = []
        self.jsonl_rows: list[dict[str, Any]] = []

        self.last_sign_signature: tuple[Any, ...] | None = None
        self.last_route_signature: tuple[Any, ...] | None = None
        self.last_pose_print_wall = 0.0

        self.create_subscription(String, args.sign_topic, self.handle_sign, 10)
        self.create_subscription(String, args.route_topic, self.handle_route, 10)
        self.create_subscription(ModelStates, args.model_states_topic, self.handle_model_states, 10)

    def elapsed(self) -> float:
        return time.monotonic() - self.wall_start

    def remember(self, event: str, payload: dict[str, Any]) -> None:
        self.last_any_wall = time.monotonic()
        if self.args.jsonl:
            self.jsonl_rows.append({"event": event, "t_wall": self.elapsed(), **payload})

    def decode_json_msg(self, msg: String, topic: str) -> dict[str, Any] | None:
        try:
            return json.loads(msg.data)
        except json.JSONDecodeError as exc:
            print(f"bad JSON from {topic}: {exc}: {msg.data[:160]}")
            return None

    def handle_sign(self, msg: String) -> None:
        payload = self.decode_json_msg(msg, self.args.sign_topic)
        if payload is None:
            return

        bbox = payload.get("bbox_px") or [0, 0]
        center = payload.get("center") or [0, 0]
        row = {
            "t_wall": self.elapsed(),
            "stamp_ns": payload.get("stamp_ns", 0),
            "visible": as_bool(payload.get("visible")),
            "stable": as_bool(payload.get("stable")),
            "stable_count": int(payload.get("stable_count") or 0),
            "name": payload.get("name") or "",
            "kind": payload.get("kind") or "",
            "maneuver": payload.get("maneuver") or "",
            "score": as_float(payload.get("score")),
            "area_px": int(payload.get("area_px") or 0),
            "bbox_w": int(bbox[0] if len(bbox) > 0 else 0),
            "bbox_h": int(bbox[1] if len(bbox) > 1 else 0),
            "center_x": int(center[0] if len(center) > 0 else 0),
            "center_y": int(center[1] if len(center) > 1 else 0),
            "last_applied": payload.get("last_applied") or "",
        }
        self.sign_rows.append(row)
        self.remember("sign", row)

        signature = (
            row["visible"],
            row["stable"],
            row["name"],
            row["kind"],
            row["maneuver"],
            row["last_applied"],
        )
        if signature != self.last_sign_signature:
            visible = "visible" if row["visible"] else "hidden"
            stable = "stable" if row["stable"] else f"stable_count={row['stable_count']}"
            print(
                f"sign {visible:7s} {stable:14s} "
                f"name={row['name'] or '-'} kind={row['kind'] or '-'} maneuver={row['maneuver'] or '-'} "
                f"score={row['score']:.2f} area={row['area_px']} last_applied={row['last_applied'] or '-'}"
            )
            self.last_sign_signature = signature

    def handle_route(self, msg: String) -> None:
        payload = self.decode_json_msg(msg, self.args.route_topic)
        if payload is None:
            return

        pose = payload.get("pose") or {}
        actual = payload.get("actual_pose") or {}
        row = {
            "t_wall": self.elapsed(),
            "stamp_ns": payload.get("stamp_ns", 0),
            "segment": payload.get("segment") or "",
            "distance_on_segment_m": as_float(payload.get("distance_on_segment_m")),
            "segment_length_m": as_float(payload.get("segment_length_m")),
            "segment_progress": as_float(payload.get("segment_progress")),
            "decision": payload.get("decision") or "",
            "pending_maneuver": payload.get("pending_maneuver") or "",
            "pending_source": payload.get("pending_source") or "",
            "camera_parking_armed": payload.get("camera_parking_armed") or "",
            "visible_instruction": payload.get("visible_instruction") or "",
            "applied_instruction": payload.get("applied_instruction") or "",
            "parking_route": as_bool(payload.get("parking_route")),
            "stopped_at_terminal": as_bool(payload.get("stopped_at_terminal")),
            "effective_speed_mps": as_float(payload.get("effective_speed_mps")),
            "active_speed_mps": as_float(payload.get("active_speed_mps")),
            "traffic_light_state": payload.get("traffic_light_state") or "",
            "camera_traffic_light_state": payload.get("camera_traffic_light_state") or "",
            "traffic_light_control_state": payload.get("traffic_light_control_state") or "",
            "map_x": as_float(pose.get("map_x")),
            "map_y": as_float(pose.get("map_y")),
            "yaw": as_float(pose.get("yaw")),
            "actual_map_x": as_float(actual.get("map_x"), as_float(pose.get("map_x"))),
            "actual_map_y": as_float(actual.get("map_y"), as_float(pose.get("map_y"))),
            "actual_yaw": as_float(actual.get("yaw"), as_float(pose.get("yaw"))),
        }
        self.route_rows.append(row)
        self.remember("route", row)

        signature = (
            row["segment"],
            row["decision"],
            row["pending_maneuver"],
            row["camera_parking_armed"],
            row["visible_instruction"],
            row["applied_instruction"],
            row["stopped_at_terminal"],
        )
        if signature != self.last_route_signature:
            print(
                f"route seg={row['segment']} @{row['distance_on_segment_m']:.3f}/"
                f"{row['segment_length_m']:.3f}m map=({row['map_x']:.1f},{row['map_y']:.1f}) "
                f"pending={row['pending_maneuver'] or '-'} source={row['pending_source'] or '-'} "
                f"parking_armed={row['camera_parking_armed'] or '-'} "
                f"visible={row['visible_instruction'] or '-'} applied={row['applied_instruction'] or '-'} "
                f"decision={row['decision'] or '-'} stopped={row['stopped_at_terminal']}"
            )
            self.last_route_signature = signature

    def handle_model_states(self, msg: ModelStates) -> None:
        try:
            index = msg.name.index(self.args.entity_name)
        except ValueError:
            return

        now_wall = time.monotonic()
        if now_wall - self.last_pose_wall < self.args.pose_period:
            return
        self.last_pose_wall = now_wall

        pose = msg.pose[index]
        twist = msg.twist[index]
        roll, pitch, yaw = quaternion_to_rpy(pose.orientation)
        x = float(pose.position.x)
        y = float(pose.position.y)
        map_x, map_y = model_point_to_map(x, y)
        row = {
            "t_wall": self.elapsed(),
            "stamp_ns": self.get_clock().now().nanoseconds,
            "name": self.args.entity_name,
            "x": x,
            "y": y,
            "z": float(pose.position.z),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "map_x": map_x,
            "map_y": map_y,
            "linear_x": float(twist.linear.x),
            "linear_y": float(twist.linear.y),
            "linear_z": float(twist.linear.z),
            "speed_xy": math.hypot(float(twist.linear.x), float(twist.linear.y)),
            "angular_z": float(twist.angular.z),
        }
        self.pose_rows.append(row)
        self.remember("model_state", row)

        if now_wall - self.last_pose_print_wall >= self.args.print_pose_period:
            print(
                f"pose map=({row['map_x']:.1f},{row['map_y']:.1f}) "
                f"z={row['z']:.3f} yaw={math.degrees(row['yaw']):+.1f}deg "
                f"speed={row['speed_xy']:.3f}m/s"
            )
            self.last_pose_print_wall = now_wall

    def write_csv(self, rows: list[dict[str, Any]], path_text: str) -> None:
        if not rows or not path_text:
            return
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {path}")

    def write_jsonl(self) -> None:
        if not self.jsonl_rows or not self.args.jsonl:
            return
        path = Path(self.args.jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in self.jsonl_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"wrote {len(self.jsonl_rows)} events to {path}")

    def print_summary(self) -> None:
        print(
            "summary "
            f"signs={len(self.sign_rows)} routes={len(self.route_rows)} poses={len(self.pose_rows)} "
            f"last_sign={(self.sign_rows[-1]['name'] if self.sign_rows else '-') or '-'} "
            f"last_applied={(self.sign_rows[-1]['last_applied'] if self.sign_rows else '-') or '-'} "
            f"last_segment={(self.route_rows[-1]['segment'] if self.route_rows else '-') or '-'} "
            f"pending={(self.route_rows[-1]['pending_maneuver'] if self.route_rows else '-') or '-'}"
        )

    def finish(self) -> None:
        self.print_summary()
        self.write_csv(self.sign_rows, self.args.sign_csv)
        self.write_csv(self.route_rows, self.args.route_csv)
        self.write_csv(self.pose_rows, self.args.pose_csv)
        self.write_jsonl()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace Gazebo sign detection, route decisions, and model state together."
    )
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--no-message-timeout", type=float, default=8.0)
    parser.add_argument("--entity-name", default="track_preview_car")
    parser.add_argument("--sign-topic", default="/gazebo_track_car/camera_sign_detection")
    parser.add_argument("--route-topic", default="/gazebo_track_car/route_trace")
    parser.add_argument("--model-states-topic", default="/model_states")
    parser.add_argument("--pose-period", type=float, default=0.10)
    parser.add_argument("--print-pose-period", type=float, default=1.0)
    parser.add_argument("--sign-csv", default="/tmp/gazebo_debug_sign.csv")
    parser.add_argument("--route-csv", default="/tmp/gazebo_debug_route.csv")
    parser.add_argument("--pose-csv", default="/tmp/gazebo_debug_pose.csv")
    parser.add_argument("--jsonl", default="/tmp/gazebo_debug_events.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = GazeboDebugTrace(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            elapsed = node.elapsed()
            if elapsed >= args.duration:
                node.done = True
            if node.last_any_wall == 0.0 and elapsed >= args.no_message_timeout:
                print(f"no trace messages after {args.no_message_timeout:.1f}s")
                break
    finally:
        node.finish()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
