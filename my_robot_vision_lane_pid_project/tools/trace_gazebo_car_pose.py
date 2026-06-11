#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import rclpy
from gazebo_msgs.msg import ModelStates
from rclpy.node import Node

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from my_robot_vision.gazebo_track_car import (  # noqa: E402
    DEFAULT_CLEARANCE_PX,
    DEFAULT_ROUTE_OFFSET_X_M,
    DEFAULT_ROUTE_OFFSET_Y_M,
    FOOTPRINT_FRONT_M,
    FOOTPRINT_HALF_WIDTH_M,
    FOOTPRINT_MARGIN_M,
    FOOTPRINT_REAR_M,
    DirectedTrack,
    check_road_footprint,
    model_point_to_map,
    normalize_angle,
    quaternion_to_rpy,
)


class GazeboCarTrace(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("trace_gazebo_car_pose")
        self.args = args
        self.track = DirectedTrack(
            args.reverse_direction,
            0.0,
            args.route_seed,
            args.clearance_px,
            args.route_offset_x_m,
            args.route_offset_y_m,
        )
        self.footprint_front_m = FOOTPRINT_FRONT_M + FOOTPRINT_MARGIN_M
        self.footprint_rear_m = FOOTPRINT_REAR_M + FOOTPRINT_MARGIN_M
        self.footprint_half_width_m = FOOTPRINT_HALF_WIDTH_M + FOOTPRINT_MARGIN_M
        self.rows: list[dict[str, float | str | bool]] = []
        self.wall_start = time.monotonic()
        self.start_ns: int | None = None
        self.last_print_ns = 0
        self.done = False
        self.create_subscription(ModelStates, "/model_states", self.handle_model_states, 10)

    def handle_model_states(self, msg: ModelStates) -> None:
        try:
            index = msg.name.index(self.args.entity_name)
        except ValueError:
            return

        now_ns = self.get_clock().now().nanoseconds
        if self.start_ns is None:
            self.start_ns = now_ns

        pose = msg.pose[index]
        twist = msg.twist[index]
        roll, pitch, yaw = quaternion_to_rpy(pose.orientation)
        x = float(pose.position.x)
        y = float(pose.position.y)
        speed = math.hypot(float(twist.linear.x), float(twist.linear.y))
        projection = self.track.nearest_projection(x, y, outer_only=False)
        route_yaw = projection.yaw
        yaw_error = normalize_angle(yaw - route_yaw)
        footprint = check_road_footprint(
            self.track.clearance,
            x,
            y,
            yaw,
            self.footprint_front_m,
            self.footprint_rear_m,
            self.footprint_half_width_m,
        )
        map_x, map_y = model_point_to_map(x, y)
        t = (now_ns - self.start_ns) * 1e-9

        row: dict[str, float | str | bool] = {
            "t": t,
            "x": x,
            "y": y,
            "z": float(pose.position.z),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "speed": speed,
            "map_x": map_x,
            "map_y": map_y,
            "nearest_segment": projection.segment.name,
            "nearest_segment_distance": projection.distance_on_segment,
            "route_x": projection.x,
            "route_y": projection.y,
            "route_yaw": route_yaw,
            "route_distance": projection.distance_m,
            "yaw_error": yaw_error,
            "footprint_on_road": footprint.on_road,
            "min_clearance_px": footprint.min_clearance_px,
        }
        self.rows.append(row)

        should_print = (
            not footprint.on_road
            or now_ns - self.last_print_ns >= int(max(self.args.print_period, 0.05) * 1e9)
        )
        if should_print:
            state = "road" if footprint.on_road else "OFFROAD"
            print(
                f"t={t:6.2f}s {state:7s} "
                f"pos=({x:+.3f},{y:+.3f}) yaw={math.degrees(yaw):+6.1f}deg "
                f"route={projection.segment.name}@{projection.distance_on_segment:.3f}m "
                f"route_dist={projection.distance_m:.3f}m "
                f"yaw_err={math.degrees(yaw_error):+6.1f}deg "
                f"clearance={footprint.min_clearance_px:.2f}px "
                f"map=({map_x:.1f},{map_y:.1f})"
            )
            self.last_print_ns = now_ns

        if t >= self.args.duration:
            self.done = True

    def write_csv(self) -> None:
        if not self.args.csv or not self.rows:
            return
        path = Path(self.args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"wrote {len(self.rows)} samples to {path}")

    def print_summary(self) -> None:
        if not self.rows:
            print("no samples captured")
            return
        offroad = [row for row in self.rows if not bool(row["footprint_on_road"])]
        worst = min(self.rows, key=lambda row: float(row["min_clearance_px"]))
        print(
            f"summary samples={len(self.rows)} offroad={len(offroad)} "
            f"worst_clearance={float(worst['min_clearance_px']):.2f}px "
            f"worst_pos=({float(worst['x']):+.3f},{float(worst['y']):+.3f}) "
            f"worst_route={worst['nearest_segment']} "
            f"yaw={math.degrees(float(worst['yaw'])):+.1f}deg "
            f"yaw_err={math.degrees(float(worst['yaw_error'])):+.1f}deg"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace Gazebo car pose, route projection, and footprint clearance.")
    parser.add_argument("--entity-name", default="track_preview_car")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--no-sample-timeout", type=float, default=5.0)
    parser.add_argument("--print-period", type=float, default=0.5)
    parser.add_argument("--csv", default="/tmp/gazebo_car_trace.csv")
    parser.add_argument("--clearance-px", type=int, default=DEFAULT_CLEARANCE_PX)
    parser.add_argument("--route-seed", type=int, default=7)
    parser.add_argument("--route-offset-x-m", type=float, default=DEFAULT_ROUTE_OFFSET_X_M)
    parser.add_argument("--route-offset-y-m", type=float, default=DEFAULT_ROUTE_OFFSET_Y_M)
    direction = parser.add_mutually_exclusive_group()
    direction.add_argument("--reverse-direction", dest="reverse_direction", action="store_true")
    direction.add_argument("--no-reverse-direction", dest="reverse_direction", action="store_false")
    parser.set_defaults(reverse_direction=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = GazeboCarTrace(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.start_ns is None and time.monotonic() - node.wall_start >= args.no_sample_timeout:
                print(f"no /model_states samples received after {args.no_sample_timeout:.1f}s")
                break
    finally:
        node.print_summary()
        node.write_csv()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
