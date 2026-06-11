#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import select
import signal
import subprocess
import sys
import termios
import time
import tty
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    import rclpy
    from gazebo_msgs.msg import EntityState, ModelStates
    from gazebo_msgs.srv import SetEntityState
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
except ImportError as exc:  # pragma: no cover - user environment guard
    print(
        "Could not import ROS2 modules. Run:\n"
        "  source /opt/ros/foxy/setup.bash\n"
        "  source /home/quocbao/my_robot_vision_full_project/install/setup.bash",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


MODEL_SIZE_M = 6.0
MAP_TEXTURE_SIZE = 256.0
MAP_VISUAL_OFFSET_X_M = 0.0
MAP_VISUAL_OFFSET_Y_M = 0.30

DEFAULT_START_POSES = {
    "top_left": (-1.828125, 2.1796875, 0.02, 0.0, 0.0, math.pi),
    "top_right": (1.828125, 2.1796875, 0.02, 0.0, 0.0, -math.pi * 0.5),
    "bottom_right": (1.828125, -1.8796875, 0.02, 0.0, 0.0, 0.0),
    "bottom_left": (-1.828125, -1.8796875, 0.02, 0.0, 0.0, math.pi * 0.5),
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def default_source_world(package_root: Path) -> Path:
    downloads_map = Path("/home/quocbao/Downloads/map_1")
    if downloads_map.exists():
        return downloads_map
    return package_root / "worlds" / "autonomous_map_3d.world"


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child_text(element: ET.Element, child_name: str) -> str | None:
    for child in element:
        if local_name(child.tag) == child_name:
            return child.text
    return None


def find_child(element: ET.Element, child_name: str) -> ET.Element | None:
    for child in element:
        if local_name(child.tag) == child_name:
            return child
    return None


def is_road_sign_model_name(name: str | None) -> bool:
    if not name:
        return False
    lower = name.lower()
    return lower.startswith("sign_") or "road_sign" in lower


def pose_text(pose: tuple[float, float, float, float, float, float]) -> str:
    return " ".join(f"{value:.6f}" for value in pose)


def indent_xml(element: ET.Element, level: int = 0) -> None:
    spacer = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = spacer + "  "
        for child in element:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = spacer
    if level and (not element.tail or not element.tail.strip()):
        element.tail = spacer


def generate_signless_world(source_world: Path, output_world: Path, start_pose: tuple[float, ...]) -> dict:
    tree = ET.parse(str(source_world))
    root = tree.getroot()
    world = find_child(root, "world")
    if world is None:
        raise RuntimeError(f"World file has no <world>: {source_world}")

    removed_models: list[str] = []
    for child in list(world):
        if local_name(child.tag) == "model" and is_road_sign_model_name(child.get("name")):
            removed_models.append(child.get("name") or "")
            world.remove(child)

    # Saved <state> blocks can resurrect signs or old car poses. Remove them for a clean recording world.
    removed_state_count = 0
    for child in list(world):
        if local_name(child.tag) == "state":
            world.remove(child)
            removed_state_count += 1

    car_model = None
    for child in world:
        if local_name(child.tag) == "model" and child.get("name") == "track_preview_car":
            car_model = child
            break

    if car_model is None:
        include = ET.Element("include")
        uri = ET.SubElement(include, "uri")
        uri.text = "model://track_preview_car"
        name = ET.SubElement(include, "name")
        name.text = "track_preview_car"
        pose = ET.SubElement(include, "pose")
        pose.text = pose_text(start_pose)
        world.append(include)
        car_spawn_mode = "include_added"
    else:
        pose = find_child(car_model, "pose")
        if pose is None:
            pose = ET.SubElement(car_model, "pose")
        pose.text = pose_text(start_pose)
        car_spawn_mode = "existing_model_pose_updated"

    output_world.parent.mkdir(parents=True, exist_ok=True)
    indent_xml(root)
    tree.write(str(output_world), encoding="utf-8", xml_declaration=True)
    return {
        "source_world": str(source_world),
        "output_world": str(output_world),
        "removed_road_sign_models": removed_models,
        "removed_state_blocks": removed_state_count,
        "car_spawn_mode": car_spawn_mode,
        "start_pose": list(start_pose),
    }


def quaternion_to_yaw(q) -> float:
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)
    sin_yaw_cos_pitch = 2.0 * (w * z + x * y)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)


def yaw_to_quaternion(yaw: float):
    from geometry_msgs.msg import Quaternion

    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def model_to_map_px(
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


@dataclass
class RawSample:
    stamp_s: float
    segment_id: int
    x: float
    y: float
    z: float
    yaw: float
    linear_cmd: float
    angular_cmd: float

    def to_payload(self) -> dict:
        map_x, map_y = model_to_map_px(self.x, self.y)
        return {
            "stamp_s": self.stamp_s,
            "segment_id": self.segment_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "map_x": map_x,
            "map_y": map_y,
            "linear_cmd": self.linear_cmd,
            "angular_cmd": self.angular_cmd,
        }


@dataclass
class GraphNode:
    node_id: int
    x_sum: float
    y_sum: float
    z_sum: float
    yaw_sin_sum: float
    yaw_cos_sum: float
    visits: int = 1
    first_stamp_s: float = 0.0
    last_stamp_s: float = 0.0

    @property
    def x(self) -> float:
        return self.x_sum / max(self.visits, 1)

    @property
    def y(self) -> float:
        return self.y_sum / max(self.visits, 1)

    @property
    def z(self) -> float:
        return self.z_sum / max(self.visits, 1)

    @property
    def yaw(self) -> float:
        return math.atan2(self.yaw_sin_sum, self.yaw_cos_sum)

    def update(self, sample: RawSample) -> None:
        self.x_sum += sample.x
        self.y_sum += sample.y
        self.z_sum += sample.z
        self.yaw_sin_sum += math.sin(sample.yaw)
        self.yaw_cos_sum += math.cos(sample.yaw)
        self.visits += 1
        self.last_stamp_s = sample.stamp_s

    def to_payload(self) -> dict:
        map_x, map_y = model_to_map_px(self.x, self.y)
        return {
            "id": self.node_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "map_x": map_x,
            "map_y": map_y,
            "visits": self.visits,
            "first_stamp_s": self.first_stamp_s,
            "last_stamp_s": self.last_stamp_s,
        }


@dataclass
class GraphEdge:
    source: int
    target: int
    visits: int = 0
    length_sum_m: float = 0.0
    segment_ids: set[int] = field(default_factory=set)

    @property
    def length_m(self) -> float:
        return self.length_sum_m / max(self.visits, 1)

    def to_payload(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "bidirectional": True,
            "visits": self.visits,
            "length_m": self.length_m,
            "segment_ids": sorted(self.segment_ids),
        }


class RouteGraphBuilder:
    def __init__(self, merge_radius_m: float) -> None:
        self.merge_radius_m = max(merge_radius_m, 0.01)
        self.cell_size_m = self.merge_radius_m
        self.nodes: dict[int, GraphNode] = {}
        self.edges: dict[tuple[int, int], GraphEdge] = {}
        self.spatial_index: dict[tuple[int, int], set[int]] = {}
        self.next_node_id = 1

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor(x / self.cell_size_m)), int(math.floor(y / self.cell_size_m)))

    def _candidate_node_ids(self, x: float, y: float) -> Iterable[int]:
        cx, cy = self._cell(x, y)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                yield from self.spatial_index.get((cx + dx, cy + dy), ())

    def add_sample(self, sample: RawSample) -> int:
        best_id = None
        best_dist = float("inf")
        for node_id in self._candidate_node_ids(sample.x, sample.y):
            node = self.nodes[node_id]
            distance = math.hypot(sample.x - node.x, sample.y - node.y)
            if distance < best_dist:
                best_id = node_id
                best_dist = distance

        if best_id is not None and best_dist <= self.merge_radius_m:
            self.nodes[best_id].update(sample)
            return best_id

        node_id = self.next_node_id
        self.next_node_id += 1
        self.nodes[node_id] = GraphNode(
            node_id=node_id,
            x_sum=sample.x,
            y_sum=sample.y,
            z_sum=sample.z,
            yaw_sin_sum=math.sin(sample.yaw),
            yaw_cos_sum=math.cos(sample.yaw),
            visits=1,
            first_stamp_s=sample.stamp_s,
            last_stamp_s=sample.stamp_s,
        )
        self.spatial_index.setdefault(self._cell(sample.x, sample.y), set()).add(node_id)
        return node_id

    def add_edge(self, source: int, target: int, segment_id: int) -> None:
        if source == target:
            return
        key = (min(source, target), max(source, target))
        source_node = self.nodes[source]
        target_node = self.nodes[target]
        length_m = math.hypot(source_node.x - target_node.x, source_node.y - target_node.y)
        edge = self.edges.get(key)
        if edge is None:
            edge = GraphEdge(source=key[0], target=key[1])
            self.edges[key] = edge
        edge.visits += 1
        edge.length_sum_m += length_m
        edge.segment_ids.add(segment_id)

    def to_payload(self) -> dict:
        return {
            "bidirectional_edges": True,
            "merge_radius_m": self.merge_radius_m,
            "nodes": [self.nodes[node_id].to_payload() for node_id in sorted(self.nodes)],
            "edges": [self.edges[key].to_payload() for key in sorted(self.edges)],
        }


class ManualDriveRecorder(Node):
    def __init__(self, args: argparse.Namespace, output_dir: Path) -> None:
        super().__init__("manual_route_recorder")
        self.args = args
        self.output_dir = output_dir
        self.entity_name = args.entity_name
        self.cmd_pub = self.create_publisher(Twist, args.cmd_topic, 10)
        self.state_sub = self.create_subscription(ModelStates, args.model_states_topic, self.on_model_states, 20)
        self.set_entity_client = self.create_client(SetEntityState, args.set_entity_state_service)
        self.graph = RouteGraphBuilder(args.node_merge_radius_m)
        self.raw_samples: list[RawSample] = []
        self.current_pose: tuple[float, float, float, float] | None = None
        self.last_recorded_pose: tuple[float, float, float, float] | None = None
        self.last_node_id: int | None = None
        self.linear_cmd = 0.0
        self.angular_cmd = 0.0
        self.recording_enabled = True
        self.segment_id = 1
        self.last_save_wall_s = 0.0
        self.last_cmd_publish_s = 0.0
        self.last_status_s = 0.0
        self.start_wall_s = time.time()

    def on_model_states(self, msg: ModelStates) -> None:
        try:
            index = msg.name.index(self.entity_name)
        except ValueError:
            return
        pose = msg.pose[index]
        yaw = quaternion_to_yaw(pose.orientation)
        self.current_pose = (
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
            yaw,
        )

    def reset_to_pose(self, pose: tuple[float, float, float, float, float, float]) -> bool:
        if not self.set_entity_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("set_entity_state service is not ready; cannot reset car pose.")
            return False

        state = EntityState()
        state.name = self.entity_name
        state.pose.position.x = pose[0]
        state.pose.position.y = pose[1]
        state.pose.position.z = pose[2]
        state.pose.orientation = yaw_to_quaternion(pose[5])
        request = SetEntityState.Request()
        request.state = state
        future = self.set_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        ok = bool(future.done() and future.result() and future.result().success)
        if ok:
            self.linear_cmd = 0.0
            self.angular_cmd = 0.0
            self.new_segment()
        else:
            self.get_logger().warn("Gazebo refused the pose reset request.")
        return ok

    def new_segment(self) -> None:
        self.segment_id += 1
        self.last_node_id = None
        self.last_recorded_pose = None

    def maybe_record(self) -> None:
        if not self.recording_enabled or self.current_pose is None:
            return

        x, y, z, yaw = self.current_pose
        should_record = self.last_recorded_pose is None
        if not should_record:
            last_x, last_y, _, last_yaw = self.last_recorded_pose
            distance = math.hypot(x - last_x, y - last_y)
            yaw_delta = abs(normalize_angle(yaw - last_yaw))
            should_record = distance >= self.args.sample_distance_m or yaw_delta >= self.args.sample_yaw_rad

        if not should_record:
            return

        sample = RawSample(
            stamp_s=time.time() - self.start_wall_s,
            segment_id=self.segment_id,
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            linear_cmd=self.linear_cmd,
            angular_cmd=self.angular_cmd,
        )
        self.raw_samples.append(sample)
        node_id = self.graph.add_sample(sample)
        if self.last_node_id is not None:
            self.graph.add_edge(self.last_node_id, node_id, self.segment_id)
        self.last_node_id = node_id
        self.last_recorded_pose = (x, y, z, yaw)

    def publish_cmd(self) -> None:
        now_s = time.time()
        if now_s - self.last_cmd_publish_s < 1.0 / max(self.args.cmd_rate_hz, 1.0):
            return
        twist = Twist()
        twist.linear.x = self.linear_cmd
        twist.angular.z = self.angular_cmd
        self.cmd_pub.publish(twist)
        self.last_cmd_publish_s = now_s

    def maybe_autosave(self) -> None:
        now_s = time.time()
        if now_s - self.last_save_wall_s >= self.args.autosave_period_s:
            self.save_outputs(reason="autosave", quiet=True)

    def save_outputs(self, reason: str, quiet: bool = False) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_payload = [sample.to_payload() for sample in self.raw_samples]
        graph_payload = self.graph.to_payload()
        payload = {
            "metadata": {
                "reason": reason,
                "entity_name": self.entity_name,
                "created_wall_time_s": time.time(),
                "sample_distance_m": self.args.sample_distance_m,
                "sample_yaw_rad": self.args.sample_yaw_rad,
                "node_merge_radius_m": self.args.node_merge_radius_m,
                "model_size_m": MODEL_SIZE_M,
                "map_texture_size_px": MAP_TEXTURE_SIZE,
                "map_visual_offset_m": [MAP_VISUAL_OFFSET_X_M, MAP_VISUAL_OFFSET_Y_M],
                "notes": "Edges are undirected so reverse-direction passes merge into the same road graph.",
            },
            "raw_samples": raw_payload,
            "graph": graph_payload,
        }

        with (self.output_dir / "recorded_route_graph.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.write_csv_files(raw_payload, graph_payload)
        self.write_preview_png(graph_payload)
        self.last_save_wall_s = time.time()
        if not quiet:
            print(
                f"\nSaved {len(self.raw_samples)} samples, "
                f"{len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges -> {self.output_dir}",
                flush=True,
            )

    def write_csv_files(self, raw_payload: list[dict], graph_payload: dict) -> None:
        def write_rows(path: Path, rows: list[dict]) -> None:
            if not rows:
                path.write_text("", encoding="utf-8")
                return
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        write_rows(self.output_dir / "recorded_route_raw.csv", raw_payload)
        write_rows(self.output_dir / "recorded_route_graph_nodes.csv", graph_payload["nodes"])
        write_rows(self.output_dir / "recorded_route_graph_edges.csv", graph_payload["edges"])

    def write_preview_png(self, graph_payload: dict) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            return

        package_root = repo_root_from_script()
        texture_candidates = [
            package_root / "maps" / "autonomous_map_visual.png",
            package_root
            / "models"
            / "autonomous_map_3d"
            / "materials"
            / "textures"
            / "autonomous_map_visual.png",
        ]
        image = None
        for texture_path in texture_candidates:
            if texture_path.exists():
                image = cv2.imread(str(texture_path), cv2.IMREAD_COLOR)
                break
        if image is None:
            image = np.full((256, 256, 3), 48, dtype=np.uint8)

        scale = 4
        preview = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        nodes = {node["id"]: node for node in graph_payload["nodes"]}
        for edge in graph_payload["edges"]:
            source = nodes.get(edge["source"])
            target = nodes.get(edge["target"])
            if source is None or target is None:
                continue
            p1 = (int(round(source["map_x"] * scale)), int(round(source["map_y"] * scale)))
            p2 = (int(round(target["map_x"] * scale)), int(round(target["map_y"] * scale)))
            cv2.line(preview, p1, p2, (255, 0, 255), 2, cv2.LINE_AA)
        for node in graph_payload["nodes"]:
            point = (int(round(node["map_x"] * scale)), int(round(node["map_y"] * scale)))
            cv2.circle(preview, point, 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.imwrite(str(self.output_dir / "recorded_route_overlay.png"), preview)

    def status_line(self) -> str:
        pose_text_value = "pose=waiting"
        if self.current_pose is not None:
            x, y, _, yaw = self.current_pose
            pose_text_value = f"pose=({x:+.2f},{y:+.2f}) yaw={yaw:+.2f}"
        mode = "REC" if self.recording_enabled else "PAUSED"
        return (
            f"{mode} v={self.linear_cmd:+.2f} wz={self.angular_cmd:+.2f} "
            f"samples={len(self.raw_samples)} nodes={len(self.graph.nodes)} "
            f"edges={len(self.graph.edges)} seg={self.segment_id} {pose_text_value}"
        )


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def handle_key(key: str, node: ManualDriveRecorder) -> bool:
    args = node.args
    if key in {"q", "\x03"}:
        return False
    if key == "w":
        node.linear_cmd = clamp(node.linear_cmd + args.linear_step_mps, -args.max_reverse_mps, args.max_linear_mps)
    elif key == "s":
        node.linear_cmd = clamp(node.linear_cmd - args.linear_step_mps, -args.max_reverse_mps, args.max_linear_mps)
    elif key == "a":
        node.angular_cmd = clamp(node.angular_cmd + args.angular_step_rps, -args.max_angular_rps, args.max_angular_rps)
    elif key == "d":
        node.angular_cmd = clamp(node.angular_cmd - args.angular_step_rps, -args.max_angular_rps, args.max_angular_rps)
    elif key == "e":
        node.angular_cmd = 0.0
    elif key == " ":
        node.linear_cmd = 0.0
        node.angular_cmd = 0.0
    elif key == "r":
        node.recording_enabled = not node.recording_enabled
        node.new_segment()
        print(f"\nRecording {'enabled' if node.recording_enabled else 'paused'}.", flush=True)
    elif key == "n":
        node.new_segment()
        print("\nStarted a new disconnected segment.", flush=True)
    elif key == "p":
        node.save_outputs(reason="manual_save")
    elif key in {"1", "2", "3", "4"}:
        corner_names = ["top_left", "top_right", "bottom_right", "bottom_left"]
        corner_name = corner_names[int(key) - 1]
        node.reset_to_pose(DEFAULT_START_POSES[corner_name])
        print(f"\nReset car to {corner_name}.", flush=True)
    return True


def print_controls(output_dir: Path) -> None:
    print(
        "\nManual route recording controls\n"
        "  w/s  : increase/decrease linear velocity\n"
        "  a/d  : steer left/right by changing angular velocity\n"
        "  e    : zero angular velocity\n"
        "  space: stop car\n"
        "  r    : pause/resume recording\n"
        "  n    : new disconnected segment (use after reset or a bad jump)\n"
        "  p    : save now\n"
        "  1-4  : reset to top-left/top-right/bottom-right/bottom-left\n"
        "  q    : save and quit\n\n"
        f"Output directory: {output_dir}\n",
        flush=True,
    )


def keyboard_loop(node: ManualDriveRecorder) -> None:
    print_controls(node.output_dir)
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        running = True
        while running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            while select.select([sys.stdin], [], [], 0.0)[0]:
                key = sys.stdin.read(1)
                running = handle_key(key, node)
                if not running:
                    break
            node.publish_cmd()
            node.maybe_record()
            node.maybe_autosave()
            now_s = time.time()
            if now_s - node.last_status_s > 0.5:
                print("\r" + node.status_line() + " " * 8, end="", flush=True)
                node.last_status_s = now_s
            time.sleep(0.01)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.linear_cmd = 0.0
        node.angular_cmd = 0.0
        for _ in range(5):
            node.publish_cmd()
            time.sleep(0.02)
        node.save_outputs(reason="session_end")
        print()


def start_gazebo(world_path: Path, output_dir: Path, package_root: Path, args: argparse.Namespace):
    env = os.environ.copy()
    model_path = str(package_root / "models")
    if env.get("GAZEBO_MODEL_PATH"):
        env["GAZEBO_MODEL_PATH"] = model_path + os.pathsep + env["GAZEBO_MODEL_PATH"]
    else:
        env["GAZEBO_MODEL_PATH"] = model_path
    if args.gazebo_master_uri:
        env["GAZEBO_MASTER_URI"] = args.gazebo_master_uri

    log_path = output_dir / "gazebo_session.log"
    log_handle = log_path.open("w", encoding="utf-8")
    cmd = [
        "ros2",
        "launch",
        "gazebo_ros",
        "gazebo.launch.py",
        f"world:={world_path}",
        f"gui:={'false' if args.headless else 'true'}",
    ]
    process = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    return process, log_handle, log_path


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()


def session_output_dir(base_dir: Path, name: str | None) -> Path:
    if name:
        return base_dir / name
    timestamp = time.strftime("session_%Y%m%d_%H%M%S")
    return base_dir / timestamp


def parse_args() -> argparse.Namespace:
    package_root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description="Launch a signless Gazebo map, manually drive the car, and record a bidirectional road graph.",
    )
    parser.add_argument("--source-world", type=Path, default=default_source_world(package_root))
    parser.add_argument("--output-base", type=Path, default=package_root / "route_recording_lab" / "outputs")
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--generated-world-name", default="autonomous_map_recording_signless.world")
    parser.add_argument("--start-corner", choices=sorted(DEFAULT_START_POSES), default="top_left")
    parser.add_argument("--generate-world-only", action="store_true")
    parser.add_argument("--no-gazebo", action="store_true", help="Attach to an already running Gazebo session.")
    parser.add_argument("--headless", action="store_true", help="Start Gazebo without the GUI client.")
    parser.add_argument("--gazebo-master-uri", default=None)
    parser.add_argument("--entity-name", default="track_preview_car")
    parser.add_argument("--cmd-topic", default="/track_preview_car/cmd_vel")
    parser.add_argument("--model-states-topic", default="/model_states")
    parser.add_argument("--set-entity-state-service", default="/set_entity_state")
    parser.add_argument("--sample-distance-m", type=float, default=0.045)
    parser.add_argument("--sample-yaw-rad", type=float, default=0.20)
    parser.add_argument("--node-merge-radius-m", type=float, default=0.085)
    parser.add_argument("--autosave-period-s", type=float, default=2.0)
    parser.add_argument("--cmd-rate-hz", type=float, default=30.0)
    parser.add_argument("--max-linear-mps", type=float, default=0.36)
    parser.add_argument("--max-reverse-mps", type=float, default=0.18)
    parser.add_argument("--max-angular-rps", type=float, default=2.8)
    parser.add_argument("--linear-step-mps", type=float, default=0.04)
    parser.add_argument("--angular-step-rps", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_root = repo_root_from_script()
    output_dir = session_output_dir(args.output_base, args.session_name)
    generated_dir = package_root / "route_recording_lab" / "generated"
    generated_world = generated_dir / args.generated_world_name
    start_pose = DEFAULT_START_POSES[args.start_corner]

    output_dir.mkdir(parents=True, exist_ok=True)
    world_info = generate_signless_world(args.source_world, generated_world, start_pose)
    with (output_dir / "signless_world_info.json").open("w", encoding="utf-8") as handle:
        json.dump(world_info, handle, indent=2)
    print(f"Generated signless world: {generated_world}")
    print(f"Removed {len(world_info['removed_road_sign_models'])} road sign models.")
    if args.generate_world_only:
        return 0

    gazebo_process = None
    gazebo_log_handle = None
    try:
        if not args.no_gazebo:
            gazebo_process, gazebo_log_handle, log_path = start_gazebo(generated_world, output_dir, package_root, args)
            print(f"Gazebo is starting. Log: {log_path}")
            time.sleep(2.0)

        rclpy.init()
        node = ManualDriveRecorder(args, output_dir)
        try:
            deadline_s = time.time() + 12.0
            while rclpy.ok() and node.current_pose is None and time.time() < deadline_s:
                rclpy.spin_once(node, timeout_sec=0.1)
            node.reset_to_pose(start_pose)
            keyboard_loop(node)
        finally:
            node.destroy_node()
            rclpy.shutdown()
    finally:
        if gazebo_log_handle is not None:
            gazebo_log_handle.close()
        if not args.no_gazebo:
            stop_process(gazebo_process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
