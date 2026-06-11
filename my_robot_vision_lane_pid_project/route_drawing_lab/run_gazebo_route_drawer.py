#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import rclpy
    from gazebo_msgs.msg import ModelStates
    from gazebo_msgs.srv import DeleteEntity, SpawnEntity
    from rclpy.node import Node
except ImportError as exc:  # pragma: no cover - user environment guard
    print(
        "Could not import ROS2/Gazebo modules. Run:\n"
        "  source /opt/ros/foxy/setup.bash\n"
        "  source /home/quocbao/my_robot_vision_full_project/install/setup.bash",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


MODEL_SIZE_M = 6.0
MAP_TEXTURE_SIZE = 256.0
MAP_POSE_X_M = 0.0
MAP_POSE_Y_M = 0.3
MARKER_PREFIX = "route_draw_wp_"
EDGE_PREFIX = "route_draw_edge_"


@dataclass
class Waypoint:
    name: str
    x: float
    y: float
    z: float = 0.08


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def model_to_code_px(x: float, y: float) -> tuple[float, float]:
    px = (x / MODEL_SIZE_M + 0.5) * MAP_TEXTURE_SIZE
    py = (0.5 - y / MODEL_SIZE_M) * MAP_TEXTURE_SIZE
    return px, py


def code_px_to_model(px: float, py: float) -> tuple[float, float]:
    x = (px / MAP_TEXTURE_SIZE - 0.5) * MODEL_SIZE_M
    y = (0.5 - py / MAP_TEXTURE_SIZE) * MODEL_SIZE_M
    return x, y


def model_to_visual_mirror_px(x: float, y: float) -> tuple[float, float]:
    local_mesh_x = -(x - MAP_POSE_X_M)
    local_mesh_y = y - MAP_POSE_Y_M
    px = (local_mesh_x / MODEL_SIZE_M + 0.5) * MAP_TEXTURE_SIZE
    py = (0.5 - local_mesh_y / MODEL_SIZE_M) * MAP_TEXTURE_SIZE
    return px, py


def marker_sdf(radius: float = 0.055) -> str:
    return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="route_draw_waypoint">
    <static>1</static>
    <link name="marker_link">
      <visual name="marker_visual">
        <geometry>
          <sphere><radius>{radius:.4f}</radius></sphere>
        </geometry>
        <material>
          <ambient>1 0.9 0 1</ambient>
          <diffuse>1 0.9 0 1</diffuse>
          <emissive>0.7 0.45 0 1</emissive>
        </material>
      </visual>
      <collision name="marker_collision">
        <geometry>
          <sphere><radius>{radius:.4f}</radius></sphere>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
"""


def edge_sdf(length: float, thickness: float = 0.024) -> str:
    length = max(float(length), 0.001)
    return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="route_draw_edge">
    <static>1</static>
    <link name="edge_link">
      <visual name="edge_visual">
        <geometry>
          <box><size>{length:.6f} {thickness:.6f} {thickness:.6f}</size></box>
        </geometry>
        <material>
          <ambient>1 0 1 1</ambient>
          <diffuse>1 0 1 1</diffuse>
          <emissive>0.8 0 0.8 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def pose_xml(x: float, y: float, z: float, yaw: float = 0.0) -> str:
    return f"{x:.6f} {y:.6f} {z:.6f} 0 0 {yaw:.6f}"


def write_drawing_world(world_path: Path, map_mesh: Path) -> None:
    world_path.parent.mkdir(parents=True, exist_ok=True)
    world_path.write_text(
        f"""<?xml version="1.0"?>
<sdf version="1.7">
  <world name="route_drawing_world">
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <update_rate>30.0</update_rate>
    </plugin>
    <scene>
      <ambient>0.85 0.85 0.85 1</ambient>
      <background>0.86 0.91 0.98 1</background>
      <shadows>0</shadows>
    </scene>
    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>0 0 7.5 0 1.570796 1.570796</pose>
        <view_controller>orbit</view_controller>
      </camera>
    </gui>
    <light name="sun" type="directional">
      <cast_shadows>0</cast_shadows>
      <pose>0 0 8 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <direction>0 0 -1</direction>
    </light>
    <model name="autonomous_map_3d_mirror_x_view">
      <static>1</static>
      <pose>{pose_xml(MAP_POSE_X_M, MAP_POSE_Y_M, 0.0)}</pose>
      <link name="map_link">
        <visual name="mirrored_map_visual">
          <geometry>
            <mesh>
              <uri>{map_mesh.resolve().as_uri()}</uri>
              <scale>-1 1 1</scale>
            </mesh>
          </geometry>
        </visual>
      </link>
    </model>
  </world>
</sdf>
""",
        encoding="utf-8",
    )


def default_template_points(template: str) -> list[tuple[float, float]]:
    templates_px = {
        "middle": [
            (35.0, 118.0),
            (62.0, 118.0),
            (106.0, 118.0),
            (148.0, 118.0),
            (221.0, 118.0),
            (221.0, 150.0),
        ],
        "top_cut": [
            (148.0, 35.0),
            (148.0, 76.0),
            (148.0, 113.0),
            (221.0, 113.0),
            (221.0, 150.0),
        ],
        "lower_cut": [
            (126.0, 221.0),
            (126.0, 190.0),
            (106.0, 190.0),
            (106.0, 113.0),
            (62.0, 113.0),
            (35.0, 128.0),
        ],
        "empty": [],
    }
    return [code_px_to_model(px, py) for px, py in templates_px[template]]


class RouteDrawingNode(Node):
    def __init__(self, output_dir: Path, initial_points: list[tuple[float, float]]) -> None:
        super().__init__("route_drawing_lab")
        self.output_dir = output_dir
        self.waypoints: list[Waypoint] = [
            Waypoint(f"{MARKER_PREFIX}{index:03d}", x, y)
            for index, (x, y) in enumerate(initial_points)
        ]
        self.latest_pose_by_name: dict[str, tuple[float, float, float]] = {}
        self.edge_count = 0
        self.last_edge_signature: tuple[tuple[float, float], ...] | None = None
        self.last_edge_update_s = 0.0
        self.command_queue: queue.Queue[str] = queue.Queue()

        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete_client = self.create_client(DeleteEntity, "/delete_entity")
        self.create_subscription(ModelStates, "/model_states", self.handle_model_states, 10)

    def wait_for_services(self) -> None:
        for client, name in ((self.spawn_client, "/spawn_entity"), (self.delete_client, "/delete_entity")):
            while rclpy.ok() and not client.wait_for_service(timeout_sec=0.5):
                self.get_logger().info(f"Waiting for {name}...")

    def spawn_model(self, name: str, sdf: str, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        request = SpawnEntity.Request()
        request.name = name
        request.xml = sdf
        request.robot_namespace = ""
        request.initial_pose.position.x = float(x)
        request.initial_pose.position.y = float(y)
        request.initial_pose.position.z = float(z)
        request.initial_pose.orientation.z = math.sin(yaw * 0.5)
        request.initial_pose.orientation.w = math.cos(yaw * 0.5)
        future = self.spawn_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

    def delete_model(self, name: str) -> None:
        request = DeleteEntity.Request()
        request.name = name
        future = self.delete_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

    def spawn_initial_waypoints(self) -> None:
        for waypoint in self.waypoints:
            self.delete_model(waypoint.name)
            self.spawn_model(waypoint.name, marker_sdf(), waypoint.x, waypoint.y, waypoint.z)
        self.update_edges(force=True)

    def handle_model_states(self, msg: ModelStates) -> None:
        name_to_index = {name: index for index, name in enumerate(msg.name)}
        for waypoint in self.waypoints:
            index = name_to_index.get(waypoint.name)
            if index is None:
                continue
            pose = msg.pose[index]
            waypoint.x = float(pose.position.x)
            waypoint.y = float(pose.position.y)
            waypoint.z = max(0.08, float(pose.position.z))
            self.latest_pose_by_name[waypoint.name] = (waypoint.x, waypoint.y, waypoint.z)

    def edge_signature(self) -> tuple[tuple[float, float], ...]:
        return tuple((round(waypoint.x, 3), round(waypoint.y, 3)) for waypoint in self.waypoints)

    def maybe_update_edges(self) -> None:
        signature = self.edge_signature()
        if signature != self.last_edge_signature and time.monotonic() - self.last_edge_update_s >= 0.35:
            self.update_edges(force=True)

    def update_edges(self, force: bool = False) -> None:
        signature = self.edge_signature()
        if not force and signature == self.last_edge_signature:
            return
        for edge_index in range(max(self.edge_count, max(0, len(self.waypoints) - 1))):
            name = f"{EDGE_PREFIX}{edge_index:03d}"
            self.delete_model(name)
        self.edge_count = max(0, len(self.waypoints) - 1)
        for edge_index, (start, end) in enumerate(zip(self.waypoints, self.waypoints[1:])):
            dx = end.x - start.x
            dy = end.y - start.y
            length = math.hypot(dx, dy)
            if length < 1e-4:
                continue
            self.spawn_model(
                f"{EDGE_PREFIX}{edge_index:03d}",
                edge_sdf(length),
                (start.x + end.x) * 0.5,
                (start.y + end.y) * 0.5,
                0.055,
                math.atan2(dy, dx),
            )
        self.last_edge_signature = signature
        self.last_edge_update_s = time.monotonic()

    def add_waypoint(self, x: float, y: float, index: int | None = None) -> None:
        if index is None:
            index = len(self.waypoints)
        index = max(0, min(index, len(self.waypoints)))
        waypoint = Waypoint(f"{MARKER_PREFIX}{int(time.time() * 1000) % 1000000:06d}", x, y)
        self.waypoints.insert(index, waypoint)
        self.spawn_model(waypoint.name, marker_sdf(), waypoint.x, waypoint.y, waypoint.z)
        self.update_edges(force=True)

    def delete_waypoint(self, index: int) -> None:
        if index < 0 or index >= len(self.waypoints):
            self.get_logger().warn(f"No waypoint at index {index}.")
            return
        waypoint = self.waypoints.pop(index)
        self.delete_model(waypoint.name)
        self.update_edges(force=True)

    def save(self, label: str = "drawn_route") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"{label}.json"
        csv_path = self.output_dir / f"{label}.csv"
        png_path = self.output_dir / f"{label}_overlay.png"

        payload_waypoints = []
        for index, waypoint in enumerate(self.waypoints):
            code_px = model_to_code_px(waypoint.x, waypoint.y)
            visual_px = model_to_visual_mirror_px(waypoint.x, waypoint.y)
            payload_waypoints.append(
                {
                    "index": index,
                    "name": waypoint.name,
                    "model": {"x": waypoint.x, "y": waypoint.y, "z": waypoint.z},
                    "code_px_no_offset": {"x": code_px[0], "y": code_px[1]},
                    "visual_mirror_px": {"x": visual_px[0], "y": visual_px[1]},
                }
            )

        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "map": {
                "model_size_m": MODEL_SIZE_M,
                "texture_size_px": MAP_TEXTURE_SIZE,
                "pose_m": [MAP_POSE_X_M, MAP_POSE_Y_M, 0.0],
                "mirror_axis": "x",
            },
            "waypoints": payload_waypoints,
            "edges": [
                {"start_index": index, "end_index": index + 1}
                for index in range(max(0, len(self.waypoints) - 1))
            ],
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "name", "model_x_m", "model_y_m", "model_z_m", "code_px", "code_py", "visual_px", "visual_py"])
            for row in payload_waypoints:
                writer.writerow(
                    [
                        row["index"],
                        row["name"],
                        row["model"]["x"],
                        row["model"]["y"],
                        row["model"]["z"],
                        row["code_px_no_offset"]["x"],
                        row["code_px_no_offset"]["y"],
                        row["visual_mirror_px"]["x"],
                        row["visual_mirror_px"]["y"],
                    ]
                )

        self.write_overlay_png(png_path)
        self.get_logger().info(f"Saved route drawing to {json_path}")
        return json_path

    def write_overlay_png(self, path: Path) -> None:
        map_path = repo_root() / "maps" / "autonomous_map_visual.png"
        image = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
        if image is None:
            return
        image = cv2.flip(image, 1)
        points = []
        for waypoint in self.waypoints:
            px, py = model_to_visual_mirror_px(waypoint.x, waypoint.y)
            points.append([int(round(px)), int(round(py))])
        if len(points) >= 2:
            polyline = np.array(points, dtype=np.int32)
            cv2.polylines(image, [polyline], False, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.polylines(image, [polyline], False, (255, 0, 255), 1, cv2.LINE_AA)
        for index, point in enumerate(points):
            cv2.circle(image, tuple(point), 4, (0, 230, 255), -1, cv2.LINE_AA)
            cv2.putText(image, str(index), (point[0] + 4, point[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(path), image)

    def print_waypoints(self) -> None:
        for index, waypoint in enumerate(self.waypoints):
            code_px = model_to_code_px(waypoint.x, waypoint.y)
            visual_px = model_to_visual_mirror_px(waypoint.x, waypoint.y)
            print(
                f"{index:02d} {waypoint.name} model=({waypoint.x:+.3f},{waypoint.y:+.3f}) "
                f"code_px=({code_px[0]:.1f},{code_px[1]:.1f}) visual_px=({visual_px[0]:.1f},{visual_px[1]:.1f})"
            )

    def handle_command(self, command: str) -> bool:
        parts = command.strip().split()
        if not parts:
            return True
        op = parts[0].lower()
        try:
            if op in {"q", "quit", "exit"}:
                self.save("drawn_route")
                return False
            if op in {"h", "help"}:
                print(HELP_TEXT)
            elif op in {"p", "print", "list"}:
                self.print_waypoints()
            elif op == "save":
                label = parts[1] if len(parts) > 1 else "drawn_route"
                self.save(label)
            elif op == "add":
                self.add_waypoint(float(parts[1]), float(parts[2]))
            elif op == "addpx":
                x, y = code_px_to_model(float(parts[1]), float(parts[2]))
                self.add_waypoint(x, y)
            elif op == "insert":
                self.add_waypoint(float(parts[2]), float(parts[3]), int(parts[1]))
            elif op == "insertpx":
                x, y = code_px_to_model(float(parts[2]), float(parts[3]))
                self.add_waypoint(x, y, int(parts[1]))
            elif op in {"del", "delete", "rm"}:
                self.delete_waypoint(int(parts[1]))
            elif op == "refresh":
                self.update_edges(force=True)
            else:
                print(f"Unknown command: {command!r}. Type 'help'.")
        except (IndexError, ValueError) as exc:
            print(f"Bad command: {command!r} ({exc}). Type 'help'.")
        return True


HELP_TEXT = """
Commands:
  help                  show this help
  print                 print waypoint coordinates
  save [label]          save JSON/CSV/PNG into this session folder
  add X Y               add waypoint in Gazebo model meters
  addpx PX PY           add waypoint in route/code pixels
  insert I X Y          insert waypoint at index I in model meters
  insertpx I PX PY      insert waypoint at index I in route/code pixels
  delete I              delete waypoint index I
  refresh               rebuild connecting line segments
  quit                  save and quit

In Gazebo:
  Select a yellow waypoint sphere, press the translate tool, drag it on the map.
  The magenta line reconnects the waypoints in index order.
""".strip()


def stdin_reader(command_queue: queue.Queue[str]) -> None:
    for line in sys.stdin:
        command_queue.put(line)


def launch_gazebo(world_path: Path, headless: bool) -> subprocess.Popen | None:
    if headless:
        return subprocess.Popen(
            [
                "gzserver",
                str(world_path),
                "-s",
                "libgazebo_ros_init.so",
                "-s",
                "libgazebo_ros_factory.so",
            ],
        )
    return subprocess.Popen(
        [
            "ros2",
            "launch",
            "gazebo_ros",
            "gazebo.launch.py",
            f"world:={world_path}",
            "gui:=true",
        ]
    )


def parse_args() -> argparse.Namespace:
    package_root = repo_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Draw and save a connected route directly from Gazebo waypoint markers.")
    parser.add_argument("--template", choices=("middle", "top_cut", "lower_cut", "empty"), default="middle")
    parser.add_argument("--output-dir", type=Path, default=package_root / "route_drawing_lab" / "outputs" / f"session_{timestamp}")
    parser.add_argument("--generate-world-only", action="store_true")
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_root = repo_root()
    world_path = args.output_dir / "route_drawing_world.world"
    map_mesh = package_root / "models" / "autonomous_map_3d" / "meshes" / "autonomous_map_3d.dae"
    write_drawing_world(world_path, map_mesh)
    print(f"Drawing world: {world_path}")
    if args.generate_world_only:
        return 0

    process = None if args.no_launch else launch_gazebo(world_path, args.headless)
    if process is not None:
        print("Waiting a few seconds for Gazebo services...")
        time.sleep(4.0)

    rclpy.init()
    node = RouteDrawingNode(args.output_dir, default_template_points(args.template))
    keep_running = True
    try:
        node.wait_for_services()
        node.spawn_initial_waypoints()
        node.print_waypoints()
        print(HELP_TEXT)
        threading.Thread(target=stdin_reader, args=(node.command_queue,), daemon=True).start()
        while rclpy.ok() and keep_running:
            rclpy.spin_once(node, timeout_sec=0.1)
            node.maybe_update_edges()
            while not node.command_queue.empty():
                keep_running = node.handle_command(node.command_queue.get_nowait())
                if not keep_running:
                    break
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
