from __future__ import annotations

import json
import math
from pathlib import Path

import rclpy
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import SetEntityState
from rclpy.node import Node
from std_msgs.msg import String


TRAFFIC_LIGHT_CYCLE_ORDER = ("green", "yellow", "red")


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


class TrafficLightVisualCycle(Node):
    def __init__(self) -> None:
        super().__init__("gazebo_traffic_light_visual_cycle")
        self.declare_parameter("route_config", "")
        self.declare_parameter("green_duration_s", 5.0)
        self.declare_parameter("yellow_duration_s", 1.0)
        self.declare_parameter("red_duration_s", 8.0)
        self.declare_parameter("cycle_enabled", True)
        self.declare_parameter("initial_state", "green")
        self.declare_parameter("indicator_hidden_pose", [0.0, 0.0, -5.0])
        self.declare_parameter("publish_topic", "/lane_pid_experiment/visual_traffic_light_state")
        self.declare_parameter("update_period_s", 0.05)

        self.route_config_path = str(self.get_parameter("route_config").value).strip()
        self.scene_default_poses: dict[str, tuple[float, float, float, float]] = {}
        self.indicator_models: dict[str, tuple[tuple[str, str, float], ...]] = {}
        self.indicator_local_xy_m = (0.0, -0.038)
        self.indicator_hidden_pose = tuple(float(value) for value in self.get_parameter("indicator_hidden_pose").value)
        self.load_traffic_light_config()

        self.visual_client = self.create_client(SetEntityState, "/set_entity_state")
        self.model_states_sub = self.create_subscription(ModelStates, "/model_states", self.handle_model_states, 10)
        self.state_pub = self.create_publisher(String, str(self.get_parameter("publish_topic").value), 10)
        self.scene_poses = dict(self.scene_default_poses)
        self.last_state: str | None = None
        self.last_signature: tuple[tuple[str, float, float, float, float], ...] | None = None
        self.wait_logged = False
        self.start_time = self.get_clock().now().nanoseconds * 1e-9
        self.create_timer(float(self.get_parameter("update_period_s").value), self.update_visuals)

        self.get_logger().info(
            f"Traffic-light visual cycle loaded {len(self.scene_default_poses)} scene models "
            f"from {self.route_config_path or 'built-in defaults'}; "
            "this node only updates simulation visuals."
        )

    def load_traffic_light_config(self) -> None:
        if not self.route_config_path:
            self.get_logger().warn("route_config is empty; traffic-light visual cycle has no configured lamps.")
            return
        path = Path(self.route_config_path).expanduser()
        if not path.is_file():
            self.get_logger().warn(f"route_config does not exist: {path}")
            return
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        traffic = data.get("traffic_lights", {})
        if not isinstance(traffic, dict):
            return

        scene_models = traffic.get("scene_models", ())
        if isinstance(scene_models, list):
            for entry in scene_models:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name", ""))
                pose = entry.get("default_pose", ())
                if name and isinstance(pose, list) and len(pose) == 4:
                    self.scene_default_poses[name] = tuple(float(value) for value in pose)

        local_xy = traffic.get("indicator_local_xy_m")
        if isinstance(local_xy, list) and len(local_xy) == 2:
            self.indicator_local_xy_m = (float(local_xy[0]), float(local_xy[1]))

        hidden_pose = traffic.get("indicator_hidden_pose")
        if isinstance(hidden_pose, list) and len(hidden_pose) == 3:
            self.indicator_hidden_pose = tuple(float(value) for value in hidden_pose)

        indicator_models = traffic.get("indicator_models", {})
        if isinstance(indicator_models, dict):
            parsed: dict[str, tuple[tuple[str, str, float], ...]] = {}
            for state, entries in indicator_models.items():
                state_entries: list[tuple[str, str, float]] = []
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, list) and len(entry) == 3:
                            state_entries.append((str(entry[0]), str(entry[1]), float(entry[2])))
                parsed[str(state)] = tuple(state_entries)
            self.indicator_models = parsed

    def handle_model_states(self, msg: ModelStates) -> None:
        name_to_index = {name: index for index, name in enumerate(msg.name)}
        for name in self.scene_default_poses:
            index = name_to_index.get(name)
            if index is None:
                continue
            pose = msg.pose[index]
            self.scene_poses[name] = (
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
                normalize_angle(quaternion_to_yaw(pose.orientation)),
            )

    def current_state(self) -> str:
        initial_state = str(self.get_parameter("initial_state").value).lower()
        if initial_state not in TRAFFIC_LIGHT_CYCLE_ORDER:
            initial_state = "green"
        if not bool(self.get_parameter("cycle_enabled").value):
            return initial_state
        now_s = self.get_clock().now().nanoseconds * 1e-9
        elapsed = max(0.0, now_s - self.start_time)
        durations = {
            "green": max(0.1, float(self.get_parameter("green_duration_s").value)),
            "yellow": max(0.1, float(self.get_parameter("yellow_duration_s").value)),
            "red": max(0.1, float(self.get_parameter("red_duration_s").value)),
        }
        phase = elapsed % sum(durations.values())
        for state in TRAFFIC_LIGHT_CYCLE_ORDER:
            if phase < durations[state]:
                return state
            phase -= durations[state]
        return "red"

    def scene_signature(self) -> tuple[tuple[str, float, float, float, float], ...]:
        signature = []
        for name, pose in sorted(self.scene_poses.items()):
            x, y, z, yaw = pose
            signature.append((name, round(x, 4), round(y, 4), round(z, 4), round(yaw, 4)))
        return tuple(signature)

    def indicator_pose(self, parent_name: str, local_z: float) -> tuple[float, float, float, float]:
        parent_x, parent_y, parent_z, parent_yaw = self.scene_poses.get(
            parent_name,
            self.scene_default_poses.get(parent_name, (0.0, 0.0, 0.0, 0.0)),
        )
        local_x, local_y = self.indicator_local_xy_m
        cosine = math.cos(parent_yaw)
        sine = math.sin(parent_yaw)
        x = parent_x + local_x * cosine - local_y * sine
        y = parent_y + local_x * sine + local_y * cosine
        return x, y, parent_z + local_z, parent_yaw

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

    def update_visuals(self) -> None:
        state = self.current_state()
        signature = self.scene_signature()
        if state == self.last_state and signature == self.last_signature:
            self.publish_state(state)
            return
        if not self.visual_client.service_is_ready():
            self.visual_client.wait_for_service(timeout_sec=0.0)
            if not self.wait_logged:
                self.get_logger().warn("Waiting for /set_entity_state before moving traffic-light visual lamps.")
                self.wait_logged = True
            return

        hidden_x, hidden_y, hidden_z = self.indicator_hidden_pose
        for lamp_state, entries in self.indicator_models.items():
            active = lamp_state == state
            for model_name, parent_name, local_z in entries:
                request = SetEntityState.Request()
                if active:
                    x, y, z, yaw = self.indicator_pose(parent_name, local_z)
                    request.state = self.make_entity_state(model_name, x, y, z, yaw)
                else:
                    request.state = self.make_entity_state(model_name, hidden_x, hidden_y, hidden_z)
                self.visual_client.call_async(request)

        if state != self.last_state:
            self.get_logger().info(f"visual traffic light is now {state.upper()}.")
        self.last_state = state
        self.last_signature = signature
        self.publish_state(state)

    def publish_state(self, state: str) -> None:
        message = String()
        message.data = json.dumps(
            {
                "stamp_ns": self.get_clock().now().nanoseconds,
                "state": state,
                "source": "visual_cycle_only",
                "for_camera_detector_ground_truth": False,
            },
            sort_keys=True,
        )
        self.state_pub.publish(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TrafficLightVisualCycle()
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
