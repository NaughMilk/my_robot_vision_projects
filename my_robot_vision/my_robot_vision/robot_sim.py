from __future__ import annotations

import time

import cv2
import numpy as np

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D, Twist
from rclpy.node import Node
from sensor_msgs.msg import Image

from .world import (
    ACCENT_BGR,
    TEXT_BGR,
    build_track_points,
    camera_trapezoid,
    draw_robot,
    draw_static_world,
    make_start_pose,
    nearest_track_point,
    update_pose,
)


def as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class RobotSimulator(Node):
    def __init__(self) -> None:
        super().__init__("robot_sim")

        self.declare_parameter("fps", 30.0)
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("pose_topic", "/robot_pose")
        self.declare_parameter("show_gui", True)
        self.declare_parameter("publish_world_image", True)

        self.fps = float(self.get_parameter("fps").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.show_gui = as_bool(self.get_parameter("show_gui").value)
        self.publish_world_image = as_bool(self.get_parameter("publish_world_image").value)

        self.bridge = CvBridge()
        self.track_points = build_track_points()
        self.world = draw_static_world(self.track_points)
        self.pose = make_start_pose(self.track_points)
        self.history: list[tuple[int, int]] = [(int(self.pose.x), int(self.pose.y))]

        self.linear_cmd = 0.0
        self.angular_cmd = 0.0
        self.last_cmd_time = time.monotonic()
        self.elapsed = 0.0
        self.progress_index = 0

        self.cmd_sub = self.create_subscription(Twist, self.cmd_topic, self.cmd_callback, 10)
        self.pose_pub = self.create_publisher(Pose2D, self.pose_topic, 10)
        self.world_pub = self.create_publisher(Image, "/robot_sim/world_image", 10)
        self.timer = self.create_timer(1.0 / max(1.0, self.fps), self.step)

        self.get_logger().info(f"Robot simulator listening to {self.cmd_topic} and publishing {self.pose_topic}.")

    def cmd_callback(self, msg: Twist) -> None:
        self.linear_cmd = float(msg.linear.x)
        self.angular_cmd = float(msg.angular.z)
        self.last_cmd_time = time.monotonic()

    def step(self) -> None:
        dt = 1.0 / max(1.0, self.fps)
        self.elapsed += dt

        if time.monotonic() - self.last_cmd_time > 0.8:
            linear_cmd = 0.0
            angular_cmd = 0.0
        else:
            linear_cmd = self.linear_cmd
            angular_cmd = self.angular_cmd

        update_pose(self.pose, linear_cmd, angular_cmd, dt)
        self.history.append((int(self.pose.x), int(self.pose.y)))
        self.publish_pose()

        panel = self.draw_panel(linear_cmd, angular_cmd)
        if self.publish_world_image:
            msg = self.bridge.cv2_to_imgmsg(panel, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "virtual_world"
            self.world_pub.publish(msg)

        if self.show_gui:
            cv2.imshow("Robot Simulator", panel)
            cv2.waitKey(1)

    def publish_pose(self) -> None:
        msg = Pose2D()
        msg.x = float(self.pose.x)
        msg.y = float(self.pose.y)
        msg.theta = float(self.pose.heading)
        self.pose_pub.publish(msg)

    def draw_panel(self, linear_cmd: float, angular_cmd: float) -> np.ndarray:
        panel = self.world.copy()

        if len(self.history) > 1:
            cv2.polylines(panel, [np.array(self.history, dtype=np.int32)], False, (255, 120, 0), 4, cv2.LINE_AA)

        cam_poly = np.round(camera_trapezoid(self.pose)).astype(np.int32)
        cv2.polylines(panel, [cam_poly], True, ACCENT_BGR, 3, cv2.LINE_AA)
        draw_robot(panel, self.pose)

        idx, nearest_pt, cross_track = nearest_track_point(self.track_points, self.pose)
        self.progress_index = max(self.progress_index, idx)
        cv2.line(
            panel,
            (int(self.pose.x), int(self.pose.y)),
            (int(nearest_pt[0]), int(nearest_pt[1])),
            (140, 240, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(panel, (int(nearest_pt[0]), int(nearest_pt[1])), 8, (140, 240, 255), -1)

        progress = 100.0 * self.progress_index / max(1, len(self.track_points) - 1)
        lines = [
            "ROS2 Virtual Robot Simulator",
            f"linear.x={linear_cmd:+.2f}  angular.z={angular_cmd:+.2f}",
            f"progress={progress:5.1f}%  cross_track={cross_track:5.1f}px",
            f"pose x={self.pose.x:6.1f} y={self.pose.y:6.1f} theta={self.pose.heading:+.2f}",
        ]

        cv2.rectangle(panel, (20, 20), (850, 150), (18, 21, 26), -1)
        cv2.rectangle(panel, (20, 20), (850, 150), (70, 80, 92), 2)
        for i, line in enumerate(lines):
            cv2.putText(panel, line, (38, 55 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, TEXT_BGR, 2, cv2.LINE_AA)

        return cv2.resize(panel, (1200, 720), interpolation=cv2.INTER_AREA)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RobotSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
