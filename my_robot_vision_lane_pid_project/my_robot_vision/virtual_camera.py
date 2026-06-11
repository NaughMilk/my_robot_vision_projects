from __future__ import annotations

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from sensor_msgs.msg import Image

from .world import RobotPose, build_track_points, draw_static_world, make_start_pose, render_camera


class VirtualCamera(Node):
    def __init__(self) -> None:
        super().__init__("virtual_camera")

        self.declare_parameter("fps", 30.0)
        self.declare_parameter("pose_topic", "/robot_pose")
        self.declare_parameter("camera_topic", "/camera/image_raw")

        self.fps = float(self.get_parameter("fps").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.camera_topic = str(self.get_parameter("camera_topic").value)

        self.bridge = CvBridge()
        self.track_points = build_track_points()
        self.world = draw_static_world(self.track_points)
        self.pose = make_start_pose(self.track_points)

        self.pose_sub = self.create_subscription(Pose2D, self.pose_topic, self.pose_callback, 10)
        self.image_pub = self.create_publisher(Image, self.camera_topic, 10)
        self.timer = self.create_timer(1.0 / max(1.0, self.fps), self.publish_image)

        self.get_logger().info(f"Virtual camera publishing {self.camera_topic} from pose {self.pose_topic}.")

    def pose_callback(self, msg: Pose2D) -> None:
        self.pose = RobotPose(float(msg.x), float(msg.y), float(msg.theta))

    def publish_image(self) -> None:
        frame, _ = render_camera(self.world, self.pose)
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = "virtual_camera"
        self.image_pub.publish(image_msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VirtualCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
