from __future__ import annotations

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class GazeboCameraViewer(Node):
    def __init__(self) -> None:
        super().__init__("gazebo_camera_viewer")
        self.declare_parameter("image_topic", "/track_preview_car/camera/image_raw")
        self.declare_parameter("window_name", "Track Car Camera")
        self.declare_parameter("display_width", 760)
        self.declare_parameter("display_height", 428)

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.window_name = self.get_parameter("window_name").get_parameter_value().string_value
        self.display_width = int(self.get_parameter("display_width").get_parameter_value().integer_value)
        self.display_height = int(self.get_parameter("display_height").get_parameter_value().integer_value)

        self.bridge = CvBridge()
        self.frame_received = False

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        cv2.moveWindow(self.window_name, 40, 40)
        self.show_waiting_frame()

        self.create_subscription(Image, self.image_topic, self.handle_image, 10)
        self.create_timer(0.03, self.keep_window_responsive)
        self.get_logger().info(f"Camera viewer listening on {self.image_topic}")

    def show_waiting_frame(self) -> None:
        frame = np.full((self.display_height, self.display_width, 3), (30, 34, 40), dtype=np.uint8)
        cv2.putText(
            frame,
            "Waiting for Gazebo camera...",
            (24, self.display_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 235, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def handle_image(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Could not convert camera image: {exc}")
            return

        if frame.shape[1] != self.display_width or frame.shape[0] != self.display_height:
            frame = cv2.resize(frame, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA)

        if not self.frame_received:
            self.get_logger().info("Receiving Gazebo camera frames.")
            self.frame_received = True

        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def keep_window_responsive(self) -> None:
        cv2.waitKey(1)


def main() -> None:
    rclpy.init()
    node = GazeboCameraViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            pass
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
