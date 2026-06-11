#!/usr/bin/env python3
"""Capture ROS2 sensor_msgs/Image frames to image files."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    encoding = (msg.encoding or "").lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in {"bgr8", "rgb8"}:
        row = raw.reshape((msg.height, msg.step))[:, : msg.width * 3]
        image = row.reshape((msg.height, msg.width, 3))
        if encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    if encoding in {"bgra8", "rgba8"}:
        row = raw.reshape((msg.height, msg.step))[:, : msg.width * 4]
        image = row.reshape((msg.height, msg.width, 4))
        code = cv2.COLOR_BGRA2BGR if encoding == "bgra8" else cv2.COLOR_RGBA2BGR
        return cv2.cvtColor(image, code)

    if encoding in {"mono8", "8uc1"}:
        row = raw.reshape((msg.height, msg.step))[:, : msg.width]
        return cv2.cvtColor(row, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported image encoding: {msg.encoding!r}")


class ImageCapture(Node):
    def __init__(
        self,
        topic: str,
        output_dir: Path,
        max_images: int,
        min_interval: float,
        jpeg_quality: int,
    ) -> None:
        super().__init__("capture_ros_images")
        self.output_dir = output_dir
        self.max_images = max_images
        self.min_interval = min_interval
        self.jpeg_quality = jpeg_quality
        self.saved = 0
        self.last_save_time = 0.0
        self.last_message_time = 0.0
        self.subscription = self.create_subscription(Image, topic, self._on_image, 10)
        self.get_logger().info(f"Capturing {topic} -> {output_dir}")

    def _on_image(self, msg: Image) -> None:
        now = time.monotonic()
        self.last_message_time = now
        if now - self.last_save_time < self.min_interval:
            return

        image = image_msg_to_bgr(msg)
        self.saved += 1
        self.last_save_time = now
        output_path = self.output_dir / f"frame_{self.saved:04d}.jpg"
        ok = cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError(f"Failed to write {output_path}")
        self.get_logger().info(f"saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/track_preview_car/camera/image_raw")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-images", type=int, default=40)
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--min-interval", type=float, default=0.5)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = ImageCapture(
        args.topic,
        args.output_dir,
        args.max_images,
        args.min_interval,
        args.jpeg_quality,
    )
    start = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            elapsed = time.monotonic() - start
            if node.saved >= args.max_images:
                break
            if elapsed >= args.seconds:
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if node.saved == 0:
        print("capture_done saved=0")
        return 2

    print(f"capture_done saved={node.saved} output_dir={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
