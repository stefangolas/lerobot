import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import time
import cv2



class RosCamera:
    def __init__(self, config):
        self.config = config
        self.image_topic = config.image_topic
        self.rotation = config.rotation
        self.mock = config.mock
        self.width = config.width
        self.height = config.height
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.channels = 3
        self.fps = config.fps or 30



        self.bridge = CvBridge()
        self.image = None
        self.depth = None
        self.logs = {}
        self.is_connected = False

        self.node = None
        self.subscription = None
        self.spin_thread = None

    def connect(self):
        if self.is_connected:
            raise RuntimeError(f"RosCamera({self.image_topic}) is already connected.")

        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node("ros_camera_subscriber")
        self.subscription = self.node.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            10
        )

        # Start spinning in a background thread
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()

        self.is_connected = True

    def _image_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            if self.rotation == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == -90:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif self.rotation == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)

            if self.color_mode == "bgr" and img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            self.image = img
            self.logs["timestamp_utc"] = time.time()

        except Exception as e:
            self.node.get_logger().error(f"Failed to decode ROS image: {e}")

    def read(self):
        if not self.is_connected:
            raise Exception("ROS camera not connected. Call `connect()` first.")
        if self.image is None:
            raise RuntimeError("No image received yet.")
        return self.image

    def disconnect(self):
        if self.subscription:
            self.subscription.destroy()
        if self.node:
            self.node.destroy_node()
        self.image = None
        self.is_connected = False
        self.subscription = None
        self.node = None
        if rclpy.ok():
            rclpy.shutdown()

    def async_read(self):
        return self.image
