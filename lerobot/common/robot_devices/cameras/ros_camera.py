import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import time
import cv2
from lerobot.common.utils.utils import capture_timestamp_utc
from threading import Thread


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
        self.logs = {}
        self.is_connected = False

        # ROS objects
        self.node = None
        self.subscription = None
        self.spin_thread = None

    def connect(self):
        if self.is_connected:
            raise RuntimeError(f"RosCamera({self.image_topic}) is already connected.")

        # Initialize ROS if needed
        if not rclpy.ok():
            rclpy.init()

        # Create a node
        self.node = rclpy.create_node("ros_camera_subscriber")

        # Create a subscription
        self.subscription = self.node.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            10
        )

        # Start spinning in the background so callbacks can fire
        # If you prefer spin_once in a loop, you can do that instead
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()

        self.is_connected = True
        while self.image is None:
            print("Waiting for first frame...")
            time.sleep(0.1)


    def _image_callback(self, msg: Image):
        try:
            # Convert to CV image
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # Perform optional rotation
            if self.rotation == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == -90:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif self.rotation == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)

            # Convert RGB -> BGR if needed
            if self.color_mode == "bgr" and img.ndim == 3 and img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Store the latest image
            self.image = img

            now = time.time()
            last_ts = self.logs.get("timestamp_utc", None)
            self.logs["timestamp_utc"] = now
            self.logs["delta_timestamp_s"] = now - last_ts if last_ts else 0.0

        except Exception as e:
            if self.node is not None:
                self.node.get_logger().error(f"Failed to decode ROS image: {e}")

    def read(self):
        """
        Returns the most recent image that arrived via subscription.

        If you prefer to log the read-time deltas, you can do it here
        instead of in the callback. Right now, we do it in the callback.
        """
        if not self.is_connected:
            raise RuntimeError("ROS camera not connected. Call `connect()` first.")
        if self.image is None:
            raise RuntimeError("No image received yet.")


        return self.image

    def async_read(self):
        """
        A convenience alias for read() for frameworks expecting an async_* method.
        """
        return self.read()

    def disconnect(self):
        """
        Shut down the subscription, node, and spinner thread.
        """
        if self.subscription:
            self.subscription.destroy()
            self.subscription = None

        if self.node:
            # This will also stop rclpy.spin in the background thread
            self.node.destroy_node()
            self.node = None

        if rclpy.ok():
            rclpy.shutdown()

        self.image = None
        self.is_connected = False

        if self.spin_thread is not None and self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1.0)
            self.spin_thread = None
