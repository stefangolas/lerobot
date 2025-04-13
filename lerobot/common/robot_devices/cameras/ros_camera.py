import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import time
import cv2

from lerobot.common.robot_devices.cameras.errors import RobotDeviceNotConnectedError


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

        self.bridge = CvBridge()
        self.image = None
        self.depth = None
        self.logs = {}
        self.is_connected = False

    def connect(self):
        if self.is_connected:
            raise RuntimeError(f"RosCamera({self.image_topic}) is already connected.")
        
        rospy.init_node("ros_camera_subscriber", anonymous=True)
        rospy.Subscriber(self.image_topic, Image, self._image_callback)
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
            rospy.logerr(f"Failed to decode ROS image: {e}")

    def read(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("ROS camera not connected. Call `connect()` first.")
        if self.image is None:
            raise RuntimeError("No image received yet.")
        return self.image

    def disconnect(self):
        self.is_connected = False
        self.image = None

    def async_read(self):
        return self.image
