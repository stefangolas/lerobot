import time
import math
import traceback
import logging
import numpy as np

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from copy import deepcopy
from collections.abc import Sequence
from dynamixel import MODEL_CONTROL_TABLE, MODEL_BAUDRATE_TABLE, MODEL_RESOLUTION


PROTOCOL_VERSION = 2.0
TIMEOUT_MS = 50
NUM_READ_RETRY = 3
NUM_WRITE_RETRY = 3
MAX_ID_RANGE = 254

# Example placeholders so the code structure remains similar:
HALF_TURN_DEGREE = 180.0
LOWER_BOUND_DEGREE = -270.0
UPPER_BOUND_DEGREE = 270.0
LOWER_BOUND_LINEAR = -10.0
UPPER_BOUND_LINEAR = 110.0

class RobotDeviceAlreadyConnectedError(Exception):
    pass

class RobotDeviceNotConnectedError(Exception):
    pass

class ConnectionError(Exception):
    pass

class JointOutOfRangeError(Exception):
    pass

class CalibrationMode:
    DEGREE = "DEGREE"
    LINEAR = "LINEAR"

class RosMotorsBus:
    """
    The RosMotorsBus class mirrors the same structure as DynamixelMotorsBus,
    but uses ROS publishers/subscribers to communicate with a simulated
    robot (e.g., in Isaac Sim) instead of hardware over a serial port.

    Example usage:
    ```python
    config = RosMotorsBusConfig(
        command_topic="/joint_group_position_controller/command",
        state_topic="/joint_states",
        motors={"joint1": [1, "fake_model"], "joint2": [2, "fake_model"]},
    )
    motors_bus = RosMotorsBus(config)
    motors_bus.connect()

    positions = motors_bus.read("Present_Position")
    motors_bus.write("Goal_Position", positions + 0.1)  # small offset
    motors_bus.disconnect()
    ```
    """

    def __init__(self, config):
        self.port = config.port  # Typically "ROS" or a dummy
        self.motors = config.motors
        self.mock = config.mock

        # Borrowing the structure from the original code:
        self.model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
        self.model_resolution = deepcopy(MODEL_RESOLUTION)

        self.is_connected = False

        # Instead of groupSyncRead/Write, we'll track everything in local dicts:
        self.group_readers = {}
        self.group_writers = {}

        self.logs = {}

        # For storing calibration info if you want (as done in Dynamixel):
        self.calibration = None

        # We'll store the latest joint states from ROS in a dict:
        # key: joint_name -> value: float position
        self.joint_positions = {}

        # ROS pub/sub handles:
        self.cmd_pub = None
        self.state_sub = None

        self.command_topic = config.command_topic
        self.state_topic = config.state_topic

    def connect(self):
        """
        Opens the 'connection' to the ROS environment.
        In hardware code, you'd open the serial port. Here, we init a ROS node, 
        set up publishers/subscribers, and mark ourselves connected.
        """
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"RosMotorsBus({self.port}) is already connected. Do not call `motors_bus.connect()` twice."
            )

        # If a node is not already initialized:
        if not rospy.core.is_initialized():
            rospy.init_node("ros_motors_bus", anonymous=True)

        self.cmd_pub = rospy.Publisher(self.command_topic, JointState, queue_size=1)
        self.state_sub = rospy.Subscriber(self.state_topic, JointState, self._on_joint_state)

        self.is_connected = True
        # In a hardware scenario, you'd do baud rate setup, etc. here.

    def _on_joint_state(self, msg):
        """
        ROS callback for receiving joint positions from Isaac Sim or real robot.
        We'll store them in a dictionary for subsequent 'read()' calls.
        """
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = pos

    def reconnect(self):
        """
        Equivalent to re-opening the port if it was closed or lost connection.
        For ROS, we basically re-init the publisher/subscriber if not connected.
        """
        if self.is_connected:
            logging.warning("Already connected. Reconnect call is unnecessary.")
            return

        self.connect()

    def disconnect(self):
        """
        Closes the 'connection.' For hardware, we'd close the serial port.
        For ROS, we can unregister the subscriber/publisher if desired.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"RosMotorsBus({self.port}) is not connected. Try running `motors_bus.connect()` first."
            )

        if self.state_sub:
            self.state_sub.unregister()
            self.state_sub = None
        self.cmd_pub = None

        self.is_connected = False

    def read_with_motor_ids(self, motor_models, motor_ids, data_name, num_retry=NUM_READ_RETRY):
        """
        In the Dynamixel code, this does a GroupSyncRead on specific motor IDs.
        Here, we don't truly 'poll' the robot. We rely on our local dictionary.

        We'll replicate the structure to keep the code parallel.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"RosMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        if not isinstance(motor_ids, list):
            motor_ids = [motor_ids]

        # There's no real group read in ROS. We'll just gather from self.joint_positions:
        values = []
        for idx in motor_ids:
            # For demonstration, let's pretend 'idx' maps to a motor name or we
            # find the name in self.motors that matches 'idx'.
            # In real code, you'd do a name->id mapping. Here, we keep it simple:
            name = None
            for k, (m_id, m_model) in self.motors.items():
                if m_id == idx:
                    name = k
                    break

            if name is None:
                # means motor not found
                continue
            val = self.joint_positions.get(name, 0.0)
            values.append(val)

        values = np.array(values)

        # If needed: apply_calibration_autocorrect(values, motor_names)
        # or handle data_name logic.

        return values

    def read(self, data_name, motor_names=None):
        """
        Mirroring Dynamixel's read() method, but substituting a local dictionary
        in place of GroupSyncRead. If motor_names is None, we return all known motors.
        data_name might be "Present_Position" or "Present_Velocity" in hardware code.

        We'll do a basic version for present positions only.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"RosMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        if motor_names is None:
            motor_names = list(self.motors.keys())
        elif isinstance(motor_names, str):
            motor_names = [motor_names]

        values = []
        for name in motor_names:
            val = self.joint_positions.get(name, 0.0)
            values.append(val)

        values = np.array(values)

        # If your policy or code expects calibration, apply it here if self.calibration is set.
        # e.g. values = self.apply_calibration_autocorrect(values, motor_names)

        # Log read time
        delta_ts_name = f"delta_timestamp_s_read_{data_name}"
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # Log UTC if you want
        # ts_utc_name = ...
        # self.logs[ts_utc_name] = capture_timestamp_utc()

        return values

    def write_with_motor_ids(self, motor_models, motor_ids, data_name, values, num_retry=NUM_WRITE_RETRY):
        """
        Replicates the group write pattern. We'll compose a JointState message
        with the given motor IDs. This is a contrived approach in ROS, but 
        it keeps the code parallel to Dynamixel.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"RosMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        if not isinstance(motor_ids, list):
            motor_ids = [motor_ids]
        if not isinstance(values, list):
            values = [values]

        # In hardware code, we do a GroupSyncWrite. For ROS, we just publish once.
        # Let's build a JointState message that sets each motor to the specified value.
        js = JointState()
        js.header = Header()
        js.header.stamp = rospy.Time.now()

        pub_names = []
        pub_positions = []
        for idx, val in zip(motor_ids, values):
            # find the name in self.motors
            name = None
            for k, (m_id, m_model) in self.motors.items():
                if m_id == idx:
                    name = k
                    break
            if name is None:
                continue
            pub_names.append(name)
            pub_positions.append(val)

        js.name = pub_names
        js.position = pub_positions

        self.cmd_pub.publish(js)

    def write(self, data_name, values, motor_names=None):
        """
        The main function to send commands to Isaac Sim. 
        In hardware code, we do GroupSyncWrite for 'Goal_Position' or similar.
        We'll replicate that logic using a single JointState publish.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"RosMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        if motor_names is None:
            motor_names = list(self.motors.keys())
        elif isinstance(motor_names, str):
            motor_names = [motor_names]

        if isinstance(values, (int, float, np.integer, np.floating)):
            values = [float(values)] * len(motor_names)

        values = np.array(values).tolist()

        # Possibly revert calibration here if needed
        # values = self.revert_calibration(values, motor_names)

        # Build the JointState
        js = JointState()
        js.header = Header()
        js.header.stamp = rospy.Time.now()
        js.name = motor_names
        js.position = values

        self.cmd_pub.publish(js)

        # Logging
        delta_ts_name = f"delta_timestamp_s_write_{data_name}"
        self.logs[delta_ts_name] = time.perf_counter() - start_time

    def set_calibration(self, calibration: dict[str, list]):
        """
        Allows user to inject a calibration dict if we want to interpret joint states
        consistently with the real hardware. 
        For simulation, we might leave this unneeded.
        """
        self.calibration = calibration

    def apply_calibration_autocorrect(self, values, motor_names):
        """
        Mirrors the approach in Dynamixel code where out-of-range angles
        are automatically corrected by shifting. In a perfect sim, you might skip this.
        """
        # For brevity, we skip the actual logic here. In real code, you'd replicate 
        # the shifting logic from the hardware code if your sim angles differ from training angles.
        return values

    def apply_calibration(self, values, motor_names):
        """
        Convert from raw sim data to degrees or linear values if needed.
        """
        # Stub
        return values

    def revert_calibration(self, values, motor_names):
        """
        Inverse of apply_calibration.
        """
        # Stub
        return np.round(values).astype(np.int32)

    def are_motors_configured(self):
        """
        In hardware code, we read motor IDs to see if they match expected. 
        For sim, you might skip or always return True.
        """
        return True

    def find_motor_indices(self, possible_ids=None, num_retry=2):
        """
        In hardware, we discover which motors are physically connected.
        In sim, there's no real "search." So we can skip or return a fixed list.
        """
        return list(range(len(self.motors)))

    def set_bus_baudrate(self, baudrate):
        """
        ROS doesn't use baud rate. No-op here.
        """
        pass

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
