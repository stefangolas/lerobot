import numpy as np
from lerobot.common.robot_devices.motors.ros_dynamixel import RosDynamixelMotorsBus
from lerobot.common.robot_devices.motors.dynamixel import (
    CalibrationMode,
    TorqueMode,
    convert_degrees_to_steps,
)
from lerobot.common.robot_devices.motors.utils import MotorsBus

URL_TEMPLATE = (
    "https://raw.githubusercontent.com/huggingface/lerobot/main/media/{robot}/{arm}_{position}.webp"
)

ZERO_POSITION_DEGREE = 0
ROTATED_POSITION_DEGREE = 90


def assert_drive_mode(drive_mode):
    if not np.all(np.isin(drive_mode, [0, 1])):
        raise ValueError(f"`drive_mode` contains values other than 0 or 1: ({drive_mode})")


def apply_drive_mode(position, drive_mode):
    assert_drive_mode(drive_mode)
    signed_drive_mode = -(drive_mode * 2 - 1)
    return position * signed_drive_mode


def compute_nearest_rounded_position(position, models):
    delta_turn = convert_degrees_to_steps(ROTATED_POSITION_DEGREE, models)
    nearest_pos = np.round(position.astype(float) / delta_turn) * delta_turn
    return nearest_pos.astype(position.dtype)


def run_arm_calibration(arm: RosDynamixelMotorsBus, robot_type: str, arm_name: str, arm_type: str):
    if (arm.read("Torque_Enable") != TorqueMode.DISABLED.value).any():
        raise ValueError("To run calibration, the torque must be disabled on all motors.")

    print(f"\nRunning calibration of {robot_type} {arm_name} {arm_type}...")

    print("\nMove arm to zero position")
    print("See: " + URL_TEMPLATE.format(robot=robot_type, arm=arm_type, position="zero"))
    input("Press Enter to continue...")

    zero_target_pos = convert_degrees_to_steps(ZERO_POSITION_DEGREE, arm.motor_models)
    zero_pos = arm.read("Present_Position")
    zero_nearest_pos = compute_nearest_rounded_position(zero_pos, arm.motor_models)
    homing_offset = zero_target_pos - zero_nearest_pos

    print("\nMove arm to rotated target position")
    print("See: " + URL_TEMPLATE.format(robot=robot_type, arm=arm_type, position="rotated"))
    input("Press Enter to continue...")

    rotated_target_pos = convert_degrees_to_steps(ROTATED_POSITION_DEGREE, arm.motor_models)
    rotated_pos = arm.read("Present_Position")
    drive_mode = (rotated_pos < zero_pos).astype(np.int32)

    rotated_drived_pos = apply_drive_mode(rotated_pos, drive_mode)
    rotated_nearest_pos = compute_nearest_rounded_position(rotated_drived_pos, arm.motor_models)
    homing_offset = rotated_target_pos - rotated_nearest_pos

    print("\nMove arm to rest position")
    print("See: " + URL_TEMPLATE.format(robot=robot_type, arm=arm_type, position="rest"))
    input("Press Enter to continue...\n")

    calib_mode = [CalibrationMode.DEGREE.name] * len(arm.motor_names)

    if robot_type in ["widowx", "aloha"] and "gripper" in arm.motor_names:
        calib_idx = arm.motor_names.index("gripper")
        calib_mode[calib_idx] = CalibrationMode.LINEAR.name

    calib_data = {
        "homing_offset": homing_offset.tolist(),
        "drive_mode": drive_mode.tolist(),
        "start_pos": zero_pos.tolist(),
        "end_pos": rotated_pos.tolist(),
        "calib_mode": calib_mode,
        "motor_names": arm.motor_names,
    }
    return calib_data
