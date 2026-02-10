import math, time
from protocol import FollowerArm, LeaderArm
from typing import Tuple, Dict, Optional

L1 = 0.155
L2 = 0.155
L3 = 0.080
SHOULDER_HEIGHT = 0.05

YAW_CENTER = 0.5
YAW_RANGE = 0.25

SHOULDER_FORWARD = 1.0
SHOULDER_BACKWARD = 0.5

ELBOW_STRAIGHT = 0.0
ELBOW_FOLDED = 0.5

WRIST_UP = 0.0
WRIST_DOWN = 0.5

FOLLOWER_PORT = "/dev/tty.usbmodem5B3E1188421"
LEADER_PORT = "/dev/tty.usbmodem5B3E1228011"


def servo_to_angle(servo_pos: float, servo_min: float, servo_max: float) -> float:
    t = (servo_pos - servo_min) / (servo_max - servo_min)
    t = max(0.0, min(1.0, t))
    return t * 180.0


def angle_to_servo(angle_deg: float, servo_min: float, servo_max: float) -> float:
    angle_deg = max(0.0, min(180.0, angle_deg))
    t = angle_deg / 180.0
    servo_pos = servo_min + t * (servo_max - servo_min)
    return max(0.0, min(1.0, servo_pos))


def yaw_to_servo(yaw_deg: float) -> float:
    servo_pos = YAW_CENTER - (yaw_deg / 90.0) * YAW_RANGE
    return max(0.25, min(0.75, servo_pos))

def forward_kinematics(
    yaw_servo: float,
    shoulder_servo: float,
    elbow_servo: float,
    wrist_servo: float
) -> Tuple[float, float, float, float]:
    
    yaw_deg = (YAW_CENTER - yaw_servo) / YAW_RANGE * 90.0

    shoulder_deg = servo_to_angle(shoulder_servo, SHOULDER_FORWARD, SHOULDER_BACKWARD)
    elbow_deg = servo_to_angle(elbow_servo, ELBOW_STRAIGHT, ELBOW_FOLDED)
    wrist_deg = servo_to_angle(wrist_servo, WRIST_UP, WRIST_DOWN)

    shoulder_rad = math.radians(shoulder_deg)
    elbow_rad = math.radians(elbow_deg)

    forearm_angle_rad = shoulder_rad - elbow_rad
    forearm_angle_deg = math.degrees(forearm_angle_rad)

    wrist_relative_deg = 90.0 - wrist_deg
    wrist_pitch_deg = wrist_relative_deg + forearm_angle_deg

    elbow_x = L1 * math.cos(shoulder_rad)
    elbow_y = L1 * math.sin(shoulder_rad)

    wrist_x = elbow_x + L2 * math.cos(forearm_angle_rad)
    wrist_y = elbow_y + L2 * math.sin(forearm_angle_rad)

    wrist_y += SHOULDER_HEIGHT

    wrist_pitch_rad = math.radians(wrist_pitch_deg)

    claw_x_plane = wrist_x + L3 * math.cos(wrist_pitch_rad)
    claw_y = wrist_y + L3 * math.sin(wrist_pitch_rad)

    yaw_rad = math.radians(yaw_deg)

    x = claw_x_plane * math.sin(yaw_rad)
    z = claw_x_plane * math.cos(yaw_rad)
    y = claw_y

    return x, y, z, wrist_pitch_deg

def inverse_kinematics(
    x: float,
    y: float,
    z: float,
    wrist_pitch_deg: float = 0.0
) -> Tuple[float, float, float, float]:
    
    yaw_deg = math.degrees(math.atan2(x, z))
    horizontal_dist = math.sqrt(x*x + z*z)
    
    wrist_pitch_rad = math.radians(wrist_pitch_deg)
    wrist_target_x_abs = horizontal_dist - L3 * math.cos(wrist_pitch_rad)
    wrist_target_y_abs = y - L3 * math.sin(wrist_pitch_rad)
    
    wrist_target_x = wrist_target_x_abs
    wrist_target_y = wrist_target_y_abs - SHOULDER_HEIGHT
    
    r = math.sqrt(wrist_target_x**2 + wrist_target_y**2)
    
    max_reach = L1 + L2
    min_reach = abs(L1 - L2)
    
    if r > max_reach:
        r = max_reach - 0.001
        scale = r / math.sqrt(wrist_target_x**2 + wrist_target_y**2)
        wrist_target_x *= scale
        wrist_target_y *= scale
    elif r < min_reach:
        r = min_reach + 0.001
        scale = r / max(0.001, math.sqrt(wrist_target_x**2 + wrist_target_y**2))
        wrist_target_x *= scale
        wrist_target_y *= scale
    
    cos_elbow_internal = (L1**2 + L2**2 - r**2) / (2 * L1 * L2)
    cos_elbow_internal = max(-1.0, min(1.0, cos_elbow_internal))
    elbow_internal_rad = math.acos(cos_elbow_internal)
    elbow_internal_deg = math.degrees(elbow_internal_rad)
    elbow_deg = 180.0 - elbow_internal_deg
    
    alpha = math.atan2(wrist_target_y, wrist_target_x)
    
    cos_beta = (L1**2 + r**2 - L2**2) / (2 * L1 * r)
    cos_beta = max(-1.0, min(1.0, cos_beta))
    beta = math.acos(cos_beta)
    
    shoulder_rad = alpha + beta
    shoulder_deg = math.degrees(shoulder_rad)
    shoulder_deg = max(0.0, min(180.0, shoulder_deg))
    
    forearm_angle_rad = shoulder_rad - math.radians(elbow_deg)
    forearm_angle_deg = math.degrees(forearm_angle_rad)
    
    desired_wrist_global_deg = wrist_pitch_deg
    wrist_relative_deg = desired_wrist_global_deg - forearm_angle_deg
    wrist_deg = 90.0 - wrist_relative_deg
    wrist_deg = max(0.0, min(180.0, wrist_deg))
    
    yaw_servo = yaw_to_servo(yaw_deg)
    shoulder_servo = angle_to_servo(shoulder_deg, SHOULDER_FORWARD, SHOULDER_BACKWARD)
    elbow_servo = angle_to_servo(elbow_deg, ELBOW_STRAIGHT, ELBOW_FOLDED)
    wrist_servo = angle_to_servo(wrist_deg, WRIST_UP, WRIST_DOWN)
    
    return yaw_servo, shoulder_servo, elbow_servo, wrist_servo


class ArmController:
    
    def __init__(self, follower_port: str = FOLLOWER_PORT, update_rate: int = 100):
        self.follower: FollowerArm = FollowerArm(follower_port, update_rate=update_rate, use_sync_write=True)

    def format_arm_state_for_ai(self) -> str:
        state = self.get_position_cm()
        return (
            "Arm state (all units in cm or degrees):\n"
            f"X: {state['x'] * 100:.2f}  -> horizontal, +right\n"
            f"Y: {state['y'] * 100:.2f}  -> vertical, +up (ground ≈ -3)\n"
            f"Z: {state['z'] * 100:.2f}  -> forward, +forward\n"
            f"WristPitch: {state['wrist_pitch_deg']:.1f}° -> claw tilt, 0=horizontal, +up\n"
            f"Roll: {state['roll']:.2f} -> claw roll, 0.5=center\n"
            f"Claw: {state['claw']:.2f} -> gripper, 0=closed, 1=open"
        )
        
    def set_position(
        self,
        x: float = 0.0,
        y: float = 0.15,
        z: float = 0.20,
        wrist_pitch_deg: float = 0.0,
        roll: float = 0.5,
        claw: float = 0.5
    ) -> Dict[str, float]:
        
        yaw, shoulder, elbow, wrist = inverse_kinematics(x, y, z, wrist_pitch_deg)
        
        positions = {
            "yaw": yaw,
            "shoulder": shoulder,
            "elbow": elbow,
            "wrist": wrist,
            "roll": roll,
            "claw": claw,
        }
        
        _ = self.follower.set_positions(positions)
        return positions
    
    def set_position_cm(
        self,
        x_cm: float = 0.0,
        y_cm: float = 15.0,
        z_cm: float = 20.0,
        wrist_pitch_deg: float = 0.0,
        roll: float = 0.5,
        claw: float = 0.5
    ) -> Dict[str, float]:
        
        return self.set_position(
            x_cm / 100.0,
            y_cm / 100.0,
            z_cm / 100.0,
            wrist_pitch_deg,
            roll,
            claw
        )
    
    def close(self):
        self.follower.close()


    def get_position_cm(self) -> Dict[str, float]:
        pos = self.follower.get_positions()

        x, y, z, wrist_pitch = forward_kinematics(
            pos["yaw"],
            pos["shoulder"],
            pos["elbow"],
            pos["wrist"]
        )

        return {
            "x": x * 100,
            "y": y * 100,
            "z": z * 100,
            "wrist_pitch_deg": wrist_pitch,
            "roll": pos["roll"],
            "claw": pos["claw"],
        }
