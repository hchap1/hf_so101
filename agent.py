from protocol import FollowerArm, LeaderArm
from vector import Vector
import math

def find_angle(translation: float, extension: float):
    degrees = max(-45, min(45, math.degrees(math.atan2(translation, extension))))
    print(degrees)
    return yaw(degrees + 90)
    

def yaw(degrees: float) -> float:
    degrees = max(0, degrees)
    degrees = min(180, degrees)
    degrees /= 360
    return 0.75 - degrees

def get_elbow(shoulder: float) -> float:
    pitch_angle = shoulder * 360 - 198
    shoulder_vector = Vector.polar(10, pitch_angle)
    elbow_horizontal = math.sqrt(10**2 - shoulder_vector.y**2)
    elbow_vector = Vector.cartesian(elbow_horizontal, shoulder_vector.y)
    return 1 - shoulder + (elbow_vector.direction() / 360)

def get_wrist(shoulder: float) -> float:
    pitch_angle = shoulder * 360 - 198
    shoulder_vector = Vector.polar(10, pitch_angle)
    elbow_horizontal = math.sqrt(10**2 - shoulder_vector.y**2)
    elbow_vector = Vector.cartesian(elbow_horizontal, shoulder_vector.y)
    wrist_direction = -0.00272667 * elbow_vector.direction() + 0.2954
    return wrist_direction + 0.1

def get_extension(shoulder: float) -> float:
    pitch_angle = shoulder * 360 - 198
    shoulder_vector = Vector.polar(10, pitch_angle)
    elbow_horizontal = math.sqrt(10**2 - shoulder_vector.y**2)
    elbow_vector = Vector.cartesian(elbow_horizontal, shoulder_vector.y)
    extension = shoulder_vector.sub(elbow_vector).magnitude()
    return extension + 15

FOLLOWER_PORT = '/dev/tty.usbmodem5B3E1188421'
LEADER_PORT = '/dev/tty.usbmodem5B3E1228011'
follower = FollowerArm(FOLLOWER_PORT, update_rate=100, use_sync_write=True)
leader = LeaderArm(LEADER_PORT, update_rate=100)

try:
    while True:
        shoulder_position = leader.get_positions()["shoulder"]
        elbow_position = get_elbow(shoulder_position)
        wrist_position = get_wrist(shoulder_position)
        claw = leader.get_positions()["claw"]
        yaw_position = find_angle(10, get_extension(shoulder_position))
        _ = follower.set_positions(
            {
                "shoulder":shoulder_position,
                 "elbow":elbow_position,
                 "wrist":wrist_position,
                 "claw":claw,
                 "roll":0.75,
                 "yaw":yaw_position
             }
        )
finally:
    follower.close()
    leader.close()
