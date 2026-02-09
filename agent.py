from protocol import FollowerArm, LeaderArm
import math

def yaw(degrees: float) -> float:
    degrees = max(0, degrees)
    degrees = min(180, degrees)
    degrees /= 360
    return 0.75 - degrees

def get_elbow(shoulder: float) -> float:
    pitch_angle = shoulder * 360 - 198
    vertical = math.sin(math.radians(pitch_angle))
    return 1 - shoulder + vertical

FOLLOWER_PORT = '/dev/tty.usbmodem5B3E1188421'
LEADER_PORT = '/dev/tty.usbmodem5B3E1228011'
follower = FollowerArm(FOLLOWER_PORT, update_rate=100, use_sync_write=True)
leader = LeaderArm(LEADER_PORT, update_rate=100)

try:
    while True:
        shoulder_position = leader.get_positions()["shoulder"]
        elbow_position = get_elbow(shoulder_position)
        _ = follower.set_positions({"shoulder":shoulder_position, "elbow":elbow_position})
finally:
    follower.close()
    leader.close()
