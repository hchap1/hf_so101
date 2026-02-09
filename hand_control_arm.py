#!/usr/bin/env python3
"""
Hand-controlled robot arm teleoperation (6-DOF).

Controls:
    - Open palm: Enable arm control
    - Fist: Disable control (arm goes limp)
    - Hand position: Controls arm position
    - Pinch gesture: Controls gripper/claw
    - Wrist bending: Controls robot wrist pitch
    - Hand twist: Controls wrist roll
"""

import cv2
import numpy as np
from hand_detector import HandDetector, get_pinch_distance, is_fist, is_open_palm
from protocol import FollowerArm
import time


class HandToArmMapper:
    """Maps hand landmarks to robot arm joint positions (6-DOF) with smoothing."""

    def __init__(self, smoothing=0.2):
        # Workspace boundaries
        self.hand_x_range = (0.2, 0.8)
        self.hand_y_range = (0.0, 0.6)
        self.hand_z_range = (0, 0.1)

        # Joint ranges (normalized 0-1)
        self.joint_ranges = {
            'yaw': (0.3, 0.7),
            'shoulder': (0.5, 1.0),
            'elbow': (0.0, 0.5),
            'wrist': (0.5, 0.6),
            'roll': (0.2, 0.8),
            'claw': (0.4, 0.75),
        }

        # Previous joint positions for smoothing
        self.prev_joints = {k: 0.5 for k in self.joint_ranges.keys()}
        self.smoothing = smoothing

    def map_hand_to_arm(self, hand_landmarks: dict, pinch_distance: float) -> dict:
        # Key landmarks
        wrist = np.array(hand_landmarks[HandDetector.WRIST])
        index_mcp = np.array(hand_landmarks[HandDetector.INDEX_FINGER_MCP])
        middle_mcp = np.array(hand_landmarks[HandDetector.MIDDLE_FINGER_MCP])
        pinky_mcp = np.array(hand_landmarks[HandDetector.PINKY_MCP])
        index_tip = np.array(hand_landmarks[HandDetector.INDEX_FINGER_TIP])
        middle_tip = np.array(hand_landmarks[HandDetector.MIDDLE_FINGER_TIP])

        # --- Stable hand base for shoulder/elbow ---
        hand_base = (index_mcp + middle_mcp + pinky_mcp) / 3.0

        # --- Base rotation (yaw) ---
        yaw = self._map_range(
            hand_base[0],
            self.hand_x_range[0], self.hand_x_range[1],
            self.joint_ranges['yaw'][0], self.joint_ranges['yaw'][1]
        )

        # --- Shoulder ---
        shoulder = self._map_range(
            hand_base[1],
            self.hand_y_range[0], self.hand_y_range[1],
            self.joint_ranges['shoulder'][0], self.joint_ranges['shoulder'][1]
        )

        # --- Elbow extension ---
        forearm_vec = hand_base - wrist
        elbow_distance = np.linalg.norm(forearm_vec)
        elbow = self._map_range(
            elbow_distance,
            0.15, 0.25,  # tune for your robot
            self.joint_ranges['elbow'][0], self.joint_ranges['elbow'][1]
        )

        # --- Wrist pitch ---
        hand_dir = middle_mcp - wrist
        wrist_pitch_angle = np.arctan2(hand_dir[1], hand_dir[2])
        wrist = 0.5 + (wrist_pitch_angle / np.pi)
        wrist = np.clip(wrist, self.joint_ranges['wrist'][0], self.joint_ranges['wrist'][1])

        # --- Wrist roll ---
        roll_vec = pinky_mcp - index_mcp
        roll = (np.arctan2(roll_vec[2], roll_vec[0]) + np.pi) / (2 * np.pi)
        roll = np.clip(roll + 0.2, self.joint_ranges['roll'][0], self.joint_ranges['roll'][1])

        # --- Claw ---
        claw = self._map_range(
            pinch_distance,
            0.05, 0.2,
            self.joint_ranges['claw'][0], self.joint_ranges['claw'][1]
        )

        # --- Apply smoothing ---
        joints = {
            'yaw': yaw,
            'shoulder': shoulder,
            'elbow': elbow,
            'wrist': wrist,
            'roll': roll,
            'claw': claw
        }
        for k in joints:
            joints[k] = self.prev_joints[k] + self.smoothing * (joints[k] - self.prev_joints[k])
            self.prev_joints[k] = joints[k]

        return joints

    def _map_range(self, value, in_min, in_max, out_min, out_max):
        value = np.clip(value, in_min, in_max)
        normalized = (value - in_min) / (in_max - in_min)
        output = out_min + normalized * (out_max - out_min)
        return np.clip(output, out_min, out_max)


def main():
    print("="*70)
    print("HAND-CONTROLLED ROBOT ARM (6-DOF)")
    print("="*70)
    print("\nGestures:")
    print("  Open palm - Enable arm control")
    print("  Fist - Disable control")
    print("  Pinch - Control gripper")
    print("\nHand position controls:")
    print("  Left/Right - Base rotation")
    print("  Up/Down - Shoulder")
    print("  Forward/Back - Elbow extension")
    print("  Bend wrist up/down - Wrist pitch")
    print("  Twist hand - Wrist roll")
    print("\nPress 'q' to quit")
    print("="*70)

    detector = HandDetector(max_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mapper = HandToArmMapper(smoothing=0.25)
    FOLLOWER_PORT = '/dev/tty.usbmodem5B3E1188421'
    follower = None
    arm_enabled = False

    try:
        print("\nInitializing follower arm...")
        follower = FollowerArm(FOLLOWER_PORT, update_rate=50, use_sync_write=True, enable_monitoring=False)
        print("✓ Follower arm ready")
        follower.bus.disable_all_torque([1,2,3,4,5,6])
        print("✓ Arm in free mode (make fist to disable control)")

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("✓ Webcam ready\nShow your hand to begin!\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks, annotated = detector.process_frame(frame, return_image=True)

            if landmarks and len(landmarks) > 0:
                hand_landmarks = landmarks[0]
                is_palm_gesture = is_open_palm(hand_landmarks)
                is_fist_gesture = is_fist(hand_landmarks)
                pinch_dist = get_pinch_distance(hand_landmarks)

                # Enable/disable arm control
                if is_palm_gesture and not arm_enabled:
                    print("✓ Arm control ENABLED")
                    follower.bus.enable_all_torque_safely([1,2,3,4,5,6])
                    arm_enabled = True
                elif is_fist_gesture and arm_enabled:
                    print("○ Arm control DISABLED")
                    follower.bus.disable_all_torque([1,2,3,4,5,6])
                    arm_enabled = False

                if arm_enabled:
                    arm_positions = mapper.map_hand_to_arm(hand_landmarks, pinch_dist)
                    follower.set_positions(arm_positions)

                status_text = "CONTROLLING" if arm_enabled else "DISABLED"
                status_color = (0,255,0) if arm_enabled else (0,0,255)
                cv2.putText(annotated, f"Arm: {status_text}", (10,70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                cv2.putText(annotated, f"Pinch: {pinch_dist:.3f}", (10,110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
            else:
                if arm_enabled:
                    print("○ Lost hand tracking - disabling arm")
                    follower.bus.disable_all_torque([1,2,3,4,5,6])
                    arm_enabled = False
                cv2.putText(annotated, "No hand detected", (10,70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            fps = detector.get_fps()
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            cv2.imshow("Hand Control", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback; traceback.print_exc()
    finally:
        if follower:
            print("\nDisabling arm and closing...")
            follower.close()
        cv2.destroyAllWindows()
        detector.close()
        print("Done!")


if __name__ == "__main__":
    main()
