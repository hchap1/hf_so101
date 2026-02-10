import cv2
import numpy as np
from dataclasses import dataclass, asdict
from ultralytics import YOLO

# ==========================
# USER-TUNABLE PARAMETERS
# ==========================
CAMERA_INDEX = 0
CONF_THRESHOLD = 0.4

# Desk plane dimensions in cm (used to map clicked rectangle)
DESK_WIDTH = 40.0
DESK_HEIGHT = 30.0

# Hardcoded arm base in desk plane coordinates
ARM_BASE = np.array([0.0, 0.0])

# ==========================
# DATA STRUCTURES
# ==========================
@dataclass
class DetectedObject:
    name: str
    confidence: float
    image_xy: tuple
    world_xy: tuple
    relative_to_arm: tuple

# ==========================
# GLOBAL STATE
# ==========================
mouse_pos = (0, 0)
clicked_points = []
homography = None

# ==========================
# MOUSE CALLBACK
# ==========================
def mouse_callback(event, x, y, flags, param):
    global mouse_pos, clicked_points
    mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        print(f"Clicked point {len(clicked_points)}: {x}, {y}")

# ==========================
# VISION SYSTEM
# ==========================
class VisionWorldModel:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.model = YOLO("yolov8n.pt")  # make sure you have this

    def compute_homography(self):
        global homography
        if len(clicked_points) != 4:
            raise ValueError("Need exactly 4 clicked points to compute homography")

        image_pts = np.array(clicked_points, dtype=np.float32)
        world_pts = np.array([
            [-20, 0],
            [-20, 40],
            [20, 40],
            [20, 0]
        ], dtype=np.float32)

        homography, _ = cv2.findHomography(image_pts, world_pts)
        print("Homography computed and locked!")

    def pixel_to_world(self, px, py):
        pt = np.array([[[px, py]]], dtype=np.float32)
        world_pt = cv2.perspectiveTransform(pt, homography)[0][0]
        return world_pt

    def detect_objects(self, frame):
        results = self.model(frame, verbose=False)[0]

        objects = []
        for box in results.boxes:
            conf = float(box.conf)
            if conf < CONF_THRESHOLD:
                continue

            cls_id = int(box.cls)
            name = self.model.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0]
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)

            world_xy = self.pixel_to_world(cx, cy)
            rel_xy = world_xy - ARM_BASE

            objects.append(
                DetectedObject(
                    name=name,
                    confidence=conf,
                    image_xy=(cx, cy),
                    world_xy=(float(world_xy[0]), float(world_xy[1])),
                    relative_to_arm=(float(rel_xy[0]), float(rel_xy[1]))
                )
            )
        return objects

    def draw_debug(self, frame, objects):
        # Draw clicked points
        for idx, p in enumerate(clicked_points):
            cv2.circle(frame, p, 6, (0, 0, 255), -1)
            cv2.putText(frame, f"P{idx+1}", (p[0]+5, p[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # Draw detected objects
        for obj in objects:
            x, y = map(int, obj.image_xy)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            label = f"{obj.name} ({obj.world_xy[0]:.1f},{obj.world_xy[1]:.1f})"
            cv2.putText(frame, label, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw homography quadrilateral if computed
        if homography is not None:
            pts = np.array(clicked_points, dtype=int)
            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)

        # Draw mouse position
        if homography is not None:
            desk_pos = self.pixel_to_world(mouse_pos[0], mouse_pos[1])
            cv2.putText(frame, f"Mouse Desk: {desk_pos[0]:.1f},{desk_pos[1]:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)
        else:
            cv2.putText(frame, f"Mouse Pixel: {mouse_pos[0]},{mouse_pos[1]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

    def get_world_state(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera read failed")

        objects = []
        if homography is not None:
            objects = self.detect_objects(frame)

        self.draw_debug(frame, objects)
        cv2.imshow("Vision World Model", frame)
        cv2.waitKey(1)

        return {
            "arm_base": tuple(ARM_BASE),
            "objects": [asdict(o) for o in objects]
        }

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

import time

class Vision:

    def __init__(self):
        self.vision = VisionWorldModel()
        cv2.namedWindow("Vision World Model")
        cv2.setMouseCallback("Vision World Model", mouse_callback)
        print("Click the 4 desk corners in the camera feed window.")
        while len(clicked_points) < 4:
            self.vision.get_world_state()  # show video + mouse overlay
        self.vision.compute_homography()

    def compute(self, duration=1.0):
        """
        Runs detection for `duration` seconds and returns
        the frame with the most individual objects detected.
        """
        start_time = time.time()
        best_state = None
        max_objects = -1

        while time.time() - start_time < duration:
            world_state = self.vision.get_world_state()
            num_objects = len(world_state["objects"])
            if num_objects > max_objects:
                max_objects = num_objects
                best_state = world_state

        # Format output
        output = ""
        if best_state:
            for obj in best_state["objects"]:
                name = obj["name"]
                x, y = obj["world_xy"]
                output += f"{name} is at ({x:.1f}, {y:.1f})\n"

        return output
