#!/usr/bin/env python3
"""
Simple test script for hand detector.
Shows landmarks in real-time with minimal overhead.
"""

import cv2
from hand_detector import HandDetector
import numpy as np


def main():
    """Minimal hand tracking example."""
    
    print("="*60)
    print("Hand Landmark Detector Test")
    print("="*60)
    print("\nControls:")
    print("  'q' - Quit")
    print("  'd' - Toggle landmark drawing")
    print("  'i' - Print landmark info")
    print("\nStarting webcam...")
    
    # Initialize detector with optimized settings
    detector = HandDetector(
        max_hands=1,  # Track one hand for better performance
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    draw_landmarks = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            landmarks, annotated = detector.process_frame(
                frame,
                return_image=draw_landmarks
            )
            
            # Use annotated image if drawing, otherwise original
            display_frame = annotated if draw_landmarks and annotated is not None else frame
            
            # Add FPS counter
            fps = detector.get_fps()
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Show hand count
            hand_count = len(landmarks) if landmarks else 0
            cv2.putText(
                display_frame,
                f"Hands: {hand_count}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Display
            cv2.imshow('Hand Detector', display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                draw_landmarks = not draw_landmarks
                print(f"Landmark drawing: {'ON' if draw_landmarks else 'OFF'}")
            elif key == ord('i'):
                if landmarks:
                    print("\n" + "="*60)
                    print("Current Landmarks:")
                    for hand_idx, hand_landmarks in enumerate(landmarks):
                        print(f"\nHand {hand_idx}:")
                        print(f"  Wrist: {hand_landmarks[HandDetector.WRIST]}")
                        print(f"  Thumb tip: {hand_landmarks[HandDetector.THUMB_TIP]}")
                        print(f"  Index tip: {hand_landmarks[HandDetector.INDEX_FINGER_TIP]}")
                        print(f"  Middle tip: {hand_landmarks[HandDetector.MIDDLE_FINGER_TIP]}")
                        print(f"  Ring tip: {hand_landmarks[HandDetector.RING_FINGER_TIP]}")
                        print(f"  Pinky tip: {hand_landmarks[HandDetector.PINKY_TIP]}")
                    print("="*60)
                else:
                    print("No hands detected")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        
        avg_time = detector.get_average_processing_time()
        print(f"\nStatistics:")
        print(f"  Average processing time: {avg_time:.2f} ms")
        print(f"  Theoretical max FPS: {1000/avg_time:.1f}")


if __name__ == "__main__":
    main()
