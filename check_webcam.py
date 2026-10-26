"""
Diagnostic script to test webcam connectivity and grab a verification frame.
"""

from __future__ import annotations

import sys
import cv2

from config import CAMERA_SOURCE


def test_webcam():
    print(f"Attempting to connect to camera source: {CAMERA_SOURCE} ...")
    cap = cv2.VideoCapture(CAMERA_SOURCE)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera source {CAMERA_SOURCE}.")
        print("Note: If running in a headless VM, Docker, or if webcam is disabled, set CAMERA_SOURCE to a video file or RTSP stream in config.py or .env.")
        return False

    ret, frame = cap.read()
    if not ret or frame is None:
        print("[WARNING] Camera opened, but failed to grab a frame.")
        cap.release()
        return False

    h, w, c = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[SUCCESS] Camera is working! Resolution: {w}x{h}, Channels: {c}, Reported FPS: {fps}")

    # Save a verification test frame
    out_file = "camera_test_snapshot.jpg"
    cv2.imwrite(out_file, frame)
    print(f"[SUCCESS] Saved test frame to '{out_file}'.")

    cap.release()
    return True


if __name__ == "__main__":
    success = test_webcam()
    sys.exit(0 if success else 1)
