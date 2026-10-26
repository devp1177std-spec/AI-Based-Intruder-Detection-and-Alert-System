"""
Enrollment tool to capture and register authorized personnel.
Opens the webcam and allows capturing 5 clear face photos by pressing SPACEBAR.
Saves the templates to `known_faces/<person_name>/`.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import cv2
import face_recognition

from config import (
    CAMERA_SOURCE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    KNOWN_FACES_DIR,
    ensure_directories,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("IntruderDetection.RegisterFace")


def sanitize_name(name: str) -> str:
    """Sanitize directory name to alphanumeric, underscores, and hyphens."""
    clean = re.sub(r"[^\w\s-]", "", name).strip()
    return re.sub(r"[-\s]+", "_", clean)


def register_authorized_person(person_name: str, camera_source=CAMERA_SOURCE, target_photos: int = 5) -> bool:
    """
    Launch interactive camera feed to capture authorized face samples.

    :param person_name: Full name of the authorized individual
    :param camera_source: Camera index or video stream URL
    :param target_photos: Number of snapshots to capture (default 5)
    :return: True if registration completed successfully
    """
    ensure_directories()
    clean_name = sanitize_name(person_name)
    if not clean_name:
        logger.error("Invalid person name provided.")
        return False

    person_dir = KNOWN_FACES_DIR / clean_name
    person_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Enrolling '%s'. Photos will be saved to: %s", clean_name, person_dir)

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        logger.error("Could not open camera source '%s'. Check webcam connection.", camera_source)
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    captured_count = 0
    window_name = f"Register Authorized Face: {clean_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    logger.info("Camera ready. Position your face in front of the camera.")
    logger.info("Controls: Press [SPACE] to capture photo | Press [Q] or [ESC] to exit.")

    try:
        while captured_count < target_photos:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Failed to grab video frame. Retrying...")
                cv2.waitKey(30)
                continue

            display_frame = frame.copy()
            h, w = display_frame.shape[:2]

            # Detect face in current frame for live feedback
            rgb_small = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb_small, model="hog")
            face_found = len(face_locs) > 0

            # Draw face bounding box if detected
            if face_found:
                top, right, bottom, left = [coord * 2 for coord in face_locs[0]]
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                status_text = "Face Detected - Ready to Capture"
                status_color = (0, 255, 0)
            else:
                status_text = "No Face Detected - Please center yourself"
                status_color = (0, 165, 255)

            # Draw info banners
            cv2.rectangle(display_frame, (0, 0), (w, 45), (30, 30, 30), cv2.FILLED)
            cv2.putText(
                display_frame,
                f"Enrolling: {clean_name} | Captured: {captured_count}/{target_photos}",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.rectangle(display_frame, (0, h - 45), (w, h), (30, 30, 30), cv2.FILLED)
            cv2.putText(
                display_frame,
                status_text,
                (15, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "[SPACE] = Capture | [Q] = Cancel",
                (w - 280, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                logger.info("Registration cancelled by user.")
                break
            elif key == 32:  # SPACEBAR
                if not face_found:
                    logger.warning("No face detected in frame. Please look at the camera.")
                    # Flash red warning overlay briefly
                    flash = display_frame.copy()
                    cv2.rectangle(flash, (0, 0), (w, h), (0, 0, 200), 6)
                    cv2.imshow(window_name, flash)
                    cv2.waitKey(100)
                    continue

                captured_count += 1
                img_filename = person_dir / f"photo_{captured_count}.jpg"
                cv2.imwrite(str(img_filename), frame)
                logger.info("Captured photo %d/%d -> %s", captured_count, target_photos, img_filename)

                # Flash green confirmation overlay
                flash = display_frame.copy()
                cv2.rectangle(flash, (0, 0), (w, h), (0, 255, 0), 6)
                cv2.putText(
                    flash,
                    f"SAVED PHOTO {captured_count}!",
                    (w // 2 - 140, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    3,
                )
                cv2.imshow(window_name, flash)
                cv2.waitKey(200)

        if captured_count >= target_photos:
            logger.info("Enrollment complete for '%s'! 5 templates stored in %s", clean_name, person_dir)
            return True
        else:
            logger.warning("Enrollment incomplete (%d/%d photos captured).", captured_count, target_photos)
            return False

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Register an authorized person for intruder detection.")
    parser.add_argument(
        "--name",
        type=str,
        help="Name of the person to register (e.g. 'Alice', 'Dr_Smith')",
        default=None,
    )
    parser.add_argument(
        "--photos",
        type=int,
        default=5,
        help="Number of photos to capture (default 5)",
    )
    args = parser.parse_args()

    name = args.name
    if not name:
        try:
            name = input("Enter the name of the authorized person to register: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not name:
        print("Error: Name cannot be blank.")
        sys.exit(1)

    success = register_authorized_person(person_name=name, target_photos=args.photos)
    if success:
        print(f"\n[SUCCESS] Registered '{name}' successfully.")
    else:
        print(f"\n[FAILED] Registration for '{name}' was not completed.")


if __name__ == "__main__":
    main()
