"""
Detection and recognition engine combining Ultralytics YOLOv8 for human detection
and face_recognition (dlib 128-d embeddings) for authorized vs intruder classification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO

from config import (
    FACE_CROP_MARGIN,
    FACE_DETECTION_MODEL,
    FACE_MATCH_TOLERANCE,
    KNOWN_FACES_DIR,
    MODELS_DIR,
    PERSON_CLASS_ID,
    RESTRICTED_ZONE_ENABLED,
    RESTRICTED_ZONE_POLYGON,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_MODEL_NAME,
    is_box_in_restricted_zone,
)

logger = logging.getLogger("IntruderDetection.Detector")


@dataclass
class DetectedPerson:
    """Encapsulates person detection and identity metadata."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    identity: str                   # Name of known person or 'UNKNOWN'
    is_authorized: bool             # True if matches known face
    face_distance: float = 1.0      # Distance to closest known face
    in_restricted_zone: bool = True  # True if inside zone or zone disabled


class IntruderDetector:
    """
    Unified detector handling YOLOv8 human detection and face identification.
    Caches known face encodings from disk to ensure real-time inference speed.
    """

    def __init__(
        self,
        model_name: str = YOLO_MODEL_NAME,
        conf_thresh: float = YOLO_CONFIDENCE_THRESHOLD,
        tolerance: float = FACE_MATCH_TOLERANCE,
    ):
        self.conf_thresh = conf_thresh
        self.tolerance = tolerance

        logger.info("Loading YOLOv8 model '%s'...", model_name)
        # Store model in models/ directory if possible
        model_target = MODELS_DIR / model_name if MODELS_DIR.exists() else model_name
        self.yolo = YOLO(str(model_target))
        logger.info("YOLOv8 initialized successfully.")

        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.reload_known_faces()

    def reload_known_faces(self) -> int:
        """
        Scan `known_faces/` subdirectories and load 128-d face encodings.
        Directory structure:
            known_faces/
                Alice/
                    photo1.jpg
                    photo2.jpg
                Bob/
                    1.jpg
        :return: Total number of encodings loaded
        """
        encodings: List[np.ndarray] = []
        names: List[str] = []

        if not KNOWN_FACES_DIR.exists():
            KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("Created empty known_faces directory at %s", KNOWN_FACES_DIR)

        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for person_dir in KNOWN_FACES_DIR.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                person_count = 0
                for img_file in person_dir.iterdir():
                    if img_file.suffix.lower() in supported_exts:
                        try:
                            # Load via face_recognition (RGB format)
                            image = face_recognition.load_image_file(str(img_file))
                            face_locs = face_recognition.face_locations(
                                image, model=FACE_DETECTION_MODEL
                            )
                            if not face_locs:
                                # Fallback: try whole image if pre-cropped
                                face_locs = [(0, image.shape[1], image.shape[0], 0)]

                            encs = face_recognition.face_encodings(
                                image, known_face_locations=face_locs
                            )
                            for enc in encs:
                                encodings.append(enc)
                                names.append(person_name)
                                person_count += 1
                        except Exception as exc:
                            logger.warning(
                                "Failed to process face sample %s: %s", img_file, exc
                            )

                logger.info("Loaded %d face templates for '%s'", person_count, person_name)

        self.known_face_encodings = encodings
        self.known_face_names = names
        logger.info("Total loaded face encodings: %d across %d identities.",
                    len(encodings), len(set(names)))
        return len(encodings)

    def _recognize_face_in_crop(self, person_crop_bgr: np.ndarray) -> Tuple[str, bool, float]:
        """
        Given a cropped BGR image of a detected person, detect face locations
        and compare against known face encodings.

        :return: (name, is_authorized, min_distance)
        """
        if person_crop_bgr.size == 0 or len(self.known_face_encodings) == 0:
            return "UNKNOWN", False, 1.0

        # Convert to RGB as required by dlib/face_recognition
        rgb_crop = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)

        # Detect face within the person crop (much smaller area -> fast)
        face_locations = face_recognition.face_locations(
            rgb_crop, model=FACE_DETECTION_MODEL
        )

        if not face_locations:
            # If person is facing backwards or face is occluded, treat as UNKNOWN for safety
            return "UNKNOWN", False, 1.0

        encodings = face_recognition.face_encodings(rgb_crop, face_locations)
        if not encodings:
            return "UNKNOWN", False, 1.0

        # Test against the primary detected face
        face_encoding = encodings[0]
        distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

        if len(distances) == 0:
            return "UNKNOWN", False, 1.0

        best_match_idx = int(np.argmin(distances))
        best_distance = float(distances[best_match_idx])

        if best_distance <= self.tolerance:
            matched_name = self.known_face_names[best_match_idx]
            return matched_name, True, best_distance
        else:
            return "UNKNOWN", False, best_distance

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        run_face_recognition: bool = True,
        prev_detections: Optional[List[DetectedPerson]] = None,
    ) -> List[DetectedPerson]:
        """
        Process a single video frame:
        1. Run YOLOv8 to locate all humans (class 0).
        2. Filter by confidence threshold.
        3. Check restricted zone boundaries.
        4. If `run_face_recognition` is True, crop person and recognize face.
           Otherwise, carry forward previous identities or default to UNKNOWN.

        :param frame_bgr: BGR video frame from OpenCV
        :param run_face_recognition: Whether to execute face embeddings on this frame
        :param prev_detections: Detections from last processed frame for tracking
        :return: List of DetectedPerson objects
        """
        height, width = frame_bgr.shape[:2]
        results = self.yolo.predict(
            source=frame_bgr,
            conf=self.conf_thresh,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )

        detected_people: List[DetectedPerson] = []
        boxes_data = results[0].boxes

        if boxes_data is None or len(boxes_data) == 0:
            return detected_people

        for box in boxes_data:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = (
                max(0, int(xyxy[0])),
                max(0, int(xyxy[1])),
                min(width, int(xyxy[2])),
                min(height, int(xyxy[3])),
            )
            conf = float(box.conf[0].cpu().numpy())

            in_zone = is_box_in_restricted_zone((x1, y1, x2, y2))

            identity = "UNKNOWN"
            is_authorized = False
            face_dist = 1.0

            if run_face_recognition:
                # Add a slight margin if possible, then crop person
                crop_h = y2 - y1
                # Focus primarily on the upper body / head region for face recognition
                head_y2 = min(height, y1 + int(crop_h * 0.65))
                person_crop = frame_bgr[y1:head_y2, x1:x2]

                identity, is_authorized, face_dist = self._recognize_face_in_crop(person_crop)
            elif prev_detections:
                # Estimate overlap with previous detections (IoU or proximity)
                for prev in prev_detections:
                    px1, py1, px2, py2 = prev.bbox
                    # Center distance check
                    curr_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    prev_center = ((px1 + px2) / 2, (py1 + py2) / 2)
                    dist = np.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])
                    if dist < 60:
                        identity = prev.identity
                        is_authorized = prev.is_authorized
                        face_dist = prev.face_distance
                        break

            person = DetectedPerson(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                identity=identity,
                is_authorized=is_authorized,
                face_distance=face_dist,
                in_restricted_zone=in_zone,
            )
            detected_people.append(person)

        return detected_people

    @staticmethod
    def annotate_frame(
        frame_bgr: np.ndarray,
        detections: List[DetectedPerson],
        draw_zone: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes and status labels on the frame:
        - Green bounding box and name tag for authorized personnel
        - Red bounding box and 'UNKNOWN / INTRUDER' tag for intruders
        - Restricted polygon boundary line overlay
        """
        annotated = frame_bgr.copy()

        # 1. Draw restricted zone polygon if enabled
        if draw_zone and RESTRICTED_ZONE_ENABLED and RESTRICTED_ZONE_POLYGON:
            pts = np.array(RESTRICTED_ZONE_POLYGON, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(
                annotated,
                "RESTRICTED ZONE",
                (RESTRICTED_ZONE_POLYGON[0][0] + 5, RESTRICTED_ZONE_POLYGON[0][1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

        # 2. Draw person bounding boxes
        for person in detections:
            x1, y1, x2, y2 = person.bbox
            if person.is_authorized:
                color = (0, 220, 0)  # Bright Green for authorized
                label = f"{person.identity} ({person.confidence*100:.0f}%)"
            else:
                color = (0, 0, 255)  # Bright Red for unknown/intruder
                zone_flag = " [ZONE]" if (RESTRICTED_ZONE_ENABLED and person.in_restricted_zone) else ""
                label = f"UNKNOWN INTRUDER{zone_flag} ({person.confidence*100:.0f}%)"

            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label banner background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            banner_y1 = max(0, y1 - text_h - 10)
            banner_y2 = y1
            cv2.rectangle(
                annotated,
                (x1, banner_y1),
                (x1 + text_w + 10, banner_y2),
                color,
                cv2.FILLED,
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )

        return annotated
