"""
Main detection application for the AI-Based Intruder Detection and Alert System.
Captures live camera stream, runs YOLOv8 person detection, performs face recognition
verification, tracks consecutive unknown presence, saves evidence snapshots,
logs events to SQLite, and triggers Telegram alerts.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from alerts import dispatch_intruder_alert, get_remaining_cooldown, is_on_cooldown
from config import (
    ALERT_COOLDOWN_SECONDS,
    BASE_DIR,
    CAMERA_RECONNECT_DELAY,
    CAMERA_SOURCE,
    CLEANUP_ON_STARTUP,
    CONSECUTIVE_FRAMES_THRESHOLD,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_CONSECUTIVE_FRAME_FAILS,
    PROCESS_EVERY_N_FRAMES,
    RESTRICTED_ZONE_ENABLED,
    RETENTION_DAYS,
    TARGET_FPS,
    TELEGRAM_ENABLED,
    ensure_directories,
    get_current_date_intruder_dir,
    is_within_active_hours,
)
from database import cleanup_old_events, init_db, log_intruder_event
from detector import DetectedPerson, IntruderDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("IntruderDetection.App")


class IntruderDetectionSystem:
    """Orchestrates video capture, person/face analysis, intruder tracking, and alerting."""

    def __init__(self, camera_source=CAMERA_SOURCE):
        self.camera_source = camera_source
        self.detector = IntruderDetector()

        # State tracking
        self.consecutive_unknown_count: int = 0
        self.frame_index: int = 0
        self.last_detections: List[DetectedPerson] = []
        self.is_running: bool = False

        # Performance monitoring
        self.fps: float = 0.0
        self.last_fps_time: float = time.time()
        self.fps_frame_count: int = 0

    def start(self) -> None:
        """Run the main detection loop with automatic camera reconnection."""
        ensure_directories()
        init_db()

        if CLEANUP_ON_STARTUP:
            try:
                cleanup_old_events(RETENTION_DAYS)
            except Exception as exc:
                logger.error("Error during startup cleanup: %s", exc)

        logger.info("Starting AI-Based Intruder Detection System...")
        logger.info("Camera Source: %s | Target FPS: %d", self.camera_source, TARGET_FPS)
        logger.info("Telegram Alerts Enabled: %s", TELEGRAM_ENABLED)

        self.is_running = True
        window_name = "AI-Based Intruder Detection and Alert System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, FRAME_WIDTH, FRAME_HEIGHT)

        frame_interval = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 0.033

        while self.is_running:
            cap = cv2.VideoCapture(self.camera_source)
            if not cap.isOpened():
                logger.warning(
                    "Camera '%s' is unavailable. Retrying in %.1f seconds...",
                    self.camera_source,
                    CAMERA_RECONNECT_DELAY,
                )
                time.sleep(CAMERA_RECONNECT_DELAY)
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            logger.info("Connected to camera stream.")

            consecutive_fails = 0

            try:
                while self.is_running and cap.isOpened():
                    start_loop_time = time.time()

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        consecutive_fails += 1
                        logger.warning(
                            "Failed to retrieve frame (%d/%d)...",
                            consecutive_fails,
                            MAX_CONSECUTIVE_FRAME_FAILS,
                        )
                        if consecutive_fails >= MAX_CONSECUTIVE_FRAME_FAILS:
                            logger.error("Camera connection lost. Reconnecting...")
                            break
                        time.sleep(0.1)
                        continue

                    consecutive_fails = 0
                    self.frame_index += 1

                    # Resize frame if needed for processing consistency
                    if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                    # 1. Decide whether to run heavy face recognition
                    run_face_rec = (self.frame_index % PROCESS_EVERY_N_FRAMES == 0)

                    # 2. Run detection pipeline
                    detections = self.detector.process_frame(
                        frame_bgr=frame,
                        run_face_recognition=run_face_rec,
                        prev_detections=self.last_detections,
                    )
                    self.last_detections = detections

                    # 3. Analyze detections for intruders
                    self._evaluate_intruders(frame, detections)

                    # 4. Generate annotated visualization
                    annotated_frame = self.detector.annotate_frame(frame, detections)
                    self._draw_hud(annotated_frame)

                    cv2.imshow(window_name, annotated_frame)

                    # 5. Handle keyboard controls
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        logger.info("Shutdown signal received via keyboard.")
                        self.is_running = False
                        break
                    elif key in (ord("r"), ord("R")):
                        logger.info("Reloading known faces requested...")
                        count = self.detector.reload_known_faces()
                        logger.info("Reloaded %d face templates.", count)

                    # 6. Paced FPS control
                    self._update_fps()
                    elapsed = time.time() - start_loop_time
                    delay = max(0.001, frame_interval - elapsed)
                    time.sleep(delay)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Shutting down...")
                self.is_running = False
            finally:
                cap.release()

        cv2.destroyAllWindows()
        logger.info("Intruder Detection System terminated cleanly.")

    def _evaluate_intruders(self, raw_frame: np.ndarray, detections: List[DetectedPerson]) -> None:
        """
        Evaluate current frame detections against intruder criteria:
        - Identify any person categorized as UNKNOWN.
        - If restricted zone is active, verify unknown person is within the zone.
        - Maintain consecutive frames count.
        - Trigger event snapshot, logging, and alert upon threshold.
        """
        # Find unknown persons that qualify (must be inside restricted zone if zone enabled)
        qualifying_intruders = [
            p for p in detections
            if not p.is_authorized and p.in_restricted_zone
        ]

        if qualifying_intruders:
            self.consecutive_unknown_count += 1
            logger.debug(
                "Unknown person observed. Consecutive frames: %d/%d",
                self.consecutive_unknown_count,
                CONSECUTIVE_FRAMES_THRESHOLD,
            )

            # Trigger event when threshold is reached
            if self.consecutive_unknown_count == CONSECUTIVE_FRAMES_THRESHOLD:
                self._trigger_intruder_event(raw_frame, detections, qualifying_intruders)
        else:
            # Graceful decay or immediate reset
            if self.consecutive_unknown_count > 0:
                self.consecutive_unknown_count = max(0, self.consecutive_unknown_count - 1)

    def _trigger_intruder_event(
        self,
        raw_frame: np.ndarray,
        all_detections: List[DetectedPerson],
        intruders: List[DetectedPerson],
    ) -> None:
        """Execute full event sequence: annotate image, save locally, insert DB, dispatch alert."""
        timestamp_now = datetime.now()
        ts_str = timestamp_now.strftime("%Y-%m-%d %H:%M:%S")
        file_ts = timestamp_now.strftime("%Y%m%d_%H%M%S")

        # 1. Annotate snapshot with bounding boxes, labels, and timestamp watermark
        annotated_snapshot = self.detector.annotate_frame(raw_frame, all_detections)
        # Add high-visibility timestamp watermark
        h, w = annotated_snapshot.shape[:2]
        watermark = f"EVIDENCE CAPTURED: {ts_str}"
        cv2.rectangle(annotated_snapshot, (0, h - 30), (w, h), (0, 0, 180), cv2.FILLED)
        cv2.putText(
            annotated_snapshot,
            watermark,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 2. Save image locally to intruders/YYYY-MM-DD/intruder_YYYYMMDD_HHMMSS.jpg
        today_dir = get_current_date_intruder_dir()
        filename = f"intruder_{file_ts}.jpg"
        save_path = today_dir / filename
        cv2.imwrite(str(save_path), annotated_snapshot)
        logger.info("Saved intruder snapshot to %s", save_path)

        # Calculate relative path for database
        rel_path = save_path.relative_to(BASE_DIR).as_posix()

        # Highest detection confidence among intruders
        highest_conf = max(p.confidence for p in intruders) if intruders else 0.0
        person_count = len(all_detections)

        # 3. Dispatch alert (Telegram + local alarm buzzer with cooldown)
        alert_sent = dispatch_intruder_alert(
            image_path=save_path,
            person_count=person_count,
            confidence=highest_conf,
            custom_timestamp=ts_str,
        )

        # 4. Insert record into SQLite
        event_id = log_intruder_event(
            image_path=rel_path,
            person_count=person_count,
            confidence=highest_conf,
            alert_sent=alert_sent,
            custom_timestamp=ts_str,
        )
        logger.info("Intruder Event #%d logged. Alert sent: %s", event_id, alert_sent)

    def _update_fps(self) -> None:
        """Compute real-time rolling FPS."""
        self.fps_frame_count += 1
        now = time.time()
        elapsed = now - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.last_fps_time = now

    def _draw_hud(self, frame: np.ndarray) -> None:
        """Overlay real-time heads-up-display (HUD) with system metrics."""
        h, w = frame.shape[:2]

        # Top status bar
        cv2.rectangle(frame, (0, 0), (w, 35), (20, 20, 20), cv2.FILLED)

        # FPS & Resolution
        fps_text = f"FPS: {self.fps:.1f} | Frame: {self.frame_index}"
        cv2.putText(frame, fps_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Active Hours & Cooldown Status
        in_hours = is_within_active_hours()
        hours_status = "HOURS: OK" if in_hours else "HOURS: OFF"
        hours_color = (0, 255, 0) if in_hours else (100, 100, 255)
        cv2.putText(frame, hours_status, (250, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hours_color, 1)

        if is_on_cooldown():
            rem = get_remaining_cooldown()
            cd_text = f"COOLDOWN: {rem:.0f}s"
            cd_color = (0, 165, 255)
        else:
            cd_text = "READY"
            cd_color = (0, 255, 0)
        cv2.putText(frame, cd_text, (380, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cd_color, 2)

        # Consecutive frame progress
        if self.consecutive_unknown_count > 0:
            track_text = f"TRACKING: {self.consecutive_unknown_count}/{CONSECUTIVE_FRAMES_THRESHOLD}"
            cv2.putText(frame, track_text, (w - 180, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def main():
    system = IntruderDetectionSystem()
    system.start()


if __name__ == "__main__":
    main()
