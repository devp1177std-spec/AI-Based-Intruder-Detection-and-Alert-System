"""
Configuration module for the AI-Based Intruder Detection and Alert System.
Centralizes all directories, camera settings, detection thresholds,
scheduling, active hours, and alert parameters.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, time
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from dotenv import load_dotenv

# Configure basic logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("IntruderDetection.Config")

# Load environment variables from .env file
BASE_DIR: Path = Path(__file__).resolve().parent
ENV_PATH: Path = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
    logger.info("Loaded environment variables from %s", ENV_PATH)
else:
    logger.warning("No .env file found at %s. Using default/environment variables.", ENV_PATH)

# ==============================================================================
# STORAGE DIRECTORIES
# ==============================================================================
DATA_DIR: Path = BASE_DIR / "data"
DB_PATH: Path = DATA_DIR / "events.db"
KNOWN_FACES_DIR: Path = BASE_DIR / "known_faces"
INTRUDERS_DIR: Path = BASE_DIR / "intruders"
CLIPS_DIR: Path = BASE_DIR / "clips"
MODELS_DIR: Path = BASE_DIR / "models"

# ==============================================================================
# CAMERA & CAPTURE SETTINGS
# ==============================================================================
# CAMERA_SOURCE can be an integer index (e.g. 0 for built-in webcam) or an RTSP/HTTP URL
_raw_camera_source = os.getenv("CAMERA_SOURCE", "0").strip()
CAMERA_SOURCE: Union[int, str]
try:
    CAMERA_SOURCE = int(_raw_camera_source)
except ValueError:
    CAMERA_SOURCE = _raw_camera_source

FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
TARGET_FPS: int = 30
CAMERA_RECONNECT_DELAY: float = 3.0  # seconds to wait before retrying camera open
MAX_CONSECUTIVE_FRAME_FAILS: int = 15

# ==============================================================================
# YOLO & PERSON DETECTION SETTINGS
# ==============================================================================
YOLO_MODEL_NAME: str = "yolov8n.pt"  # Lightweight nano model
YOLO_CONFIDENCE_THRESHOLD: float = 0.50
PERSON_CLASS_ID: int = 0  # COCO class 0 is 'person'

# ==============================================================================
# FACE RECOGNITION SETTINGS
# ==============================================================================
# Match tolerance for face_recognition (128-d Euclidean distance)
# Lower is stricter (0.6 is default, 0.55 prevents false positives)
FACE_MATCH_TOLERANCE: float = 0.55
FACE_DETECTION_MODEL: str = "hog"  # "hog" is fast on CPU; "cnn" requires CUDA

# Optimization: run heavy face recognition only every N frames
PROCESS_EVERY_N_FRAMES: int = 3

# Margin expansion when cropping face from person bbox (as fraction of face size)
FACE_CROP_MARGIN: float = 0.15

# ==============================================================================
# INTRUDER LOGIC & ALERT TRIGGERING
# ==============================================================================
# Number of consecutive frames an UNKNOWN person must be observed to trigger alert
CONSECUTIVE_FRAMES_THRESHOLD: int = 5

# Alert cooldown in seconds to prevent spamming notifications for the same event
ALERT_COOLDOWN_SECONDS: int = 30

# Local audio alarm settings (via winsound on Windows)
ENABLE_AUDIO_ALARM: bool = True
ALARM_FREQUENCY_HZ: int = 1800
ALARM_DURATION_MS: int = 600

# ==============================================================================
# TELEGRAM BOT ALERTS
# ==============================================================================
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_ENABLED: bool = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# ==============================================================================
# RESTRICTED ZONE (OPTIONAL POLYGON)
# ==============================================================================
# When enabled, only persons whose bounding box center or feet enter the polygon
# will trigger the intruder logic.
RESTRICTED_ZONE_ENABLED: bool = False
# Define polygon vertices as (x, y) coordinates within FRAME_WIDTH x FRAME_HEIGHT
RESTRICTED_ZONE_POLYGON: List[Tuple[int, int]] = [
    (100, 100),
    (540, 100),
    (540, 440),
    (100, 440),
]

# ==============================================================================
# ACTIVE HOURS (OPTIONAL SCHEDULE)
# ==============================================================================
# When enabled, alerts are raised ONLY between START and END times
ACTIVE_HOURS_ENABLED: bool = False
ACTIVE_HOURS_START: str = "22:00"  # 10:00 PM (HH:MM 24-hr format)
ACTIVE_HOURS_END: str = "06:00"    # 06:00 AM

# ==============================================================================
# DATA RETENTION & CLEANUP
# ==============================================================================
CLEANUP_ON_STARTUP: bool = True
RETENTION_DAYS: int = 30  # Delete intruder snapshot directories older than X days


# ==============================================================================
# HELPER UTILITIES
# ==============================================================================
def ensure_directories() -> None:
    """Ensure all required project storage directories exist."""
    dirs_to_create = [
        DATA_DIR,
        KNOWN_FACES_DIR,
        INTRUDERS_DIR,
        CLIPS_DIR,
        MODELS_DIR,
    ]
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)
    logger.info("Storage directories verified: data, known_faces, intruders, clips.")


def get_current_date_intruder_dir() -> Path:
    """Return the folder path for today's intruder captures (intruders/YYYY-MM-DD/)."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    today_dir = INTRUDERS_DIR / date_str
    today_dir.mkdir(parents=True, exist_ok=True)
    return today_dir


def is_within_active_hours() -> bool:
    """
    Check if the current local time falls within configured active hours.
    Correctly handles intervals that cross midnight (e.g. 22:00 to 06:00).
    """
    if not ACTIVE_HOURS_ENABLED:
        return True

    now = datetime.now().time()
    try:
        start_parts = [int(p) for p in ACTIVE_HOURS_START.split(":")]
        end_parts = [int(p) for p in ACTIVE_HOURS_END.split(":")]
        start_time = time(start_parts[0], start_parts[1])
        end_time = time(end_parts[0], end_parts[1])
    except Exception as exc:
        logger.error("Failed to parse active hours (%s - %s): %s", ACTIVE_HOURS_START, ACTIVE_HOURS_END, exc)
        return True

    if start_time <= end_time:
        # Standard day interval (e.g. 09:00 to 18:00)
        return start_time <= now <= end_time
    else:
        # Crosses midnight interval (e.g. 22:00 to 06:00)
        return now >= start_time or now <= end_time


def is_box_in_restricted_zone(box: Tuple[int, int, int, int]) -> bool:
    """
    Check if a person's bounding box [x1, y1, x2, y2] intersects the restricted zone.
    Evaluates the bottom-center (footprint) and center of the person box.
    """
    if not RESTRICTED_ZONE_ENABLED or not RESTRICTED_ZONE_POLYGON:
        return True

    x1, y1, x2, y2 = box
    center_pt = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    feet_pt = ((x1 + x2) / 2.0, float(y2))

    pts = np.array(RESTRICTED_ZONE_POLYGON, dtype=np.int32)
    # cv2.pointPolygonTest returns positive if inside, 0 on edge, negative if outside
    dist_center = cv2.pointPolygonTest(pts, center_pt, False)
    dist_feet = cv2.pointPolygonTest(pts, feet_pt, False)

    return dist_center >= 0 or dist_feet >= 0
