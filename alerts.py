"""
Alert notification module.
Handles Telegram Bot alerts with attached evidence photos, non-blocking local audio
buzzers/alarms, and cooldown management to prevent spam.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from config import (
    ALARM_DURATION_MS,
    ALARM_FREQUENCY_HZ,
    ALERT_COOLDOWN_SECONDS,
    ENABLE_AUDIO_ALARM,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_ENABLED,
    is_within_active_hours,
)

logger = logging.getLogger("IntruderDetection.Alerts")

# Global tracking for alert cooldown
_last_alert_timestamp: float = 0.0
_cooldown_lock = threading.Lock()


def is_on_cooldown() -> bool:
    """Return True if the system is currently in the alert cooldown window."""
    with _cooldown_lock:
        elapsed = time.time() - _last_alert_timestamp
        return elapsed < ALERT_COOLDOWN_SECONDS


def get_remaining_cooldown() -> float:
    """Return remaining cooldown seconds, or 0.0 if ready."""
    with _cooldown_lock:
        elapsed = time.time() - _last_alert_timestamp
        remaining = ALERT_COOLDOWN_SECONDS - elapsed
        return max(0.0, remaining)


def record_alert_dispatched() -> None:
    """Update the last alert timestamp to current epoch time."""
    global _last_alert_timestamp
    with _cooldown_lock:
        _last_alert_timestamp = time.time()


def play_alarm_sound() -> None:
    """
    Play a local acoustic alarm in a non-blocking background thread.
    Uses `winsound.Beep` on Windows and terminal bell `\\a` as POSIX fallback.
    """
    if not ENABLE_AUDIO_ALARM:
        return

    def _beep_worker():
        try:
            if platform.system() == "Windows":
                import winsound

                # Multiple alert pulses for urgency
                for _ in range(2):
                    winsound.Beep(ALARM_FREQUENCY_HZ, ALARM_DURATION_MS // 2)
                    time.sleep(0.1)
            else:
                # Terminal bell fallback for Linux/macOS
                print("\a", end="", flush=True)
        except Exception as exc:
            logger.warning("Could not sound local alarm buzzer: %s", exc)

    thread = threading.Thread(target=_beep_worker, daemon=True)
    thread.start()


def send_telegram_alert(
    image_path: Path,
    timestamp: str,
    person_count: int = 1,
    confidence: float = 0.0,
) -> bool:
    """
    Send an intruder alert to configured Telegram chat with the evidence photo attached.

    :param image_path: Local filesystem Path to the evidence JPEG image
    :param timestamp: Human-readable timestamp string
    :param person_count: Number of persons detected
    :param confidence: Highest detector confidence
    :return: True if successfully delivered, False otherwise
    """
    if not TELEGRAM_ENABLED:
        logger.debug("Telegram alerts disabled (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing).")
        return False

    if not image_path.exists():
        logger.error("Cannot send Telegram alert: Image file does not exist: %s", image_path)
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    caption = (
        "🚨 <b>INTRUDER ALERT DETECTED!</b> 🚨\n\n"
        f"🕒 <b>Timestamp:</b> <code>{timestamp}</code>\n"
        f"👥 <b>Persons Detected:</b> {person_count}\n"
        f"🎯 <b>Detection Confidence:</b> {confidence * 100:.1f}%\n"
        f"📁 <b>Evidence File:</b> <code>{image_path.name}</code>\n\n"
        "⚠️ <i>Please inspect your premises immediately.</i>"
    )

    try:
        with open(image_path, "rb") as photo_file:
            files = {"photo": (image_path.name, photo_file, "image/jpeg")}
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": caption,
                "parse_mode": "HTML",
            }
            response = requests.post(url, data=data, files=files, timeout=12.0)

        if response.status_code == 200:
            logger.info("Telegram alert dispatched successfully for image: %s", image_path.name)
            return True
        else:
            logger.error(
                "Telegram API returned HTTP %d: %s",
                response.status_code,
                response.text[:200],
            )
            return False

    except requests.exceptions.Timeout:
        logger.error("Telegram alert request timed out while sending %s", image_path.name)
        return False
    except requests.exceptions.RequestException as exc:
        logger.error("Network error sending Telegram alert: %s", exc)
        return False
    except Exception as exc:
        logger.error("Unexpected error in send_telegram_alert: %s", exc)
        return False


def dispatch_intruder_alert(
    image_path: Path,
    person_count: int,
    confidence: float,
    custom_timestamp: Optional[str] = None,
) -> bool:
    """
    Coordinates full alert dispatch:
    - Checks cooldown and active hours
    - Fires local alarm
    - Sends Telegram notification
    - Updates cooldown timer

    :return: True if alerts were processed, False if suppressed by cooldown or hours
    """
    if not is_within_active_hours():
        logger.info("Intruder detected, but alerts suppressed due to active hours schedule.")
        return False

    if is_on_cooldown():
        rem = get_remaining_cooldown()
        logger.info("Intruder alert suppressed by cooldown (%.1fs remaining).", rem)
        return False

    ts = custom_timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Sound local audio alarm immediately
    play_alarm_sound()

    # 2. Update cooldown timestamp so subsequent frames in quick succession don't re-trigger
    record_alert_dispatched()

    # 3. Send Telegram notification (can run in background or inline)
    # We execute synchronously or via worker, but send_telegram_alert is fast with 12s timeout
    sent = send_telegram_alert(
        image_path=image_path,
        timestamp=ts,
        person_count=person_count,
        confidence=confidence,
    )
    return sent
