"""
Database module for local SQLite logging and analytics.
Stores intruder events with timestamps, relative image paths, person counts,
confidence scores, and alert statuses. Provides statistical queries and cleanup.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import BASE_DIR, DATA_DIR, DB_PATH, RETENTION_DAYS

logger = logging.getLogger("IntruderDetection.Database")


def get_db_connection() -> sqlite3.Connection:
    """Create and return a thread-safe connection to the SQLite database."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize the SQLite database schema and indices."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_path TEXT NOT NULL,
                person_count INTEGER NOT NULL DEFAULT 1,
                confidence REAL NOT NULL DEFAULT 0.0,
                alert_sent INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);"
        )
        conn.commit()
    logger.info("Database initialized successfully at %s", DB_PATH)


def log_intruder_event(
    image_path: str,
    person_count: int = 1,
    confidence: float = 0.0,
    alert_sent: bool = False,
    custom_timestamp: Optional[str] = None,
) -> int:
    """
    Insert a newly detected intruder event into SQLite.

    :param image_path: Relative path to saved snapshot file
    :param person_count: Number of persons detected in frame
    :param confidence: Highest detector confidence score
    :param alert_sent: True if Telegram alert was successfully dispatched
    :param custom_timestamp: Optional explicit timestamp (default is now)
    :return: Generated event ID
    """
    ts = custom_timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Normalize image_path to relative path using forward slashes for cross-platform UI
    rel_path = str(Path(image_path)).replace("\\", "/")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO events (timestamp, image_path, person_count, confidence, alert_sent)
            VALUES (?, ?, ?, ?, ?);
            """,
            (ts, rel_path, person_count, round(float(confidence), 3), 1 if alert_sent else 0),
        )
        conn.commit()
        event_id = cursor.lastrowid or 0

    logger.info("Logged intruder event #%d at %s (alert_sent=%s)", event_id, ts, alert_sent)
    return event_id


def get_events(
    limit: int = 1000,
    date_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve past intruder events with optional date filtering.

    :param limit: Maximum number of events to return (most recent first)
    :param date_filter: 'YYYY-MM-DD' string to filter by date
    :return: List of event dictionaries
    """
    query = "SELECT id, timestamp, image_path, person_count, confidence, alert_sent FROM events"
    params: List[Any] = []

    if date_filter:
        query += " WHERE timestamp LIKE ?"
        params.append(f"{date_filter}%")

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_stats() -> Dict[str, Any]:
    """
    Calculate summary statistics:
    - total_all_time: total intruders detected
    - total_today: intruders detected today
    - total_this_week: intruders detected in past 7 days
    - hourly_counts_today: array of 24 integers representing counts per hour today
    - hourly_counts_all_time: array of 24 integers representing counts per hour all-time
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    week_ago_str = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")

    stats: Dict[str, Any] = {
        "total_all_time": 0,
        "total_today": 0,
        "total_this_week": 0,
        "hourly_counts_today": [0] * 24,
        "hourly_counts_all_time": [0] * 24,
    }

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Total all time
        cursor.execute("SELECT COUNT(*) FROM events;")
        stats["total_all_time"] = cursor.fetchone()[0]

        # Total today
        cursor.execute("SELECT COUNT(*) FROM events WHERE timestamp LIKE ?;", (f"{today_str}%",))
        stats["total_today"] = cursor.fetchone()[0]

        # Total this week
        cursor.execute("SELECT COUNT(*) FROM events WHERE timestamp >= ?;", (week_ago_str,))
        stats["total_this_week"] = cursor.fetchone()[0]

        # Hourly counts today
        cursor.execute(
            """
            SELECT strftime('%H', timestamp) AS hr, COUNT(*)
            FROM events
            WHERE timestamp LIKE ?
            GROUP BY hr;
            """,
            (f"{today_str}%",),
        )
        for row in cursor.fetchall():
            hr = int(row[0])
            stats["hourly_counts_today"][hr] = row[1]

        # Hourly counts all-time
        cursor.execute(
            """
            SELECT strftime('%H', timestamp) AS hr, COUNT(*)
            FROM events
            GROUP BY hr;
            """
        )
        for row in cursor.fetchall():
            if row[0] is not None:
                hr = int(row[0])
                stats["hourly_counts_all_time"][hr] = row[1]

    return stats


def delete_event(event_id: int, remove_image_file: bool = True) -> bool:
    """
    Delete an event by ID from the database and optionally remove the snapshot image.

    :param event_id: The ID of the event to delete
    :param remove_image_file: If True, deletes the local file from disk
    :return: True if a record was deleted
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image_path FROM events WHERE id = ?;", (event_id,))
        row = cursor.fetchone()
        if not row:
            logger.warning("Event #%d not found for deletion.", event_id)
            return False

        image_path_str = row["image_path"]
        cursor.execute("DELETE FROM events WHERE id = ?;", (event_id,))
        conn.commit()

    logger.info("Deleted event #%d from SQLite.", event_id)

    if remove_image_file and image_path_str:
        file_path = BASE_DIR / image_path_str
        try:
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                logger.info("Deleted associated image file: %s", file_path)
        except OSError as exc:
            logger.error("Failed to delete image file %s: %s", file_path, exc)

    return True


def cleanup_old_events(retention_days: int = RETENTION_DAYS) -> int:
    """
    Remove database events and local folders older than `retention_days`.

    :param retention_days: Number of days to keep
    :return: Count of deleted database records
    """
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")

    deleted_count = 0
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, image_path FROM events WHERE timestamp < ?;", (cutoff_str,))
        rows = cursor.fetchall()
        for row in rows:
            img_path = BASE_DIR / row["image_path"]
            try:
                if img_path.exists():
                    img_path.unlink()
            except OSError:
                pass
            deleted_count += 1

        cursor.execute("DELETE FROM events WHERE timestamp < ?;", (cutoff_str,))
        conn.commit()

    # Also clean up empty date folders in intruders/
    intruders_base = BASE_DIR / "intruders"
    if intruders_base.exists():
        for date_dir in intruders_base.iterdir():
            if date_dir.is_dir():
                try:
                    dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                    if dir_date < cutoff_date:
                        for f in date_dir.iterdir():
                            if f.is_file():
                                f.unlink()
                        date_dir.rmdir()
                        logger.info("Removed expired intruder directory: %s", date_dir)
                except ValueError:
                    # Ignore non-date subfolders
                    pass

    if deleted_count > 0:
        logger.info("Retention cleanup: removed %d records older than %s", deleted_count, cutoff_str)
    return deleted_count
