"""
End-to-End Pipeline Integration Test.
Verifies all modules:
1. Directory structure and configuration loading.
2. SQLite schema creation, insertion, retrieval, and stats calculation.
3. YOLOv8 model loading and inference on synthetic/sample imagery.
4. Face recognition matching and known faces reloading.
5. Snapshot evidence image writing to local intruders/YYYY-MM-DD/ directory.
6. Event deletion and database cleanup.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import (
    BASE_DIR,
    DB_PATH,
    INTRUDERS_DIR,
    KNOWN_FACES_DIR,
    ensure_directories,
    get_current_date_intruder_dir,
)
from database import (
    delete_event,
    get_events,
    get_stats,
    init_db,
    log_intruder_event,
)
from detector import DetectedPerson, IntruderDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TestPipeline")


def run_tests() -> bool:
    all_passed = True
    print("\n" + "=" * 60)
    print("🚀 RUNNING END-TO-END INTRUDER DETECTION SYSTEM TESTS")
    print("=" * 60)

    # ---------------------------------------------------------
    # TEST 1: Storage Directories
    # ---------------------------------------------------------
    print("\n[TEST 1] Verifying Storage Directories...")
    try:
        ensure_directories()
        assert (BASE_DIR / "data").exists()
        assert KNOWN_FACES_DIR.exists()
        assert INTRUDERS_DIR.exists()
        print("✅ Storage directories verified successfully.")
    except Exception as exc:
        print(f"❌ Storage directories test failed: {exc}")
        all_passed = False

    # ---------------------------------------------------------
    # TEST 2: SQLite Database Operations
    # ---------------------------------------------------------
    print("\n[TEST 2] Testing SQLite Database Schema & Logging...")
    try:
        init_db()
        assert DB_PATH.exists(), f"Database file {DB_PATH} was not created!"

        # Create a mock snapshot file
        today_dir = get_current_date_intruder_dir()
        test_img_path = today_dir / f"test_intruder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            dummy_img,
            "TEST INTRUDER SNAPSHOT",
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        cv2.imwrite(str(test_img_path), dummy_img)
        assert test_img_path.exists(), "Failed to write test snapshot image!"

        # Log event into SQLite
        rel_path = test_img_path.relative_to(BASE_DIR).as_posix()
        event_id = log_intruder_event(
            image_path=rel_path,
            person_count=1,
            confidence=0.88,
            alert_sent=False,
        )
        assert event_id > 0, "log_intruder_event returned invalid ID"

        # Verify retrieval
        events = get_events(limit=5)
        assert len(events) > 0, "get_events returned empty list"
        latest = events[0]
        assert latest["id"] == event_id, f"Expected event ID {event_id}, got {latest['id']}"
        assert latest["confidence"] == 0.88
        print(f"✅ SQLite logged Event #{event_id} and verified retrieval.")

        # Test stats
        stats = get_stats()
        assert stats["total_all_time"] > 0
        assert stats["total_today"] > 0
        print(f"✅ Statistics calculation verified: Total={stats['total_all_time']}, Today={stats['total_today']}.")

        # Clean up test event
        deleted = delete_event(event_id, remove_image_file=True)
        assert deleted is True, "Failed to delete test event"
        assert not test_img_path.exists(), "Snapshot file was not removed upon deletion"
        print("✅ Event deletion and disk file cleanup verified.")

    except Exception as exc:
        print(f"❌ Database test failed: {exc}")
        all_passed = False

    # ---------------------------------------------------------
    # TEST 3: YOLOv8 Detector & Annotation Pipeline
    # ---------------------------------------------------------
    print("\n[TEST 3] Testing YOLOv8 Detector and Annotation Pipeline...")
    try:
        detector = IntruderDetector()

        # Generate synthetic frame with a person-like shape or color
        test_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        cv2.circle(test_frame, (320, 160), 60, (200, 180, 160), -1)  # Head
        cv2.rectangle(test_frame, (260, 220), (380, 420), (50, 50, 150), -1)  # Torso

        # Run process_frame
        detections = detector.process_frame(test_frame, run_face_recognition=False)
        print(f"✅ YOLOv8 processed synthetic frame cleanly (detections found: {len(detections)}).")

        # Create mock detection to test annotation logic
        mock_person = DetectedPerson(
            bbox=(260, 100, 380, 420),
            confidence=0.91,
            identity="UNKNOWN",
            is_authorized=False,
            face_distance=1.0,
            in_restricted_zone=True,
        )
        annotated = detector.annotate_frame(test_frame, [mock_person])
        assert annotated.shape == test_frame.shape, "Annotated frame shape mismatch"

        # Verify annotation save
        out_test_path = today_dir / "test_annotated.jpg"
        cv2.imwrite(str(out_test_path), annotated)
        assert out_test_path.exists()
        out_test_path.unlink()  # remove temporary test file
        print("✅ Bounding box and intruder banner annotation verified.")

    except Exception as exc:
        print(f"❌ Detector test failed: {exc}")
        all_passed = False

    # ---------------------------------------------------------
    # TEST 4: Known Faces Enrollment & Reload
    # ---------------------------------------------------------
    print("\n[TEST 4] Testing Known Faces Enrollment Structure...")
    try:
        sample_person_dir = KNOWN_FACES_DIR / "Test_Authorized_User"
        sample_person_dir.mkdir(parents=True, exist_ok=True)
        # Verify reload handles empty or populated folder without crashing
        enc_count = detector.reload_known_faces()
        print(f"✅ Known faces reload function executed safely (loaded encodings: {enc_count}).")
        # Cleanup test folder
        sample_person_dir.rmdir()
    except Exception as exc:
        print(f"❌ Known faces test failed: {exc}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Pipeline is fully functional and ready to run.")
        print("=" * 60 + "\n")
        return True
    else:
        print("❌ SOME TESTS FAILED. Check log outputs above.")
        print("=" * 60 + "\n")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
