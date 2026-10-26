"""
Streamlit Web Dashboard for the AI-Based Intruder Detection and Alert System.
Features:
- Live/Historical Intruder Event Log with Date Filtering
- Responsive Evidence Snapshot Image Gallery
- Key Performance Metrics (Today, Past 7 Days, All Time)
- 24-Hour Intruder Activity Distribution Chart
- One-Click Event Deletion (removes SQLite row and local image file)
- Start / Stop System Process Controller
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import psutil
import streamlit as st

from config import (
    ACTIVE_HOURS_ENABLED,
    ACTIVE_HOURS_END,
    ACTIVE_HOURS_START,
    ALERT_COOLDOWN_SECONDS,
    BASE_DIR,
    CAMERA_SOURCE,
    INTRUDERS_DIR,
    KNOWN_FACES_DIR,
    RESTRICTED_ZONE_ENABLED,
    TELEGRAM_ENABLED,
    YOLO_CONFIDENCE_THRESHOLD,
    ensure_directories,
)
from database import delete_event, get_events, get_stats, init_db

# Page Configuration
st.set_page_config(
    page_title="AI Intruder Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_directories()
init_db()


# ==============================================================================
# SUBPROCESS MANAGEMENT (START / STOP DETECTOR)
# ==============================================================================
def find_running_detector_process() -> Optional[psutil.Process]:
    """Find any running instance of app.py."""
    current_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.pid == current_pid:
                continue
            cmdline = proc.info.get("cmdline") or []
            cmd_str = " ".join(cmdline).lower()
            if "app.py" in cmd_str and ("python" in cmd_str or "pythonw" in cmd_str):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def is_detector_running() -> bool:
    """Return True if app.py is currently running."""
    return find_running_detector_process() is not None


def start_detector() -> bool:
    """Spawn app.py as an independent background process."""
    if is_detector_running():
        return True

    app_path = BASE_DIR / "app.py"
    python_exe = sys.executable

    try:
        if sys.platform == "win32":
            # Creation flag CREATE_NEW_CONSOLE or detached
            subprocess.Popen(
                [python_exe, str(app_path)],
                cwd=str(BASE_DIR),
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        else:
            subprocess.Popen(
                [python_exe, str(app_path)],
                cwd=str(BASE_DIR),
            )
        time.sleep(1.0)
        return is_detector_running()
    except Exception as exc:
        st.error(f"Failed to launch detector: {exc}")
        return False


def stop_detector() -> bool:
    """Terminate the running app.py process."""
    proc = find_running_detector_process()
    if proc:
        try:
            proc.terminate()
            proc.wait(timeout=3.0)
            return True
        except Exception:
            try:
                proc.kill()
                return True
            except Exception as exc:
                st.error(f"Failed to kill process: {exc}")
                return False
    return True


# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=70)
st.sidebar.title("System Control")

running = is_detector_running()
if running:
    st.sidebar.success("🟢 **Detection System: RUNNING**")
    if st.sidebar.button("🛑 Stop Detection Engine", use_container_width=True, type="secondary"):
        with st.spinner("Stopping system..."):
            stop_detector()
            st.rerun()
else:
    st.sidebar.warning("⚪ **Detection System: STOPPED**")
    if st.sidebar.button("▶️ Start Detection Engine", use_container_width=True, type="primary"):
        with st.spinner("Launching system..."):
            start_detector()
            st.rerun()

st.sidebar.divider()

# System Config Summary
st.sidebar.subheader("Configuration Summary")
st.sidebar.markdown(f"- **Camera Source:** `{CAMERA_SOURCE}`")
st.sidebar.markdown(f"- **Telegram Alerts:** `{'Active ✅' if TELEGRAM_ENABLED else 'Disabled ❌'}`")
st.sidebar.markdown(f"- **Zone Restriction:** `{'Enabled 🛡️' if RESTRICTED_ZONE_ENABLED else 'Disabled 🌐'}`")
st.sidebar.markdown(f"- **Active Hours:** `{'Active ⏰ (' + ACTIVE_HOURS_START + ' - ' + ACTIVE_HOURS_END + ')' if ACTIVE_HOURS_ENABLED else '24/7 Monitoring 🕒'}`")
st.sidebar.markdown(f"- **Cooldown:** `{ALERT_COOLDOWN_SECONDS}s`")
st.sidebar.markdown(f"- **YOLO Confidence:** `{int(YOLO_CONFIDENCE_THRESHOLD * 100)}%`")

# Count enrolled authorized personnel
known_persons = [d.name for d in KNOWN_FACES_DIR.iterdir() if d.is_dir()] if KNOWN_FACES_DIR.exists() else []
st.sidebar.markdown(f"- **Authorized Persons:** `{len(known_persons)} enrolled`")
if known_persons:
    with st.sidebar.expander("View Authorized Roster"):
        for person in known_persons:
            st.write(f"• {person.replace('_', ' ')}")

st.sidebar.divider()
if st.sidebar.button("🔄 Refresh Dashboard Data", use_container_width=True):
    st.rerun()


# ==============================================================================
# MAIN PAGE HEADER & STATS
# ==============================================================================
st.title("🛡️ AI-Based Intruder Detection & Alert System")
st.caption("College Capstone Project • Real-time Computer Vision & Local Forensic Evidence Logging")

stats = get_stats()

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(label="Intruders Today", value=stats["total_today"])
with m2:
    st.metric(label="Past 7 Days", value=stats["total_this_week"])
with m3:
    st.metric(label="Total All-Time Events", value=stats["total_all_time"])
with m4:
    status_label = "Monitoring Online" if running else "Standby Mode"
    st.metric(label="System Status", value=status_label)

st.divider()

# ==============================================================================
# HOURLY ACTIVITY CHART
# ==============================================================================
st.subheader("📊 Intruder Activity by Hour")
hours = [f"{h:02d}:00" for h in range(24)]
hourly_df = pd.DataFrame(
    {
        "Hour": hours,
        "Today": stats["hourly_counts_today"],
        "All-Time": stats["hourly_counts_all_time"],
    }
).set_index("Hour")

chart_view = st.radio("Activity Histogram View:", ["Today", "All-Time"], horizontal=True)
if chart_view == "Today":
    st.bar_chart(hourly_df["Today"], color="#FF4B4B")
else:
    st.bar_chart(hourly_df["All-Time"], color="#1F77B4")

st.divider()

# ==============================================================================
# DATE FILTER & TABS
# ==============================================================================
col_filter, col_spacer = st.columns([2, 4])
with col_filter:
    filter_mode = st.selectbox("📅 Filter Events by Date:", ["All Dates", "Today", "Custom Date"])
    date_query: Optional[str] = None
    if filter_mode == "Today":
        date_query = datetime.now().strftime("%Y-%m-%d")
    elif filter_mode == "Custom Date":
        selected_date = st.date_input("Select Date:", value=datetime.now())
        if selected_date:
            date_query = selected_date.strftime("%Y-%m-%d")

events = get_events(limit=500, date_filter=date_query)

tab_gallery, tab_table, tab_help = st.tabs(["🖼️ Evidence Gallery", "📋 Event Logs Table", "ℹ️ System Manual"])

# ------------------------------------------------------------------------------
# TAB 1: EVIDENCE GALLERY
# ------------------------------------------------------------------------------
with tab_gallery:
    if not events:
        st.info("No intruder events found for the selected date criteria.")
    else:
        st.write(f"Showing **{len(events)}** evidence snapshots:")
        cols_per_row = 3
        rows = [events[i:i + cols_per_row] for i in range(0, len(events), cols_per_row)]

        for row_events in rows:
            cols = st.columns(cols_per_row)
            for idx, event in enumerate(row_events):
                with cols[idx]:
                    img_path_str = event["image_path"]
                    full_img_path = BASE_DIR / img_path_str

                    if full_img_path.exists() and full_img_path.is_file():
                        st.image(
                            str(full_img_path),
                            caption=f"Event #{event['id']} • {event['timestamp']}",
                            use_container_width=True,
                        )
                    else:
                        st.warning(f"Image missing: {img_path_str}")

                    # Metadata details
                    st.markdown(
                        f"**ID:** `#{event['id']}` | **Persons:** `{event['person_count']}`\n\n"
                        f"**Confidence:** `{event['confidence']*100:.0f}%` | "
                        f"**Alert:** `{'Sent ✅' if event['alert_sent'] else 'No ❌'}`"
                    )

                    # Delete Event Button
                    del_btn_key = f"del_{event['id']}"
                    if st.button("🗑️ Delete Event", key=del_btn_key, use_container_width=True):
                        delete_event(event["id"], remove_image_file=True)
                        st.toast(f"Deleted Event #{event['id']}", icon="🗑️")
                        time.sleep(0.4)
                        st.rerun()

                    st.markdown("---")

# ------------------------------------------------------------------------------
# TAB 2: EVENT LOGS TABLE
# ------------------------------------------------------------------------------
with tab_table:
    if not events:
        st.info("No records to display.")
    else:
        df = pd.DataFrame(events)
        df["Alert Sent"] = df["alert_sent"].apply(lambda x: "Yes ✅" if x else "No ❌")
        df["Confidence"] = df["confidence"].apply(lambda x: f"{x*100:.1f}%")
        df["Evidence File"] = df["image_path"]

        display_cols = ["id", "timestamp", "person_count", "Confidence", "Alert Sent", "Evidence File"]
        st.dataframe(
            df[display_cols].rename(
                columns={
                    "id": "Event ID",
                    "timestamp": "Timestamp",
                    "person_count": "Persons Detected",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="📥 Export Event Logs as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"intruder_events_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# ------------------------------------------------------------------------------
# TAB 3: SYSTEM MANUAL & SHORTCUTS
# ------------------------------------------------------------------------------
with tab_help:
    st.markdown("""
    ### 📖 Operator Guide & Controls

    #### 1. Real-time Detection Controls
    - **`python app.py`**: Launches live webcam detection window.
    - **`Q` or `ESC`**: Gracefully terminates the detection loop and saves all states.
    - **`R`**: Hot-reloads face encodings from `known_faces/` without stopping the camera feed.

    #### 2. Registering Authorized Personnel
    Run the enrollment script:
    ```bash
    python register_face.py --name "Alice"
    ```
    - Position face in the green target box.
    - Press **`SPACE`** to snap each of the 5 calibration photos.
    - Once finished, the system will instantly recognize the authorized individual with green bounding boxes.

    #### 3. Telegram Alert Setup
    1. Create a bot using [@BotFather](https://t.me/BotFather) on Telegram and copy the API token.
    2. Message [@userinfobot](https://t.me/userinfobot) to get your numeric `Chat ID`.
    3. Put both values in `.env`:
       ```env
       TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ
       TELEGRAM_CHAT_ID=987654321
       ```
    """)
