import streamlit as st
import cv2
import numpy as np
import uuid
import json
import threading
import psycopg2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from streamlit_autorefresh import st_autorefresh

from core.vision_engine import FaceEngine
from core.storage import get_storage_engine
from logic.enrollment import EnrollmentSession
from logic.augmentation import run_augmentation

# --- Page Config ---
st.set_page_config(page_title="NSL FRAME", layout="wide", initial_sidebar_state="collapsed")

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #f5f5f5;
        font-family: 'Inter', sans-serif;
    }
    #MainMenu, footer, header {visibility: hidden;}

    .block-container { padding-top: 2rem; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        border: 1px solid #e0e0e0 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    }

    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }

    .field-label {
        font-size: 14px;
        font-weight: 500;
        color: #1a1a2e;
        margin-bottom: 4px;
    }
    .field-label .req { color: #e74c3c; }

    .stTextInput > div > div > input {
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 14px;
        font-family: 'Inter', sans-serif;
        color: #333;
        background: #fff;
    }
    .stTextInput > div > div > input:focus {
        border-color: #1a2b6d;
        box-shadow: 0 0 0 2px rgba(26,43,109,0.1);
    }
    .stTextInput > div > div > input::placeholder { color: #bbb; }

    .stButton > button, .stFormSubmitButton > button {
        background-color: #1a2b6d;
        color: #fff;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 15px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-color: #152358;
        color: #fff;
    }

    button[data-testid="stBaseButton-secondary"] {
        background: transparent !important;
        color: #1a2b6d !important;
        border: 1px solid #ddd !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        background: #f0f0f5 !important;
    }

    .stProgress > div > div > div > div {
        background-color: #1a2b6d;
        border-radius: 4px;
    }
    .stProgress > div > div > div {
        background-color: #e8e8e8;
        border-radius: 4px;
    }

    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    .pill-precheck  { background: #fff3e0; color: #e65100; }
    .pill-capturing { background: #e3f2fd; color: #1565c0; }
    .pill-complete  { background: #e8f5e9; color: #2e7d32; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DB Config
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "dbname":   "workflow_system",
    "user":     "postgres",
    "password": "9ets0n1234",
    "host":     "10.26.1.175",
    "port":     "5432",
}

ANGLE_CONFIG = {
    "center":     (True,  True),
    "look_up":    (False, False),
    "look_down":  (False, False),
    "left_semi":  (False, False),
    "right_semi": (False, False),
    "left_full":  (False, False),
    "right_full": (False, False),
}

def save_to_postgres(emp_id, emp_name, bucket_data):
    """Insert face encodings into Postgres. Vectors already normalized at source."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor()
        insert_query = """
        INSERT INTO workflow_runtime.iot_face_encodings
        (face_encoding_id, person_type, encoding_vector, encoding_model, encoding_dimension,
         face_location, face_quality_score, face_angle, is_frontal, is_primary,
         tenant_id, is_active, person_id, person_name)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING;
        """
        count = 0
        for angle, data in bucket_data.items():
            if not data.get("captured"):
                continue
            # Vector already normalized in EnrollmentSession — no re-normalization here
            vector_list = data["vector"].tolist()
            is_frontal, is_primary = ANGLE_CONFIG.get(angle, (False, False))
            cur.execute(insert_query, (
                str(uuid.uuid4()),
                "employee",
                vector_list,
                "insightface",
                len(vector_list),
                None,
                None,
                json.dumps({"angle": angle, "type": "original"}),
                is_frontal,
                is_primary,
                "T689",
                True,
                emp_id,
                emp_name,
            ))
            count += 1
        conn.commit()
        print(f"[DB] Inserted {count} encodings for {emp_id} ({emp_name})")
        return True, f"Saved {count} encodings to database."
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[DB] Error: {e}")
        return False, str(e)
    finally:
        if conn:
            cur.close()
            conn.close()

# ---------------------------------------------------------------------------
# Shared state — survives Streamlit reruns, accessed from WebRTC thread
# ---------------------------------------------------------------------------
@st.cache_resource
def get_shared_state():
    return {
        "lock": threading.Lock(),
        "data": {
            "pitch":         0.0,
            "yaw":           0.0,
            "status":        "",
            "progress":      0.0,
            "buckets_captured": [],
            "session_state": "PRE_CHECK",
        },
    }

_state  = get_shared_state()
_lock   = _state["lock"]
_shared = _state["data"]

# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------
@st.cache_resource
def get_engine():
    return FaceEngine()  # __init__ already calls prepare()

@st.cache_resource
def get_db():
    return get_storage_engine("filesystem")

@st.cache_resource
def get_enrollment_session():
    return EnrollmentSession()

# Force all heavy models to load before any WebRTC connection attempt
get_engine()
get_enrollment_session()
get_db()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_ANGLES = ["center", "look_up", "look_down", "left_semi", "right_semi", "left_full", "right_full"]

RTC_CONFIG = RTCConfiguration({"iceServers": []})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def reset_enrollment():
    get_enrollment_session.clear()
    st.session_state.saved  = False
    st.session_state.db_msg = ""
    with _lock:
        _shared["progress"]      = 0.0
        _shared["status"]        = ""
        _shared["buckets_captured"] = []
        _shared["session_state"] = "PRE_CHECK"

def make_dot(name, label, captured):
    done      = name in captured
    bg        = "#2e7d32" if done else "#e8e8e8"
    border_c  = "#2e7d32" if done else "#d0d0d0"
    text_c    = "#2e7d32" if done else "#aaa"
    icon_html = "<span style='color:#fff;font-size:13px;font-weight:700;'>&#10003;</span>" if done else ""
    return (
        "<div style='display:flex;flex-direction:column;align-items:center;gap:4px;'>"
        f"<div style='width:32px;height:32px;border-radius:50%;background:{bg};"
        f"border:2px solid {border_c};display:flex;align-items:center;justify-content:center;'>"
        f"{icon_html}</div>"
        f"<span style='font-size:11px;color:{text_c};font-weight:500;'>{label}</span>"
        "</div>"
    )

# ---------------------------------------------------------------------------
# Video processor
# ---------------------------------------------------------------------------
class FaceEnrollmentProcessor(VideoProcessorBase):
    def __init__(self):
        self.engine = get_engine()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Always fetch current session so Retry/Back resets are picked up
        self.session = get_enrollment_session()

        raw = frame.to_ndarray(format="bgr24")
        # FIX: run ML on raw unflipped frame to match recognition pipeline
        # Only flip the display output so it feels like a mirror to the user
        out = cv2.flip(raw, 1)
        h_img, w_img = raw.shape[:2]
        updates = {}

        faces = self.engine.process_frame(raw)

        if len(faces) == 1:
            face = faces[0]
            pitch, yaw, roll = self.engine.compute_pose(face)
            # No yaw mirror needed — using raw unflipped frame now

            b  = face.bbox.astype(int)
            x1, y1 = max(0, b[0]), max(0, b[1])
            x2, y2 = min(w_img, b[2]), min(h_img, b[3])

            # Face crop from raw frame for correct embedding
            face_crop = raw[y1:y2, x1:x2].copy()

            # Mirror bbox x-coords for drawing on the flipped display frame
            dx1, dx2 = w_img - x2, w_img - x1

            updates["pitch"] = float(pitch)
            updates["yaw"]   = float(yaw)
            updates["buckets_captured"] = [
                name for name in ALL_ANGLES
                if self.session.get_progress()[1].get(name, {}).get("captured", False)
            ]

            sess_state = self.session.state

            if sess_state == "PRE_CHECK":
                is_ready, msg = self.session.run_pre_check(face, pitch, yaw)
                color = (0, 220, 0) if is_ready else (80, 80, 255)
                cv2.rectangle(out, (dx1, y1), (dx2, y2), color, 2)
                cv2.putText(out, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                updates["status"] = msg

            elif sess_state == "CAPTURING":
                if face.det_score >= 0.65:
                    status, _ = self.session.process_face_capture(face, pitch, yaw, face_crop)
                else:
                    status = "Low Quality — improve lighting"
                progress, _ = self.session.get_progress()
                updates["status"]   = status
                updates["progress"] = progress
                cv2.rectangle(out, (dx1, y1), (dx2, y2), (0, 200, 200), 2)
                cv2.putText(out, f"{int(progress * 100)}%  {status}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

            elif sess_state == "COMPLETE":
                cv2.rectangle(out, (dx1, y1), (dx2, y2), (0, 200, 0), 2)
                cv2.putText(out, "ALL ANGLES CAPTURED",
                            (w_img // 2 - 180, h_img // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
                updates["status"]   = "complete"
                updates["progress"] = 1.0

            updates["session_state"] = sess_state

        else:
            msg = "Show only 1 face" if len(faces) > 1 else "No face detected"
            cv2.putText(out, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2)
            updates["status"] = msg

        with _lock:
            _shared.update(updates)

        return av.VideoFrame.from_ndarray(out, format="bgr24")


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "step"     not in st.session_state: st.session_state.step     = "form"
if "emp_name" not in st.session_state: st.session_state.emp_name = ""
if "emp_id"   not in st.session_state: st.session_state.emp_id   = ""
if "saved"    not in st.session_state: st.session_state.saved    = False
if "db_msg"   not in st.session_state: st.session_state.db_msg   = ""

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; margin-bottom:24px;">
    <div style="font-size:26px;">
        <span style="color:#1a2b6d; font-weight:700;">NSL</span>
        <span style="color:#1a1a2e; font-weight:700;"> FRAME</span>
    </div>
    <div style="font-size:13px; color:#999; margin-top:2px;">Face Registration & Enrollment System</div>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# STEP 1 — FORM
# ===========================================================================
if st.session_state.step == "form":
    _, col_form, _ = st.columns([1, 2, 1])
    with col_form:
        with st.container(border=True):
            st.markdown('<div style="font-size:18px; font-weight:600; color:#1a1a2e; margin-bottom:20px;">Employee Details</div>',
                        unsafe_allow_html=True)
            with st.form("employee_form"):
                st.markdown('<div class="field-label">Employee Name<span class="req">*</span></div>', unsafe_allow_html=True)
                emp_name = st.text_input("name", placeholder="Enter employee name", label_visibility="collapsed")

                st.markdown('<div class="field-label" style="margin-top:12px;">Employee ID<span class="req">*</span></div>', unsafe_allow_html=True)
                emp_id = st.text_input("id", placeholder="Enter employee ID", label_visibility="collapsed")

                st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
                submitted = st.form_submit_button("Start Enrollment")

                if submitted:
                    if not emp_name.strip() or not emp_id.strip():
                        st.error("Please fill in both fields.")
                    else:
                        st.session_state.emp_name = emp_name.strip()
                        st.session_state.emp_id   = emp_id.strip()
                        st.session_state.step     = "capture"
                        reset_enrollment()
                        st.rerun()


# ===========================================================================
# STEP 2 — CAPTURE
# ===========================================================================
elif st.session_state.step == "capture":

    st_autorefresh(interval=2000, key="capture_refresh")

    # Read shared state once at top of render
    with _lock:
        status   = _shared["status"]
        progress = _shared["progress"]
        captured = list(_shared["buckets_captured"])
        state    = _shared["session_state"]

    col_video, col_info = st.columns([3, 1.5], gap="medium")

    # ── Video column ────────────────────────────────────────────────────────
    with col_video:
        st.markdown(f"""
        <div style="background:#fff; border:1px solid #e8e8e8; border-radius:12px;
                    padding:12px 20px; margin-bottom:12px;">
            <span style="color:#888; font-size:13px;">Employee:</span>
            <span style="font-weight:600; color:#1a1a2e; margin-left:6px;">{st.session_state.emp_name}</span>
            <span style="color:#ddd; margin:0 12px;">|</span>
            <span style="color:#888; font-size:13px;">ID:</span>
            <span style="font-weight:600; color:#1a1a2e; margin-left:6px;">{st.session_state.emp_id}</span>
        </div>
        """, unsafe_allow_html=True)

        webrtc_streamer(
            key="enrollment",
            video_processor_factory=FaceEnrollmentProcessor,
            rtc_configuration=RTC_CONFIG,
            async_processing=True,
            media_stream_constraints={
                "video": {
                    "width":     {"ideal": 1280, "max": 1280},
                    "height":    {"ideal": 720,  "max": 720},
                    "frameRate": {"ideal": 15,   "max": 20},
                },
                "audio": False,
            },
        )

    # ── Info column ─────────────────────────────────────────────────────────
    with col_info:

        if state == "PRE_CHECK":
            pill_class, pill_text = "pill-precheck", "Pre-Check"
        elif state == "CAPTURING":
            pill_class, pill_text = "pill-capturing", f"Capturing {int(progress * 100)}%"
        else:
            pill_class, pill_text = "pill-complete", "Complete"

        with st.container(border=True):
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
                <span style="font-size:15px; font-weight:600; color:#1a1a2e;">Capture Progress</span>
                <span class="status-pill {pill_class}">{pill_text}</span>
            </div>
            """, unsafe_allow_html=True)

            if state in ("CAPTURING", "COMPLETE"):
                st.progress(progress)

            # 7-dot layout
            dot_up   = make_dot("look_up",    "Up",     captured)
            dot_l90  = make_dot("left_full",  "L 90",   captured)
            dot_l45  = make_dot("left_semi",  "L 45",   captured)
            dot_ctr  = make_dot("center",     "Center", captured)
            dot_r45  = make_dot("right_semi", "R 45",   captured)
            dot_r90  = make_dot("right_full", "R 90",   captured)
            dot_down = make_dot("look_down",  "Down",   captured)

            st.markdown(
                "<div style='margin:16px 0;'>"
                "<div style='display:flex;justify-content:center;margin-bottom:10px;'>" + dot_up + "</div>"
                "<div style='display:flex;justify-content:center;gap:12px;align-items:center;'>"
                + dot_l90 + dot_l45 + dot_ctr + dot_r45 + dot_r90 +
                "</div>"
                "<div style='display:flex;justify-content:center;margin-top:10px;'>" + dot_down + "</div>"
                "</div>",
                unsafe_allow_html=True
            )

            done_count = len(captured)
            st.markdown(f'<div style="text-align:center;font-size:13px;color:#888;">{done_count} of 7 captured</div>',
                        unsafe_allow_html=True)

            if status and status != "complete":
                st.markdown(f'<div style="text-align:center;font-size:13px;color:#555;margin-top:8px;font-weight:500;">{status}</div>',
                            unsafe_allow_html=True)

        # Save button
        if state == "COMPLETE" and not st.session_state.saved:
            st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
            if st.button("Save & Complete Enrollment", use_container_width=True, key="btn_save"):
                session  = get_enrollment_session()
                _, buckets = session.get_progress()
                emp_id   = st.session_state.emp_id
                emp_name = st.session_state.emp_name

                # 1. Save to filesystem (npy + metadata) — no re-normalization
                get_db().save_identity(emp_id, emp_name, buckets)

                # 1b. Augment (toggle via AUGMENTATION_ENABLED in logic/augmentation.py)
                run_augmentation(emp_id, buckets, get_engine(), get_db())

                # 2. Push to Postgres — no re-normalization
                ok, result = save_to_postgres(emp_id, emp_name, buckets)
                st.session_state.db_msg = result
                st.session_state.saved  = True
                st.rerun()

        # Success banner + Enroll Next
        if st.session_state.saved:
            st.markdown(f"""
            <div style="background:#e8f5e9; border:1px solid #c8e6c9; border-radius:12px;
                        padding:20px; text-align:center; margin-top:8px;">
                <div style="font-size:28px; margin-bottom:8px;">✅</div>
                <div style="color:#2e7d32; font-weight:700; font-size:16px;">
                    {st.session_state.emp_name} Enrolled Successfully!
                </div>
                <div style="color:#555; font-size:12px; margin-top:6px;">
                    ID: {st.session_state.emp_id} &nbsp;|&nbsp; {st.session_state.db_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)
            if st.button("➕  Enroll Next Employee", use_container_width=True, key="btn_enroll_next"):
                reset_enrollment()
                st.session_state.step     = "form"
                st.session_state.emp_name = ""
                st.session_state.emp_id   = ""
                st.rerun()

        # Back / Retry — hidden after save
        elif not st.session_state.saved:
            st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
            col_a, col_b = st.columns(2, gap="small")
            with col_a:
                if st.button("Back", use_container_width=True, key="btn_back", type="secondary"):
                    reset_enrollment()
                    st.session_state.step = "form"
                    st.rerun()
            with col_b:
                if st.button("Retry", use_container_width=True, key="btn_retry", type="secondary"):
                    reset_enrollment()
                    st.rerun()