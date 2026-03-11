import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading
from streamlit_autorefresh import st_autorefresh

from core.vision_engine import FaceEngine
from core.storage import get_storage_engine
from logic.enrollment import EnrollmentSession

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
    .pill-precheck { background: #fff3e0; color: #e65100; }
    .pill-capturing { background: #e3f2fd; color: #1565c0; }
    .pill-complete { background: #e8f5e9; color: #2e7d32; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Shared state — survives Streamlit reruns, accessed from WebRTC thread
# ---------------------------------------------------------------------------
@st.cache_resource
def get_shared_state():
    return {
        "lock": threading.Lock(),
        "data": {
            "pitch": 0.0,
            "yaw": 0.0,
            "status": "",
            "progress": 0.0,
            "buckets_captured": [],
            "session_state": "PRE_CHECK",   # mirror of EnrollmentSession.state
        },
    }

_state = get_shared_state()
_lock  = _state["lock"]
_shared = _state["data"]

# ---------------------------------------------------------------------------
# Cached singletons — only stateless/heavy-init objects go here
# ---------------------------------------------------------------------------
@st.cache_resource
def get_engine():
    return FaceEngine()

@st.cache_resource
def get_db():
    return get_storage_engine("filesystem")

# EnrollmentSession is stateful — one shared instance per app session,
# but we access it only through _shared["session_state"] from the UI thread.
@st.cache_resource
def get_enrollment_session():
    return EnrollmentSession()

def reset_enrollment():
    """Clear session state and shared data atomically."""
    get_enrollment_session.clear()
    with _lock:
        _shared["progress"] = 0.0
        _shared["status"] = ""
        _shared["buckets_captured"] = []
        _shared["session_state"] = "PRE_CHECK"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_ANGLES = ["center", "look_up", "look_down", "left_semi", "right_semi", "left_full", "right_full"]

RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        # TURN relay — critical for Cloudflare tunnel (blocks WebRTC UDP)
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
})

# ---------------------------------------------------------------------------
# Video processor
# ---------------------------------------------------------------------------
class FaceEnrollmentProcessor(VideoProcessorBase):
    def __init__(self):
        self.engine       = get_engine()
        self.session      = get_enrollment_session()
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        out = img.copy()            # draw on a copy so img stays clean for crop
        h_img, w_img = out.shape[:2]
        updates = {}

        faces = self.engine.process_frame(img)

        if len(faces) == 1:
            face = faces[0]
            pitch, yaw, roll = self.engine.compute_pose(face)
            yaw = -yaw

            b = face.bbox.astype(int)
            x1, y1 = max(0, b[0]), max(0, b[1])
            x2, y2 = min(w_img, b[2]), min(h_img, b[3])
            face_crop = img[y1:y2, x1:x2].copy()

            updates["pitch"] = float(pitch)
            updates["yaw"]   = float(yaw)
            updates["buckets_captured"] = [
                name for name in ALL_ANGLES
                if self.session.get_progress()[1].get(name, {}).get("captured", False)
            ]

            sess_state = self.session.state   # read once to avoid race

            if sess_state == "PRE_CHECK":
                is_ready, msg = self.session.run_pre_check(face, pitch, yaw)
                color = (0, 220, 0) if is_ready else (80, 80, 255)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                updates["status"] = msg

            elif sess_state == "CAPTURING":
                status, _ = self.session.process_face_capture(face, pitch, yaw, face_crop)
                progress, _ = self.session.get_progress()
                updates["status"]   = status
                updates["progress"] = progress
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 200), 2)
                cv2.putText(out, f"{int(progress * 100)}%  {status}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

            elif sess_state == "COMPLETE":
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
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

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; margin-bottom:24px;">
    <div style="font-size:26px;">
        <span style="color:#1a2b6d; font-weight:700;">NSL</span>
        <span style="color:#1a1a2e; font-weight:700;">FRAME</span>
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

    # Read shared state once at the top of this render
    with _lock:
        status   = _shared["status"]
        progress = _shared["progress"]
        captured = list(_shared["buckets_captured"])
        state    = _shared["session_state"]   # FIX: read from shared, not from session object

    # FIX: only auto-refresh while actively capturing; stop when done
    if state not in ("COMPLETE",) and not st.session_state.saved:
        st_autorefresh(interval=2000, key="capture_refresh")

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
            async_processing=True,   # FIX: don't block recv() — critical for smooth video
            media_stream_constraints={
                "video": {
                    "width":     {"ideal": 640, "max": 640},
                    "height":    {"ideal": 480, "max": 480},
                    "frameRate": {"ideal": 15,  "max": 20},
                },
                "audio": False,
            },
        )

    # ── Info column ─────────────────────────────────────────────────────────
    with col_info:

        # Status pill
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
            def make_dot(name, label):
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

            dot_up    = make_dot("look_up",    "Up")
            dot_l90   = make_dot("left_full",  "L 90")
            dot_l45   = make_dot("left_semi",  "L 45")
            dot_ctr   = make_dot("center",     "Center")
            dot_r45   = make_dot("right_semi", "R 45")
            dot_r90   = make_dot("right_full", "R 90")
            dot_down  = make_dot("look_down",  "Down")

            dots_html = (
                "<div style='margin:16px 0;'>"
                  "<div style='display:flex;justify-content:center;margin-bottom:10px;'>" + dot_up + "</div>"
                  "<div style='display:flex;justify-content:center;gap:12px;align-items:center;'>"
                    + dot_l90 + dot_l45 + dot_ctr + dot_r45 + dot_r90 +
                  "</div>"
                  "<div style='display:flex;justify-content:center;margin-top:10px;'>" + dot_down + "</div>"
                "</div>"
            )
            st.markdown(dots_html, unsafe_allow_html=True)

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
                session = get_enrollment_session()
                _, buckets = session.get_progress()
                get_db().save_identity(st.session_state.emp_id, st.session_state.emp_name, buckets)
                st.session_state.saved = True
                st.rerun()

        # Success banner
        if st.session_state.saved:
            st.markdown("""
            <div style="background:#e8f5e9; border:1px solid #c8e6c9; border-radius:12px;
                        padding:14px; text-align:center; margin-top:8px;">
                <div style="color:#2e7d32; font-weight:600; font-size:14px;">Enrollment saved successfully!</div>
                <div style="color:#666; font-size:12px; margin-top:4px;">Face data stored. You can enroll another employee.</div>
            </div>
            """, unsafe_allow_html=True)

        # Back / Retry
        st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2, gap="small")
        with col_a:
            if st.button("Back", use_container_width=True, key="btn_back", type="secondary"):
                reset_enrollment()
                st.session_state.step  = "form"
                st.session_state.saved = False
                st.rerun()
        with col_b:
            if st.button("Retry", use_container_width=True, key="btn_retry", type="secondary"):
                reset_enrollment()
                st.session_state.saved = False
                st.rerun()