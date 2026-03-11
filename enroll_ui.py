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

    /* Remove default padding/gaps */
    .block-container { padding-top: 2rem; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }

    /* Streamlit container border override */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        border: 1px solid #e0e0e0 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    }

    /* Form container */
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }

    /* Labels */
    .field-label {
        font-size: 14px;
        font-weight: 500;
        color: #1a1a2e;
        margin-bottom: 4px;
    }
    .field-label .req { color: #e74c3c; }

    /* Inputs */
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

    /* Buttons */
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

    /* Back/Retry buttons (non-primary) */
    button[data-testid="stBaseButton-secondary"] {
        background: transparent !important;
        color: #1a2b6d !important;
        border: 1px solid #ddd !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        background: #f0f0f5 !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #1a2b6d;
        border-radius: 4px;
    }
    .stProgress > div > div > div {
        background-color: #e8e8e8;
        border-radius: 4px;
    }

    /* Status pill */
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

# --- Shared state (thread-safe, survives Streamlit reruns) ---
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
        },
    }

_state = get_shared_state()
_lock = _state["lock"]
_shared = _state["data"]

@st.cache_resource
def get_engine():
    return FaceEngine()

@st.cache_resource
def get_enrollment_session():
    return EnrollmentSession()

@st.cache_resource
def get_db():
    return get_storage_engine("filesystem")

ALL_ANGLES = ["center", "look_up", "look_down", "left_semi", "right_semi", "left_full", "right_full"]
ANGLE_LABELS = {
    "center": "Center",
    "look_up": "Look Up",
    "look_down": "Look Down",
    "left_semi": "Left 45",
    "right_semi": "Right 45",
    "left_full": "Left 90",
    "right_full": "Right 90",
}


class FaceEnrollmentProcessor(VideoProcessorBase):
    def __init__(self):
        self.engine = get_engine()
        self.session = get_enrollment_session()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        if self._frame_count % 2 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        faces = self.engine.process_frame(img)

        if len(faces) == 1:
            face = faces[0]
            pitch, yaw, roll = self.engine.compute_pose(face)
            yaw = -yaw  # Mirror yaw to match flipped video (user's perspective)

            b = face.bbox.astype(int)
            h_img, w_img, _ = img.shape
            x1, y1 = max(0, b[0]), max(0, b[1])
            x2, y2 = min(w_img, b[2]), min(h_img, b[3])
            face_crop = img[y1:y2, x1:x2].copy()

            # Update shared state for UI
            captured_list = []
            _, buckets = self.session.get_progress()
            for name in ALL_ANGLES:
                info = buckets.get(name, {})
                if info.get("captured", False):
                    captured_list.append(name)

            with _lock:
                _shared["pitch"] = float(pitch)
                _shared["yaw"] = float(yaw)
                _shared["buckets_captured"] = captured_list

            if self.session.state == "PRE_CHECK":
                is_ready, msg = self.session.run_pre_check(face, pitch, yaw)
                color = (0, 255, 0) if is_ready else (100, 100, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                with _lock:
                    _shared["status"] = msg

            elif self.session.state == "CAPTURING":
                status, _ = self.session.process_face_capture(face, pitch, yaw, face_crop)
                progress, _ = self.session.get_progress()
                with _lock:
                    _shared["status"] = status
                    _shared["progress"] = progress
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 200), 2)
                cv2.putText(img, f"{int(progress * 100)}%  {status}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

            elif self.session.state == "COMPLETE":
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(img, "ALL ANGLES CAPTURED", (w_img // 2 - 180, h_img // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
                with _lock:
                    _shared["status"] = "complete"
                    _shared["progress"] = 1.0
        else:
            msg = "Show only 1 face" if len(faces) > 1 else "No face detected"
            cv2.putText(img, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
            with _lock:
                _shared["status"] = msg

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- Session State ---
if "step" not in st.session_state:
    st.session_state.step = "form"  # "form" or "capture"
if "emp_name" not in st.session_state:
    st.session_state.emp_name = ""
if "emp_id" not in st.session_state:
    st.session_state.emp_id = ""
if "saved" not in st.session_state:
    st.session_state.saved = False

# --- Header ---
st.markdown("""
<div style="text-align: center; margin-bottom: 24px;">
    <div style="font-size: 26px;">
        <span style="color: #1a2b6d; font-weight: 700;">NSL</span> <span style="color: #1a1a2e; font-weight: 700;">FRAME</span>
    </div>
    <div style="font-size: 13px; color: #999; margin-top: 2px;">Face Registration & Enrollment System</div>
</div>
""", unsafe_allow_html=True)


# ==================== STEP 1: FORM ====================
if st.session_state.step == "form":
    # Center the form card
    _, col_form, _ = st.columns([1, 2, 1])

    with col_form:
        with st.container(border=True):
            st.markdown('<div style="font-size:18px; font-weight:600; color:#1a1a2e; margin-bottom:20px;">Employee Details</div>', unsafe_allow_html=True)

            with st.form("employee_form"):
                st.markdown('<div class="field-label">Employee Name<span class="req">*</span></div>', unsafe_allow_html=True)
                emp_name = st.text_input("name", placeholder="Enter employee name", label_visibility="collapsed")

                st.markdown('<div class="field-label" style="margin-top:12px;">Employee ID<span class="req">*</span></div>', unsafe_allow_html=True)
                emp_id = st.text_input("id", placeholder="Enter employee ID", label_visibility="collapsed")

                st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
                submitted = st.form_submit_button("Start Enrollment")

                if submitted:
                    if not emp_name or not emp_id:
                        st.error("Please fill in both fields.")
                    else:
                        st.session_state.emp_name = emp_name
                        st.session_state.emp_id = emp_id
                        st.session_state.step = "capture"
                        get_enrollment_session.clear()
                        with _lock:
                            _shared["progress"] = 0.0
                            _shared["status"] = ""
                            _shared["buckets_captured"] = []
                        st.rerun()


# ==================== STEP 2: CAPTURE ====================
elif st.session_state.step == "capture":

    # Auto-refresh every 1.5s so dots/progress/buttons update live
    st_autorefresh(interval=2000, key="capture_refresh")

    col_video, col_info = st.columns([3, 1.5], gap="medium")

    with col_video:
        # Employee info bar
        st.markdown(f"""
        <div style="background:#fff; border:1px solid #e8e8e8; border-radius:12px; padding:12px 20px; margin-bottom:12px;">
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
            rtc_configuration=RTCConfiguration({
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},  # fallback STUN
                ]
            }),
            # media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
            media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 15}, "audio": False},

        )

    with col_info:
        # Read shared state
        with _lock:
            status = _shared["status"]
            progress = _shared["progress"]
            captured = list(_shared["buckets_captured"])

        session = get_enrollment_session()
        state = session.state

        # Status pill
        if state == "PRE_CHECK":
            pill_class = "pill-precheck"
            pill_text = "Pre-Check"
        elif state == "CAPTURING":
            pill_class = "pill-capturing"
            pill_text = f"Capturing {int(progress * 100)}%"
        else:
            pill_class = "pill-complete"
            pill_text = "Complete"

        with st.container(border=True):
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
                <span style="font-size:15px; font-weight:600; color:#1a1a2e;">Capture Progress</span>
                <span class="status-pill {pill_class}">{pill_text}</span>
            </div>
            """, unsafe_allow_html=True)

            if state in ("CAPTURING", "COMPLETE"):
                st.progress(progress)

            # --- 7 dots layout ---
            def make_dot(name, label):
                done = name in captured
                bg = "#2e7d32" if done else "#e8e8e8"
                border = "#2e7d32" if done else "#d0d0d0"
                text_c = "#2e7d32" if done else "#aaa"
                icon = "&#10003;" if done else ""
                icon_html = f'<span style="color:#fff;font-size:13px;font-weight:700;">{icon}</span>' if done else ""
                return f'''<div style="display:flex;flex-direction:column;align-items:center;gap:4px;">
                    <div style="width:32px;height:32px;border-radius:50%;background:{bg};border:2px solid {border};display:flex;align-items:center;justify-content:center;">{icon_html}</div>
                    <span style="font-size:11px;color:{text_c};font-weight:500;">{label}</span>
                </div>'''

            # Row 1: Up
            # Row 2: Left90  Left45  Center  Right45  Right90
            # Row 3: Down
            dots_html = f'''
            <div style="margin:16px 0;">
                <div style="display:flex;justify-content:center;margin-bottom:10px;">
                    {make_dot("look_up", "Up")}
                </div>
                <div style="display:flex;justify-content:center;gap:12px;align-items:center;">
                    {make_dot("left_full", "L 90")}
                    {make_dot("left_semi", "L 45")}
                    {make_dot("center", "Center")}
                    {make_dot("right_semi", "R 45")}
                    {make_dot("right_full", "R 90")}
                </div>
                <div style="display:flex;justify-content:center;margin-top:10px;">
                    {make_dot("look_down", "Down")}
                </div>
            </div>
            '''
            st.markdown(dots_html, unsafe_allow_html=True)

            done_count = len(captured)
            st.markdown(f'<div style="text-align:center;font-size:13px;color:#888;">{done_count} of 7 captured</div>', unsafe_allow_html=True)

            if status and status != "complete":
                st.markdown(f'<div style="text-align:center; font-size:13px; color:#555; margin-top:8px; font-weight:500;">{status}</div>', unsafe_allow_html=True)

        # Save button — visible when all 7 angles captured
        if state == "COMPLETE" and not st.session_state.saved:
            st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
            if st.button("Save & Complete Enrollment", use_container_width=True, key="btn_save"):
                _, buckets = session.get_progress()
                get_db().save_identity(st.session_state.emp_id, st.session_state.emp_name, buckets)
                st.session_state.saved = True
                st.rerun()

        # Success message
        if st.session_state.saved:
            st.markdown("""
            <div style="background:#e8f5e9; border:1px solid #c8e6c9; border-radius:12px; padding:14px; text-align:center; margin-top:8px;">
                <div style="color:#2e7d32; font-weight:600; font-size:14px;">Enrollment saved successfully!</div>
                <div style="color:#666; font-size:12px; margin-top:4px;">Face data stored. You can enroll another employee.</div>
            </div>
            """, unsafe_allow_html=True)

        # Action buttons
        st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2, gap="small")
        with col_a:
            if st.button("Back", use_container_width=True, key="btn_back", type="secondary"):
                get_enrollment_session.clear()
                st.session_state.step = "form"
                st.session_state.saved = False
                with _lock:
                    _shared["progress"] = 0.0
                    _shared["status"] = ""
                    _shared["buckets_captured"] = []
                st.rerun()
        with col_b:
            if st.button("Retry", use_container_width=True, key="btn_retry", type="secondary"):
                get_enrollment_session.clear()
                st.session_state.saved = False
                with _lock:
                    _shared["progress"] = 0.0
                    _shared["status"] = ""
                    _shared["buckets_captured"] = []
                st.rerun()
