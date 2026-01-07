import sys
import os
import cv2
import time
import numpy as np
import threading

# --- 1. SYSTEM PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from core.vision_engine import FaceEngine
from core.tracker import SimpleTracker
from core.api_client import APIClient
from config.settings import (
    FACE_MODEL_LITE, API_BASE_URL, API_USERNAME, 
    API_PASSWORD, TENANT_ID, GO_ID
)

# --- 2. HARDWARE CHECK ---
REALSENSE_AVAILABLE = False
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
    print("[INIT] Intel RealSense SDK loaded.")
except ImportError:
    print("[INIT] RealSense SDK not found. Will use standard webcam.")

def main():
    print("\n--- STARTING CAPTURE ENGINE (CLOUD MODE) ---")
    
    # --- 3. NETWORK LOGIN ---
    print("[NETWORK] Connecting to Workflow API...")
    api = APIClient(base_url=API_BASE_URL, tenant_id=TENANT_ID)
    is_online = api.login(API_USERNAME, API_PASSWORD)
    
    if not is_online:
        print("[CRITICAL] Login Failed. Cannot proceed without API.")
        return

    # --- 4. INITIALIZE AI CORE ---
    print("[CORE] Loading Vision Engine...")
    engine = FaceEngine(model_name=FACE_MODEL_LITE) 
    tracker = SimpleTracker(max_lost=30) # Increased persistence for smoother tracking
    
    # --- 5. CAMERA SETUP ---
    W, H = 1280, 720
    pipeline = None
    cap = None
    camera_source = "opencv"

    if REALSENSE_AVAILABLE:
        try:
            print("[CAMERA] Attempting RealSense Connection...")
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
            pipeline.start(config)
            camera_source = "realsense"
            print("[CAMERA] RealSense Connected Successfully.")
            for _ in range(15): pipeline.wait_for_frames()
        except RuntimeError:
            print("[CAMERA] Falling back to Standard Webcam.")
            camera_source = "opencv"

    if camera_source == "opencv":
        cap = cv2.VideoCapture(0)
        cap.set(3, W)
        cap.set(4, H)
        if not cap.isOpened(): return

    # --- 6. RUNTIME VARIABLES ---
    # Stores track_ids that have already triggered the API
    triggered_tracks = set() 
    prev_frame_time = 0
    
    print("\n[SYSTEM] ENGINE RUNNING. Press 'q' to exit.\n")

    try:
        while True:
            # A. CAPTURE
            frame = None
            if camera_source == "realsense":
                try:
                    frames = pipeline.wait_for_frames()
                    frame = np.asanyarray(frames.get_color_frame().get_data())
                except: break
            else:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)

            if frame is None: continue

            # B. PROCESSING
            process_scale = 0.5 
            small_frame = cv2.resize(frame, (0, 0), fx=process_scale, fy=process_scale)
            
            # Get faces and vectors
            small_faces = engine.process_frame(small_frame)
            
            # C. TRACKING PREP
            rects = []
            for face in small_faces:
                box = (face.bbox.astype(int) / process_scale).astype(int)
                face.bbox = box.astype(float) 
                rects.append(box.tolist())

            tracked_objects = tracker.update(rects)
            
            # D. LOGIC LOOP
            for track_id, box in tracked_objects.items():
                # Match tracker box to face object (to get the embedding)
                matched_face = None
                for face in small_faces:
                    iou = tracker._iou(box, face.bbox.astype(int))
                    if iou > 0.4:
                        matched_face = face
                        break
                
                # UI Defaults
                color = (0, 255, 0) # Green
                status_text = "Tracking"

                if matched_face:
                    # CHECK: Have we sent this person to the server yet?
                    if track_id not in triggered_tracks:
                        print(f"[API TRIGGER] Processing Track ID {track_id}...")
                        
                        # Mark as processed immediately so we don't spam the API
                        triggered_tracks.add(track_id)
                        status_text = "Verifying..."
                        
                        # Placeholder ID since we don't know who it is yet
                        temp_emp_id = f"Track_{track_id}"
                        
                        # FIRE AND FORGET
                        threading.Thread(
                            target=api.execute_attendance_workflow, 
                            args=(temp_emp_id, GO_ID, matched_face.embedding.tolist())
                        ).start()
                    else:
                        status_text = "Sent"

                # Draw UI
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id} [{status_text}]", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # E. CLEANUP
            # Optional: If a track is lost (person leaves frame), remove from triggered_tracks
            # so they can trigger again if they come back later.
            active_tracks = set(tracked_objects.keys())
            triggered_tracks = triggered_tracks.intersection(active_tracks)

            # F. DISPLAY
            cv2.imshow("FRAME Cloud Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        print("[EXIT] Releasing resources...")
        if camera_source == "realsense" and pipeline: pipeline.stop()
        elif cap: cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()