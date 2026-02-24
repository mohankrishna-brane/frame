
import sys
import os
import cv2
import numpy as np

# --- CRITICAL FIX: Add current directory to path ---
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.vision_engine import FaceEngine
from core.storage import get_storage_engine
from logic.enrollment import EnrollmentSession
from ui.hud import RadarHUD

# --- Intel RealSense Support ---
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

def main():
    print("--- FRAME ENROLLMENT SYSTEM ---")
    print("Initializing AI...")
    
    # 1. Setup Components
    engine = FaceEngine()
    db = get_storage_engine("filesystem")
    session = EnrollmentSession()
    
    # 2. Setup Camera (Intel RealSense if available, else fallback to OpenCV)
    W, H = 1280, 720
    pipeline = None
    cap = None
    camera_source = None

    if REALSENSE_AVAILABLE:
        try:
            print("Attempting to use Intel RealSense Camera...")
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
            pipeline.start(config)
            camera_source = "realsense"
            print("Intel RealSense Camera started successfully.")
        except Exception as e:
            print(f"RealSense camera not available: {e}")
            print("Falling back to OpenCV Camera...")
            pipeline = None

    if camera_source is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open any camera.")
            return
        cap.set(3, W)
        cap.set(4, H)
        camera_source = "opencv"
        print("Using OpenCV Camera.")
    hud = RadarHUD(W, H)
    
    print("Camera Started. Press 'q' to quit.")

    try:
        while True:
            # --- Frame Capture ---
            if camera_source == "realsense":
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1) # Mirror effect

            display = frame.copy()

            # --- Vision ---
            faces = engine.process_frame(frame)

            if len(faces) == 1:
                face = faces[0]
                pitch, yaw, roll = engine.compute_pose(face)
                
                # Extract Face Crop (For Cold Storage)
                b = face.bbox.astype(int)
                h_img, w_img, _ = frame.shape
                x1, y1 = max(0, b[0]), max(0, b[1])
                x2, y2 = min(w_img, b[2]), min(h_img, b[3])
                face_crop = frame[y1:y2, x1:x2].copy()

                # --- State Machine ---
                if session.state == "PRE_CHECK":
                    is_ready, msg = session.run_pre_check(face, pitch, yaw)
                    hud.draw_guide_box(display, is_ready, msg)

                elif session.state == "CAPTURING":
                    status, _ = session.process_face_capture(face, pitch, yaw, face_crop)
                    
                    # UI
                    progress, buckets = session.get_progress()
                    hud.draw_radar(display, pitch, yaw, buckets)
                    cv2.putText(display, f"Capturing: {int(progress*100)}%", (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(display, status, (20, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                elif session.state == "COMPLETE":
                    hud.draw_radar(display, pitch, yaw, session.buckets)
                    cv2.putText(display, "COMPLETE!", (W//2 - 140, H//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    cv2.putText(display, "Press 'S' to Save", (W//2 - 120, H//2 + 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            else:
                hud.draw_guide_box(display, False, "Show 1 Face Only")

            cv2.imshow("FRAME Enrollment", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s') and session.state == "COMPLETE":
                # Save Trigger
                print("\n--- Finalizing Data ---")
                emp_name = input("Enter Employee Name: ")
                emp_id = input("Enter Employee ID: ")
                
                _, buckets = session.get_progress()
                db.save_identity(emp_id, emp_name, buckets)
                print("Successfully Saved! Exiting...")
                break
    finally:
        if pipeline is not None:
            pipeline.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()