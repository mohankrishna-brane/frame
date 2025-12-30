import sys
import os
import cv2
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.vision_engine import FaceEngine
from core.storage import get_storage_engine
from core.tracker import SimpleTracker
from logic.attendance import RecognitionSystem
from config.settings import FACE_MODEL_LITE

# --- 1. REALSENSE LIBRARY CHECK ---
REALSENSE_AVAILABLE = False
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
    print("[INIT] Intel RealSense SDK loaded.")
except ImportError:
    print("[INIT] RealSense SDK not found. Will use standard webcam.")

def main():
    print("--- STARTING FRAME ENGINE ---")
    
    # 2. Initialize Components
    # Ensure core/vision_engine.py uses 'buffalo_s' for speed
    engine = FaceEngine(model_name=FACE_MODEL_LITE) 
    storage = get_storage_engine("filesystem")
    tracker = SimpleTracker(max_lost=20)
    system = RecognitionSystem(storage, threshold=0.5)
    
    print(f"Loaded {len(system.db)} identities.")

    # 3. Setup Camera Strategy (RealSense -> Fallback to OpenCV)
    W, H = 1280, 720 # HD Capture
    pipeline = None
    cap = None
    camera_source = "opencv" # Default starting state

    if REALSENSE_AVAILABLE:
        try:
            print("[INIT] Connecting to RealSense Camera...")
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
            
            pipeline.start(config)
            camera_source = "realsense"
            print("[SUCCESS] RealSense Connected!")
            
            # Warmup for auto-exposure
            for _ in range(10): pipeline.wait_for_frames()

        except RuntimeError as e:
            print(f"[ERROR] RealSense failed to start: {e}")
            print("Falling back to OpenCV Webcam.")
            camera_source = "opencv"

    # Fallback / Standard Webcam Setup
    if camera_source == "opencv":
        print("[INIT] Opening Standard Webcam (Index 0)...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[CRITICAL] No camera found. Exiting.")
            return
        cap.set(3, W)
        cap.set(4, H)
    
    # Attendance Log (Session-based)
    marked_attendance = set()
    
    # FPS Calculation
    prev_frame_time = 0
    new_frame_time = 0

    try:
        while True:
            # --- 4. Frame Capture Logic ---
            frame = None
            
            if camera_source == "realsense":
                try:
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame: continue
                    frame = np.asanyarray(color_frame.get_data())
                except RuntimeError:
                    print("[ERROR] RealSense stream disconnected.")
                    break
            else:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1) # Mirror effect for webcam

            if frame is None: continue

            # --- 5. OPTIMIZATION: Process on small frame, Draw on big frame ---
            # Resize to 640px width for AI (Approx 4x faster than 720p)
            process_scale = 0.5 
            small_frame = cv2.resize(frame, (0, 0), fx=process_scale, fy=process_scale)
            
            # A. Detection (Run on small frame)
            small_faces = engine.process_frame(small_frame)
            
            # B. Scaling & Prep for Tracker
            rects = []
            for face in small_faces:
                # Scale box back up to original resolution
                box = face.bbox.astype(int)
                box = (box / process_scale).astype(int)
                
                # Update face object with scaled box for later drawing
                face.bbox = box.astype(float) 
                rects.append(box.tolist())

            # C. Tracking
            tracked_objects = tracker.update(rects)
            
            # D. Identification & UI
            for track_id, box in tracked_objects.items():
                # Match track_id to the closest AI face object (using IOU logic)
                matched_face = None
                best_iou = 0
                
                for face in small_faces:
                    # Use the scaled-up box from the face object
                    iou = tracker._iou(box, face.bbox.astype(int))
                    if iou > 0.4: # Threshold for association
                        matched_face = face
                        break
                
                # Default UI values
                name = "Scanning..."
                color = (0, 165, 255) # Orange
                score_txt = ""

                # If we found the face data for this track, identify it
                if matched_face:
                    emp_id, name_found, score = system.identify(track_id, matched_face.embedding)
                    
                    if name_found != "Unknown":
                        name = name_found
                        color = (0, 255, 0) # Green
                        score_txt = f"{int(score*100)}%"
                        
                        # Log Attendance
                        if emp_id not in marked_attendance:
                            timestamp = time.strftime('%H:%M:%S')
                            print(f"[ACCESS GRANTED] {name} ({emp_id}) at {timestamp}")
                            marked_attendance.add(emp_id)
                    else:
                        color = (0, 0, 255) # Red (Unknown person)

                # Draw Logic
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label background
                label = f"{name} {score_txt}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # FPS Display
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("FRAME Access Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("[EXIT] Releasing resources...")
        if camera_source == "realsense" and pipeline:
            pipeline.stop()
        elif cap:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()