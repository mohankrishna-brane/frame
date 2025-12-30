import cv2

class RadarHUD:
    def __init__(self, width, height):
        self.W, self.H = width, height
        self.cx, self.cy = width // 2, height // 2
        self.scale = 4.0 

        # Guide Box Coordinates
        box_size = 280
        self.guide_box = (
            self.cx - box_size // 2, self.cy - box_size // 2, 
            self.cx + box_size // 2, self.cy + box_size // 2
        )

    def draw_guide_box(self, frame, is_ready=False, message="Align Face"):
        color = (0, 255, 0) if is_ready else (0, 0, 255)
        thickness = 3
        x1, y1, x2, y2 = self.guide_box
        
        # Draw "bracket" style corners
        L = 40
        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + L, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + L), color, thickness)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - L, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + L), color, thickness)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + L, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - L), color, thickness)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - L, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - L), color, thickness)

        cv2.putText(frame, message, (self.cx - 120, y1 - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    def draw_radar(self, frame, pitch, yaw, buckets):
        # 1. Draw Buckets (Targets)
        for name, data in buckets.items():
            tx = self.cx + int(data["yaw"] * self.scale)
            ty = self.cy - int(data["pitch"] * self.scale)
            
            color = (0, 255, 0) if data["captured"] else (0, 140, 255)
            filled = -1 if data["captured"] else 2
            
            cv2.circle(frame, (tx, ty), 12, color, filled)
            if not data["captured"]:
                cv2.circle(frame, (tx, ty), 4, color, -1)

        # 2. Draw User Cursor
        cursor_x = self.cx + int(yaw * self.scale)
        cursor_y = self.cy - int(pitch * self.scale)
        
        # Crosshair
        cv2.line(frame, (cursor_x - 15, cursor_y), (cursor_x + 15, cursor_y), (0, 255, 255), 2)
        cv2.line(frame, (cursor_x, cursor_y - 15), (cursor_x, cursor_y + 15), (0, 255, 255), 2)
        
        # Tether line
        cv2.line(frame, (self.cx, self.cy), (cursor_x, cursor_y), (255, 255, 255), 1)

        return frame