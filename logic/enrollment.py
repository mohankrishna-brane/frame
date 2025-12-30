class EnrollmentSession:
    def __init__(self):
        self.state = "PRE_CHECK"
        self.pre_check_counter = 0
        self.QUALITY_THRESHOLD = 0.6
        self.CENTER_TOLERANCE = 10  # Stricter: Was 15, now 10
        self.bucket_tolerance = 12

        # 1. Define Stricter Targets
        # We separate 'center' and 'down' by ensuring center has pitch > -10 
        # and down has pitch < -20. The gap (-10 to -20) is ignored.
        self.buckets = {
            "center":      {"yaw": 0,   "pitch": 0,   "captured": False, "vector": None, "image": None},
            "look_up":     {"yaw": 0,   "pitch": 25,  "captured": False, "vector": None, "image": None}, # Higher target
            "look_down":   {"yaw": 0,   "pitch": -25, "captured": False, "vector": None, "image": None}, # Lower target
            "left_semi":   {"yaw": 25,  "pitch": 0,   "captured": False, "vector": None, "image": None},
            "right_semi":  {"yaw": -25, "pitch": 0,   "captured": False, "vector": None, "image": None},
            "left_full":   {"yaw": 45,  "pitch": 0,   "captured": False, "vector": None, "image": None},
            "right_full":  {"yaw": -45, "pitch": 0,   "captured": False, "vector": None, "image": None},
        }

    def run_pre_check(self, face, pitch, yaw):
        """Stage 1: Ensure user is strictly centered."""
        if face.det_score < self.QUALITY_THRESHOLD:
            self.pre_check_counter = 0
            return False, "Low Quality. Check Lighting."

        # Strict center check for starting
        if abs(pitch) > 10 or abs(yaw) > 10:
            self.pre_check_counter = 0
            return False, "Look Strictly Straight."
        
        self.pre_check_counter += 1
        if self.pre_check_counter > 10:
            self.state = "CAPTURING"
            return True, "Perfect! Move your head..."
        
        return False, f"Hold Steady... ({self.pre_check_counter}/10)"

    def check_dead_zones(self, pitch, yaw):
        """
        Returns a warning message if the user is in an ambiguous angle.
        """
        # Ambiguity between Center and Down
        if -20 < pitch < -10:
            return "Tilt Down MORE"
        
        # Ambiguity between Center and Up
        if 10 < pitch < 20:
            return "Tilt Up MORE"

        return None

    def process_face_capture(self, face, pitch, yaw, face_image):
        """Stage 2: Capture with strict separation."""
        if face.det_score < self.QUALITY_THRESHOLD:
            return "Low Quality Frame", False

        # 1. Check for Dead Zones (Ambiguous angles)
        warning = self.check_dead_zones(pitch, yaw)
        if warning:
            return warning, False

        # 2. Check Buckets
        for name, data in self.buckets.items():
            if data["captured"]: continue

            d_yaw = abs(yaw - data["yaw"])
            d_pitch = abs(pitch - data["pitch"])
            
            # Dynamic Tolerance:
            # We allow more freedom for Yaw (turning left/right) 
            # but stay strict on Pitch (up/down) to prevent the "Center" bleed.
            yaw_tol = 15
            pitch_tol = 12
            
            # Special case: For "Center", be very strict
            if name == "center":
                yaw_tol = 8
                pitch_tol = 8

            if d_yaw < yaw_tol and d_pitch < pitch_tol:
                data["captured"] = True
                data["vector"] = face.embedding
                data["image"] = face_image
                
                if self.get_progress()[0] == 1.0:
                    self.state = "COMPLETE"
                
                return f"Captured: {name}", True
        
        return "Find the target angles...", False

    def get_progress(self):
        total = len(self.buckets)
        captured = sum(1 for b in self.buckets.values() if b["captured"])
        return captured / total, self.buckets