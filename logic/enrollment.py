import numpy as np


class EnrollmentSession:
    CANDIDATES_NEEDED = 5

    def __init__(self):
        self.state = "PRE_CHECK"
        self.pre_check_counter = 0
        self.QUALITY_THRESHOLD = 0.6
        self.CENTER_TOLERANCE = 10
        self.bucket_tolerance = 12

        self.buckets = {
            "center":      {"yaw": 0,   "pitch": 0,   "captured": False, "vector": None, "image": None, "candidates": []},
            "look_up":     {"yaw": 0,   "pitch": 25,  "captured": False, "vector": None, "image": None, "candidates": []},
            "look_down":   {"yaw": 0,   "pitch": -25, "captured": False, "vector": None, "image": None, "candidates": []},
            "left_semi":   {"yaw": 25,  "pitch": 0,   "captured": False, "vector": None, "image": None, "candidates": []},
            "right_semi":  {"yaw": -25, "pitch": 0,   "captured": False, "vector": None, "image": None, "candidates": []},
            "left_full":   {"yaw": 45,  "pitch": 0,   "captured": False, "vector": None, "image": None, "candidates": []},
            "right_full":  {"yaw": -45, "pitch": 0,   "captured": False, "vector": None, "image": None, "candidates": []},
        }

    @staticmethod
    def _normalize(vec):
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def run_pre_check(self, face, pitch, yaw):
        if face.det_score < self.QUALITY_THRESHOLD:
            self.pre_check_counter = 0
            return False, "Low Quality. Check Lighting."

        if abs(pitch) > 10 or abs(yaw) > 10:
            self.pre_check_counter = 0
            return False, "Look Strictly Straight."

        self.pre_check_counter += 1
        if self.pre_check_counter > 10:
            self.state = "CAPTURING"
            return True, "Perfect! Move your head..."

        return False, f"Hold Steady... ({self.pre_check_counter}/10)"

    def check_dead_zones(self, pitch, yaw):
        if -20 < pitch < -10:
            return "Tilt Down MORE"
        if 10 < pitch < 20:
            return "Tilt Up MORE"
        return None

    def process_face_capture(self, face, pitch, yaw, face_image):
        if face.det_score < self.QUALITY_THRESHOLD:
            return "Low Quality Frame", False

        warning = self.check_dead_zones(pitch, yaw)
        if warning:
            return warning, False

        for name, data in self.buckets.items():
            if data["captured"]:
                continue

            d_yaw   = abs(yaw   - data["yaw"])
            d_pitch = abs(pitch - data["pitch"])

            yaw_tol   = 15
            pitch_tol = 12

            if name == "center":
                yaw_tol   = 8
                pitch_tol = 8

            if d_yaw < yaw_tol and d_pitch < pitch_tol:
                data["candidates"].append((face.det_score, face.embedding.copy(), face_image))
                n = len(data["candidates"])

                if n >= self.CANDIDATES_NEEDED:
                    best = max(data["candidates"], key=lambda x: x[0])
                    data["captured"]    = True
                    data["vector"]      = self._normalize(best[1])
                    data["image"]       = best[2]
                    data["candidates"]  = []

                    if self.get_progress()[0] == 1.0:
                        self.state = "COMPLETE"

                    return f"Captured: {name}", True

                return f"Hold for {name}... ({n}/{self.CANDIDATES_NEEDED})", False

        return "Find the target angles...", False

    def get_progress(self):
        total    = len(self.buckets)
        captured = sum(1 for b in self.buckets.values() if b["captured"])
        return captured / total, self.buckets