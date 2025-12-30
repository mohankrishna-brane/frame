import numpy as np
import cv2
from insightface.app import FaceAnalysis
from config.settings import FACE_MODEL

class FaceEngine:
    def __init__(self, model_name=FACE_MODEL, ctx_id=0, det_size=(640, 640)):
        # Initialize InsightFace
        # providers=['CUDAExecutionProvider'] if you have NVIDIA GPU, else CPU
        self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def process_frame(self, frame):
        """
        Returns a list of Face objects found in the frame.
        """
        faces = self.app.get(frame)
        return faces

    @staticmethod
    def compute_pose(face):
        """
        Returns (pitch, yaw, roll) in degrees.
        """
        return face.pose

    @staticmethod
    def get_norm_embedding(face):
        """
        Returns L2 Normalized embedding for Cosine Similarity.
        """
        embedding = face.embedding
        norm = np.linalg.norm(embedding)
        return embedding / norm