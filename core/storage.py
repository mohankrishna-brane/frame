import os
import json
import numpy as np
import cv2
from datetime import datetime
from abc import ABC, abstractmethod


# --- THE INTERFACE ---
class BaseStorageAdapter(ABC):
    @abstractmethod
    def save_identity(self, emp_id, name, bucket_data):
        pass

    @abstractmethod
    def load_identities(self):
        pass


# --- FILE SYSTEM IMPLEMENTATION ---
class FileSystemAdapter(BaseStorageAdapter):
    def __init__(self, base_path="data"):
        self.paths = {
            "meta":    os.path.join(base_path, "metadata"),
            "vectors": os.path.join(base_path, "vectors"),
            "images":  os.path.join(base_path, "raw_images"),  # Cold Storage
        }
        for p in self.paths.values():
            os.makedirs(p, exist_ok=True)

    def save_identity(self, emp_id, name, bucket_data):
        # 1. Save Metadata
        meta = {
            "id":         emp_id,
            "name":       name,
            "created_at": str(datetime.now()),
            "model":      "buffalo_l",
        }
        with open(os.path.join(self.paths["meta"], f"{emp_id}.json"), "w") as f:
            json.dump(meta, f, indent=4)

        # 2. Process Vectors & Images
        vector_pack  = {}
        emp_img_path = os.path.join(self.paths["images"], emp_id)
        os.makedirs(emp_img_path, exist_ok=True)

        for angle, data in bucket_data.items():
            if data["captured"]:
                # Vector is already normalized at source in EnrollmentSession
                vector_pack[angle] = data["vector"]

                # Cold Storage: Save Face Crop JPG
                if data["image"] is not None:
                    cv2.imwrite(os.path.join(emp_img_path, f"{angle}.jpg"), data["image"])

        np.save(os.path.join(self.paths["vectors"], f"{emp_id}.npy"), vector_pack)
        print(f"[Storage] Saved {len(vector_pack)} angles for {name}")

    def load_identities(self):
        identities = {}
        if not os.path.exists(self.paths["vectors"]):
            return {}

        for file in os.listdir(self.paths["vectors"]):
            if not file.endswith(".npy"):
                continue
            emp_id = file.replace(".npy", "")
            try:
                vec_data     = np.load(os.path.join(self.paths["vectors"], file), allow_pickle=True).item()
                flat_vectors = list(vec_data.values())

                meta_path = os.path.join(self.paths["meta"], f"{emp_id}.json")
                name = "Unknown"
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        name = json.load(f)["name"]

                identities[emp_id] = {"name": name, "vectors": flat_vectors}
            except Exception as e:
                print(f"Error loading {emp_id}: {e}")
        return identities


# --- FACTORY ---
def get_storage_engine(type="filesystem", **kwargs):
    if type == "filesystem":
        return FileSystemAdapter()
    else:
        raise ValueError("Unknown storage type")