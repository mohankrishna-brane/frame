import os
import json
import numpy as np
import cv2
from datetime import datetime
from abc import ABC, abstractmethod

# import psycopg2
# import psycopg2.extras

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
            "meta": os.path.join(base_path, "metadata"),
            "vectors": os.path.join(base_path, "vectors"),
            "images": os.path.join(base_path, "raw_images") # Cold Storage
        }
        for p in self.paths.values():
            os.makedirs(p, exist_ok=True)

    def save_identity(self, emp_id, name, bucket_data):
        # 1. Save Metadata
        meta = {
            "id": emp_id,
            "name": name,
            "created_at": str(datetime.now()),
            "model": "buffalo_l"
        }
        with open(os.path.join(self.paths["meta"], f"{emp_id}.json"), 'w') as f:
            json.dump(meta, f, indent=4)

        # 2. Process Vectors & Images
        vector_pack = {}
        
        # Create folder for raw images: data/raw_images/EMP001/
        emp_img_path = os.path.join(self.paths["images"], emp_id)
        os.makedirs(emp_img_path, exist_ok=True)

        for angle, data in bucket_data.items():
            if data["captured"]:
                # A. Hot Storage: Save Normalized Vector
                vec = data["vector"]
                vec = vec / np.linalg.norm(vec)
                vector_pack[angle] = vec

                # B. Cold Storage: Save Face Crop JPG
                if data["image"] is not None:
                    img_name = f"{angle}.jpg"
                    cv2.imwrite(os.path.join(emp_img_path, img_name), data["image"])

        # Save all vectors to one .npy file
        np.save(os.path.join(self.paths["vectors"], f"{emp_id}.npy"), vector_pack)
        print(f"[Storage] Saved {len(vector_pack)} angles for {name}")

    def load_identities(self):
        identities = {}
        if not os.path.exists(self.paths["vectors"]): return {}

        for file in os.listdir(self.paths["vectors"]):
            if not file.endswith('.npy'): continue
            emp_id = file.replace(".npy", "")
            
            try:
                # Load Vectors
                vec_data = np.load(os.path.join(self.paths["vectors"], file), allow_pickle=True).item()
                flat_vectors = list(vec_data.values())

                # Load Name
                meta_path = os.path.join(self.paths["meta"], f"{emp_id}.json")
                name = "Unknown"
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        name = json.load(f)["name"]

                identities[emp_id] = {"name": name, "vectors": flat_vectors}
            except Exception as e:
                print(f"Error loading {emp_id}: {e}")
        return identities
    
# class PostgresAdapter(BaseStorageAdapter):
#     def __init__(self, dbname, user, password, host='localhost', port=5432):
#         self.conn = psycopg2.connect(
#             dbname=dbname, user=user, password=password, host=host, port=port
#         )
#         self._ensure_tables()

#     def _ensure_tables(self):
#         with self.conn.cursor() as cur:
#             cur.execute("""
#             CREATE TABLE IF NOT EXISTS identities (
#                 id TEXT PRIMARY KEY,
#                 name TEXT,
#                 created_at TIMESTAMP,
#                 model TEXT
#             );
#             """)
#             cur.execute("""
#             CREATE TABLE IF NOT EXISTS vectors (
#                 emp_id TEXT REFERENCES identities(id),
#                 angle TEXT,
#                 vector FLOAT8[],
#                 PRIMARY KEY (emp_id, angle)
#             );
#             """)
#             self.conn.commit()

#     def save_identity(self, emp_id, name, bucket_data):
#         with self.conn.cursor() as cur:
#             # 1. Save Metadata
#             cur.execute("""
#                 INSERT INTO identities (id, name, created_at, model)
#                 VALUES (%s, %s, %s, %s)
#                 ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name;
#             """, (emp_id, name, datetime.now(), "buffalo_l"))

#             # 2. Save Vectors
#             for angle, data in bucket_data.items():
#                 if data["captured"]:
#                     vec = data["vector"]
#                     vec = vec / np.linalg.norm(vec)
#                     cur.execute("""
#                         INSERT INTO vectors (emp_id, angle, vector)
#                         VALUES (%s, %s, %s)
#                         ON CONFLICT (emp_id, angle) DO UPDATE SET vector=EXCLUDED.vector;
#                     """, (emp_id, angle, vec.tolist()))
#             self.conn.commit()
#         print(f"[PostgresStorage] Saved {emp_id} ({name})")

#     def load_identities(self):
#         identities = {}
#         with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
#             cur.execute("SELECT * FROM identities;")
#             for row in cur.fetchall():
#                 emp_id = row['id']
#                 name = row['name']
#                 cur.execute("SELECT vector FROM vectors WHERE emp_id=%s;", (emp_id,))
#                 vectors = [np.array(v['vector']) for v in cur.fetchall()]
#                 identities[emp_id] = {"name": name, "vectors": vectors}
#         return identities


# --- FACTORY ---
def get_storage_engine(type="filesystem", **kwargs):
    if type == "filesystem":
        return FileSystemAdapter()
    elif type == "postgres":
        return PostgresAdapter(**kwargs)
    else:
        raise ValueError("Unknown storage type")