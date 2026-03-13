"""
Retroactively run augmentation on all existing enrolled identities.
Loads saved face crops from data/raw_images/ and regenerates all
augmented vectors (replaces any previous augmentation).

Usage:
    python reaugment_all.py
"""
import os
import cv2
import numpy as np
from core.vision_engine import FaceEngine
from core.storage import get_storage_engine
from logic.augmentation import (
    _get_rec_model, _augment_angle, AUGMENTATION_ENABLED
)

ANGLE_NAMES = ["center", "look_up", "look_down", "left_semi", "right_semi", "left_full", "right_full"]


def augment_identity(emp_id, engine, storage):
    vec_path = os.path.join(storage.paths["vectors"], f"{emp_id}.npy")
    if not os.path.exists(vec_path):
        print(f"[{emp_id}] No vector file, skipping")
        return

    # Keep only original 7 angle vectors, drop all augmented keys
    full_pack = np.load(vec_path, allow_pickle=True).item()
    vector_pack = {k: v for k, v in full_pack.items() if k in ANGLE_NAMES}

    rec_model = _get_rec_model(engine)
    img_dir   = os.path.join(storage.paths["images"], emp_id)
    total_added = 0

    for angle in ANGLE_NAMES:
        if angle not in vector_pack:
            continue
        img_path = os.path.join(img_dir, f"{angle}.jpg")
        if not os.path.exists(img_path):
            print(f"[{emp_id}] No image for {angle}, skipping")
            continue
        crop = cv2.imread(img_path)
        vecs = _augment_angle(rec_model, angle, crop, img_save_dir=img_dir)
        vector_pack.update(vecs)
        total_added += len(vecs)

    np.save(vec_path, vector_pack)
    print(f"[{emp_id}] Done: +{total_added} augmented vectors (total: {len(vector_pack)})")


def main():
    if not AUGMENTATION_ENABLED:
        print("AUGMENTATION_ENABLED is False in logic/augmentation.py — aborting.")
        return

    storage = get_storage_engine()
    engine  = FaceEngine()

    vec_dir = storage.paths["vectors"]
    ids = [f.replace(".npy", "") for f in os.listdir(vec_dir) if f.endswith(".npy")]
    print(f"Found {len(ids)} enrolled identities\n")

    for emp_id in sorted(ids):
        augment_identity(emp_id, engine, storage)

    print("\nAll done.")


if __name__ == "__main__":
    main()
