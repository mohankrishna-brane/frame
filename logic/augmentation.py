import os
import cv2
import json
import uuid
import threading
import numpy as np
import psycopg2
import albumentations as A

DB_CONFIG = {
    "dbname":   "workflow_system",
    "user":     "postgres",
    "password": "9ets0n1234",
    "host":     "10.26.1.175",
    "port":     "5432",
}

# --- Toggle augmentation on/off ---
AUGMENTATION_ENABLED = True

REC_INPUT_SIZE = (112, 112)  # ArcFace recognition model input

# Face region anchors (relative fractions) — tuned for insightface-cropped faces
_EYE_Y       = 0.38
_MASK_TOP_Y  = 0.52
_CAP_BOT_Y   = 0.28


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm < 1e-6 else vec / norm


def _get_rec_model(engine):
    return engine.app.models['recognition']


def _embed_crop(rec_model, crop: np.ndarray) -> np.ndarray:
    resized = cv2.resize(crop, REC_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    return _normalize(rec_model.get_feat(resized).flatten())


def _overlay_blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Occlusion overlays
# ---------------------------------------------------------------------------

def apply_glasses(crop: np.ndarray) -> np.ndarray:
    """Solid black or white bar over the eye region."""
    out = crop.copy()
    h, w = out.shape[:2]
    color = (0, 0, 0)   # black — more common in real occlusion scenarios
    ey = int(_EYE_Y * h)
    pad_y = int(0.12 * h)
    pad_x = int(0.06 * w)
    overlay = out.copy()
    cv2.rectangle(overlay, (pad_x, ey - pad_y), (w - pad_x, ey + pad_y), color, -1)
    return _overlay_blend(out, overlay, 0.92)


def apply_mask(crop: np.ndarray) -> np.ndarray:
    """Solid patch covering nose-to-chin."""
    out = crop.copy()
    h, w = out.shape[:2]
    top_y = int(_MASK_TOP_Y * h)
    overlay = out.copy()
    cv2.rectangle(overlay, (int(0.04 * w), top_y), (w - int(0.04 * w), h), (200, 200, 200), -1)
    return _overlay_blend(out, overlay, 0.90)


def apply_cap(crop: np.ndarray) -> np.ndarray:
    """Dark bar covering the forehead/top."""
    out = crop.copy()
    h, w = out.shape[:2]
    brim_y = int(_CAP_BOT_Y * h)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, brim_y), (20, 20, 20), -1)
    return _overlay_blend(out, overlay, 0.92)


def apply_low_brightness(crop: np.ndarray) -> np.ndarray:
    return np.clip(crop.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)


def apply_high_brightness(crop: np.ndarray) -> np.ndarray:
    return np.clip(crop.astype(np.float32) * 1.8 + 30, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Augmentation variants per crop
# ---------------------------------------------------------------------------

# Each entry: (key_suffix, transform_fn)
_VARIANTS = [
    ("glasses",          lambda c: apply_glasses(c)),
    ("mask",             lambda c: apply_mask(c)),
    ("cap",              lambda c: apply_cap(c)),
    ("glasses_mask",     lambda c: apply_mask(apply_glasses(c))),
    ("glasses_cap",      lambda c: apply_cap(apply_glasses(c))),
    ("mask_cap",         lambda c: apply_cap(apply_mask(c))),
    ("low_brightness",   lambda c: apply_low_brightness(c)),
    ("high_brightness",  lambda c: apply_high_brightness(c)),
]


def _augment_angle(rec_model, angle: str, crop: np.ndarray, img_save_dir: str = None) -> dict:
    results = {}
    flipped_crop = cv2.flip(crop, 1)
    for suffix, fn in _VARIANTS:
        for img, tag in [(crop, suffix), (flipped_crop, f"{suffix}_flip")]:
            try:
                augmented = fn(img)
                vec = _embed_crop(rec_model, augmented)
                key = f"{angle}_{tag}"
                results[key] = vec
                if img_save_dir:
                    cv2.imwrite(os.path.join(img_save_dir, f"{key}.jpg"), augmented)
            except Exception as e:
                print(f"[Augmentation] Failed {angle}_{tag}: {e}")
    return results


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def _save_augmented_to_postgres(emp_id: str, emp_name: str, augmented_vectors: dict):
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor()
        insert_query = """
        INSERT INTO workflow_runtime.iot_face_encodings
        (face_encoding_id, person_type, encoding_vector, encoding_model, encoding_dimension,
         face_location, face_quality_score, face_angle, is_frontal, is_primary,
         tenant_id, is_active, person_id, person_name)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING;
        """
        count = 0
        for key, vector in augmented_vectors.items():
            vector_list = vector.tolist() if hasattr(vector, "tolist") else vector
            cur.execute(insert_query, (
                str(uuid.uuid4()),
                "employee",
                vector_list,
                "insightface",
                len(vector_list),
                None,
                None,
                json.dumps({"angle": key, "type": "augmented"}),
                False,
                False,
                "T689",
                True,
                emp_id,
                emp_name,
            ))
            count += 1
        conn.commit()
        print(f"[Augmentation] Pushed {count} augmented vectors to Postgres for {emp_id}")
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[Augmentation] DB Error: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()


def _run(emp_id: str, buckets: dict, engine, storage):
    vec_path = os.path.join(storage.paths["vectors"], f"{emp_id}.npy")
    try:
        vector_pack = np.load(vec_path, allow_pickle=True).item()
    except Exception as e:
        print(f"[Augmentation] Could not load vector pack for {emp_id}: {e}")
        return

    try:
        meta_path = os.path.join(storage.paths["meta"], f"{emp_id}.json")
        with open(meta_path) as f:
            emp_name = json.load(f)["name"]
    except Exception as e:
        print(f"[Augmentation] Could not load metadata for {emp_id}: {e}")
        emp_name = emp_id

    rec_model     = _get_rec_model(engine)
    img_dir       = os.path.join(storage.paths["images"], emp_id)
    os.makedirs(img_dir, exist_ok=True)
    total_added   = 0
    all_augmented = {}

    for angle, data in buckets.items():
        if not data.get("captured") or data["image"] is None:
            continue
        vecs = _augment_angle(rec_model, angle, data["image"], img_save_dir=img_dir)
        vector_pack.update(vecs)
        all_augmented.update(vecs)
        total_added += len(vecs)

    np.save(vec_path, vector_pack)
    print(f"[Augmentation] Done: +{total_added} vectors for {emp_id} (total: {len(vector_pack)})")

    _save_augmented_to_postgres(emp_id, emp_name, all_augmented)


def run_augmentation(emp_id: str, buckets: dict, engine, storage):
    """
    Launch augmentation in a background thread. Returns immediately.
    Toggle with AUGMENTATION_ENABLED at the top of this file.
    """
    if not AUGMENTATION_ENABLED:
        return
    t = threading.Thread(target=_run, args=(emp_id, buckets, engine, storage), daemon=True)
    t.start()
