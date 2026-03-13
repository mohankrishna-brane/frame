# Face Enrollment Augmentation Plan

## Goal

After enrollment captures 7 face crops (one per angle), synthetically expand each person's embedding set from 7 to ~147 vectors — without re-capturing — so the recognition matcher has more data points per person.

---

## Two Augmentation Strategies

### 1. Image Augmentation (10 per angle = 70 total)

Apply photometric transforms to each saved face crop, upscale, re-run through FaceEngine to produce a real new embedding.

**Transforms (via `albumentations`):**
- `RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8)`
- `GaussianBlur(blur_limit=(3,7), p=0.5)`
- `GaussNoise(std_range=(0.01, 0.05), p=0.4)`
- `RandomGamma(gamma_limit=(70,130), p=0.4)`
- `CLAHE(p=0.3)`
- `HorizontalFlip(p=0.5)` — all angles (flipped left-profile = right-profile-like; valid since vectors are matched as a flat list)

**Critical:** Face crops must be **2x upscaled** before passing to `FaceEngine.process_frame()` — insightface cannot detect faces at native crop resolution. If det_score < 0.50, discard and retry (up to 4× per slot).

Vector keys: `{angle}_aug_0` … `{angle}_aug_9`

---

### 2. Vector Gaussian Noise (10 per angle = 70 total)

Add calibrated Gaussian noise directly to each L2-normalized embedding, then re-normalize.

- **sigma = 0.010** → cosine similarity ≈ 0.975 vs original (meaningful variation, safe margin above 0.45 threshold)
- Re-normalize to unit length after adding noise

Vector keys: `{angle}_noise_0` … `{angle}_noise_9`

---

## Resulting Vector Count Per Person

| Source | Count |
|---|---|
| Original (7 angles) | 7 |
| Image augmentation (7 × 10) | 70 |
| Vector noise (7 × 10) | 70 |
| **Total** | **147** |

---

## Implementation

### New file: `logic/augmentation.py`

```
AugmentationPipeline
  __init__(engine, storage)
  run(emp_id, buckets) -> int
    - loads data/vectors/{emp_id}.npy
    - for each captured angle:
        - generates 10 image_aug vectors
        - generates 10 noise vectors
    - re-saves expanded .npy pack
    - returns total augmented count
```

### `enroll_ui.py` — after line 504 (`get_db().save_identity(...)`)

```python
from logic.augmentation import AugmentationPipeline
aug = AugmentationPipeline(engine=get_engine(), storage=get_db())
aug.run(emp_id, buckets)
```

### `enroll.py` — after line 131 (`db.save_identity(...)`)

```python
from logic.augmentation import AugmentationPipeline
aug = AugmentationPipeline(engine=engine, storage=db)
aug.run(emp_id, buckets)
```

---

## Files NOT Needing Changes

- `core/storage.py` — `load_identities()` already flattens all `.npy` dict values; new keys auto-included
- `logic/attendance.py` — matching loop already handles variable vector count per person
- `core/vision_engine.py` — interface unchanged; just pass 2x upscaled crop

---

## Augmented vectors → local only

Augmented vectors are saved to `data/vectors/{emp_id}.npy` only. They are **not** pushed to Postgres — that table is an audit trail with structured angle metadata; synthetic vectors would pollute it.

---

## Verification

```python
import numpy as np
d = np.load("data/vectors/EMP001.npy", allow_pickle=True).item()
print(len(d))          # expect ~147
print(list(d.keys()))  # center, center_aug_0..9, center_noise_0..9, look_up, ...
```

Then run `main.py` and confirm recognition similarity is equal or better.
