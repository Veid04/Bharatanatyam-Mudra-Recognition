"""
prepare_dataset.py
──────────────────
Walks the Kaggle Bharatanatyam Mudra dataset, runs MediaPipe Hands on every
image, and saves a compact .npz file of (landmarks, labels) ready for training.

Expected dataset layout (from kagglehub download):
    <root>/
        Pataka/   image1.jpg  image2.jpg ...
        Tripataka/...
        Fist/...
        Ardhachandra/...
        Hamsasya/...
        Alapadma/...
        ...

Usage:
    python prepare_dataset.py --dataset_root <path_from_kagglehub> \
                              --output landmarks_dataset.npz
"""

import argparse
import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

# ── Configuration ────────────────────────────────────────────────────────────

# These are the classes we care about.  Names must match folder names exactly
# (case-sensitive on Linux).  Add / remove as needed.
MUDRA_CLASSES = [
    "Pathaka(1)",
    "Tripathaka(1)",
    "Mushti(1)",
    "Ardhachandran(1)",
    "Hamsasyam(1)",
    "Alapadmam(1)",
]

# MediaPipe produces 21 landmarks × 3 (x, y, z) = 63 raw features
NUM_LANDMARKS = 21
FEATURE_DIM   = NUM_LANDMARKS * 3   # 63

# ── Helpers ──────────────────────────────────────────────────────────────────

def normalize_landmarks(landmarks) -> np.ndarray:
    """
    Convert a MediaPipe NormalizedLandmarkList to a flat numpy array and
    apply wrist-origin + scale normalisation so the model is translation-
    and size-invariant.

    Steps
    -----
    1. Stack raw (x, y, z) into shape (21, 3).
    2. Subtract wrist (landmark 0) → translation-invariant.
    3. Divide by the distance from wrist to middle-finger-MCP (landmark 9)
       → scale-invariant.
    4. Flatten to (63,).
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)  # (21, 3)

    # 1. Translate so wrist is at origin
    pts -= pts[0]

    # 2. Scale by wrist→middle-MCP distance
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale

    return pts.flatten()   # (63,)


def extract_from_image(image_path: str, hands) -> np.ndarray | None:
    """
    Run MediaPipe on a single image file.
    Returns a (63,) feature vector or None if no hand is detected.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result  = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    # Take the first (most confident) detected hand
    return normalize_landmarks(result.multi_hand_landmarks[0].landmark)


# ── Main ─────────────────────────────────────────────────────────────────────

def build_dataset(dataset_root: str, output_path: str):
    root = Path(dataset_root)
    if not root.exists():
        print(f"[ERROR] Dataset root not found: {root}")
        sys.exit(1)

    label_map = {name: idx for idx, name in enumerate(MUDRA_CLASSES)}

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,       # single-image mode: more accurate, slower
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
    )

    all_features: list[np.ndarray] = []
    all_labels:   list[int]        = []
    skipped = 0

    for class_name in MUDRA_CLASSES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            # Try case-insensitive search as a fallback
            matches = [d for d in root.iterdir() if d.is_dir() and d.name.lower() == class_name.lower()]
            if matches:
                class_dir = matches[0]
                print(f"[WARN] Using '{class_dir.name}' for class '{class_name}'")
            else:
                print(f"[WARN] Class folder not found: {class_dir} — skipping")
                continue

        image_files = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ])

        print(f"\n[INFO] Processing '{class_name}' ({len(image_files)} images) …")
        label_idx = label_map[class_name]

        for img_path in tqdm(image_files, ncols=80):
            feats = extract_from_image(str(img_path), hands)
            if feats is None:
                skipped += 1
                continue
            all_features.append(feats)
            all_labels.append(label_idx)

    hands.close()

    if not all_features:
        print("[ERROR] No landmarks extracted — check your dataset path and structure.")
        sys.exit(1)

    X = np.stack(all_features, axis=0).astype(np.float32)   # (N, 63)
    y = np.array(all_labels,   dtype=np.int64)               # (N,)

    np.savez_compressed(output_path, X=X, y=y, classes=np.array(MUDRA_CLASSES))

    print(f"\n{'─'*50}")
    print(f"Saved → {output_path}")
    print(f"Total samples  : {len(y)}")
    print(f"Skipped (no hand detected): {skipped}")
    print(f"Class distribution:")
    for idx, name in enumerate(MUDRA_CLASSES):
        count = np.sum(y == idx)
        print(f"  [{idx}] {name:<15} : {count}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmarks from mudra images")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path returned by kagglehub.dataset_download()")
    parser.add_argument("--output", type=str, default="landmarks_dataset.npz",
                        help="Output .npz file path (default: landmarks_dataset.npz)")
    args = parser.parse_args()

    build_dataset(args.dataset_root, args.output)
