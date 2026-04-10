"""
gesture_classifier_ml.py
────────────────────────
Drop-in replacement for the rule-based gesture_classifier.py.

Instead of hand-crafted angle/distance rules, this module loads the
trained PyTorch MLP checkpoint and classifies landmarks via the model.

The public interface is identical to the original:
    classify_mudra(lm)  →  str  (mudra name or "Unknown")

Usage:
    from gesture_classifier_ml import classify_mudra, load_model

    # Load once at startup (not per frame)
    load_model("mudra_model.pt")

    # Then call per frame exactly as before
    name = classify_mudra(results.multi_hand_landmarks[0].landmark)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

# ── Model Definition (must match train_model.py exactly) ─────────────────────

class MudraMLPClassifier(nn.Module):
    def __init__(self, input_dim: int = 63, num_classes: int = 6,
                 hidden_dims: tuple = (256, 128, 64), dropout: float = 0.4):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Module-level state ────────────────────────────────────────────────────────

_model: Optional[MudraMLPClassifier] = None
_classes: list[str] = []
_device  = torch.device("cpu")
_CONFIDENCE_THRESHOLD = 0.50   # minimum softmax probability to accept a prediction


# ── Public API ────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = "cpu") -> None:
    """
    Load the trained model checkpoint.  Call once at application startup.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pt file saved by train_model.py.
    device : str
        "cpu" or "cuda" (default: "cpu" — safe for laptops).
    """
    global _model, _classes, _device

    _device = torch.device(device)
    ckpt    = torch.load(checkpoint_path, map_location=_device)

    _classes = list(ckpt["classes"])
    num_classes = ckpt["num_classes"]
    input_dim   = ckpt["input_dim"]

    _model = MudraMLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
    ).to(_device)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()

    print(f"[ML Classifier] Loaded '{checkpoint_path}'  "
          f"→  {num_classes} classes: {_classes}  (val_acc={ckpt.get('val_acc', '?'):.3f})")


def classify_mudra(landmarks) -> str:
    """
    Classify a MediaPipe hand landmark object.

    Parameters
    ----------
    landmarks : NormalizedLandmarkList  (mediapipe result)
        Typically:  results.multi_hand_landmarks[0].landmark

    Returns
    -------
    str
        One of the trained class names, or "Unknown" if confidence is low
        or the model hasn't been loaded yet.
    """
    if _model is None:
        # Graceful degradation: warn once and return Unknown
        return "Unknown"

    if landmarks is None:
        return "Unknown"

    feats = _normalize_landmarks(landmarks)
    if feats is None:
        return "Unknown"

    with torch.no_grad():
        x      = torch.from_numpy(feats).unsqueeze(0).to(_device)   # (1, 63)
        logits = _model(x)                                           # (1, C)
        probs  = torch.softmax(logits, dim=1).squeeze(0)             # (C,)

    confidence, pred_idx = probs.max(0)

    if confidence.item() < _CONFIDENCE_THRESHOLD:
        return "Unknown"

    return _classes[pred_idx.item()]


def classify_mudra_with_probs(landmarks) -> tuple[str, dict]:
    """
    Like classify_mudra but also returns the full probability distribution.

    Returns
    -------
    (mudra_name, {class_name: probability, ...})
    """
    if _model is None or landmarks is None:
        return "Unknown", {}

    feats = _normalize_landmarks(landmarks)
    if feats is None:
        return "Unknown", {}

    with torch.no_grad():
        x      = torch.from_numpy(feats).unsqueeze(0).to(_device)
        logits = _model(x)
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    prob_dict = {cls: float(p) for cls, p in zip(_classes, probs)}
    pred_cls  = max(prob_dict, key=prob_dict.get)

    if prob_dict[pred_cls] < _CONFIDENCE_THRESHOLD:
        pred_cls = "Unknown"

    return pred_cls, prob_dict


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalize_landmarks(landmarks) -> Optional[np.ndarray]:
    """
    Convert MediaPipe NormalizedLandmarkList → normalised (63,) numpy array.

    Normalisation
    ─────────────
    • Translate so wrist (lm[0]) is at origin.
    • Scale by wrist→middle-MCP (lm[9]) distance  →  size-invariant.
    • Flatten to 1-D.
    """
    try:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    except Exception:
        return None

    if pts.shape[0] != 21:
        return None

    pts -= pts[0]                                         # wrist at origin
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts  /= scale                                         # scale-invariant

    return pts.flatten()                                  # (63,)
