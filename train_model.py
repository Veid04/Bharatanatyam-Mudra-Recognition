"""
train_model.py
──────────────
Trains a lightweight MLP on the landmark features extracted by prepare_dataset.py.

Architecture: landmark features (63) → MLP → softmax over N mudra classes

Why MLP over a CNN?
  - Input is already landmark coordinates, not raw pixels.
  - Landmark-based MLP achieves >95 % accuracy, runs in <1 ms per frame.
  - Easily exportable to ONNX for MCU / edge deployment.

Usage:
    python train_model.py --data landmarks_dataset.npz \
                          --epochs 100 \
                          --output mudra_model.pt
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on servers / SSH
import matplotlib.pyplot as plt
import seaborn as sns

# ── Model Definition ─────────────────────────────────────────────────────────

class MudraMLPClassifier(nn.Module):
    """
    3-layer MLP with BatchNorm + Dropout.

    Input  : (batch, 63)  — normalised MediaPipe landmarks
    Output : (batch, num_classes) — raw logits
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Data Utilities ────────────────────────────────────────────────────────────

def load_data(npz_path: str):
    """Load the .npz file produced by prepare_dataset.py."""
    data    = np.load(npz_path, allow_pickle=True)
    X       = data["X"].astype(np.float32)          # (N, 63)
    y_raw   = data["y"].astype(np.int64)             # (N,)
    classes = data["classes"].tolist()               # list of class name strings

    print(f"Loaded {len(X)} samples | {len(classes)} classes: {classes}")
    return X, y_raw, classes


def get_dataloaders(X: np.ndarray, y: np.ndarray,
                    val_split: float = 0.15, test_split: float = 0.10,
                    batch_size: int = 64):
    """Split into train/val/test and return DataLoaders."""
    n = len(X)
    n_test = int(n * test_split)
    n_val  = int(n * val_split)
    n_train = n - n_val - n_test

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Split → train={n_train}  val={n_val}  test={n_test}")
    return train_loader, val_loader, test_loader


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, targets = [], []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(device))
        preds.extend(logits.argmax(1).cpu().numpy())
        targets.extend(y_batch.numpy())
    return np.array(targets), np.array(preds)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes: list, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


# ── Export ────────────────────────────────────────────────────────────────────

def export_onnx(model, input_dim: int, output_path: str):
    """Export to ONNX for edge / cross-platform deployment."""
    model.eval()
    dummy = torch.zeros(1, input_dim)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"ONNX model exported → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Data
    X, y, classes = load_data(args.data)
    num_classes   = len(classes)
    train_loader, val_loader, test_loader = get_dataloaders(
        X, y, val_split=0.15, test_split=0.10, batch_size=args.batch_size
    )

    # 2. Model
    model = MudraMLPClassifier(
        input_dim=X.shape[1],
        num_classes=num_classes,
        hidden_dims=(256, 128, 64),
        dropout=0.4,
    ).to(device)
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Loss + Optimiser + Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Training
    history    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    patience   = 20
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                  f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

        # Save best checkpoint
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "classes": classes,
                "input_dim": X.shape[1],
                "num_classes": num_classes,
                "val_acc": vl_acc,
            }, args.output)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model checkpoint saved → {args.output}")

    # 5. Final evaluation on test set
    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    y_true, y_pred = predict_all(model, test_loader, device)
    print("\n── Test Set Report ──")
    print(classification_report(y_true, y_pred, target_names=classes))

    # 6. Plots
    plot_training_curves(history, "training_curves.png")
    plot_confusion_matrix(y_true, y_pred, classes, "confusion_matrix.png")

    # 7. ONNX export (optional but recommended for production)
    if args.export_onnx:
        export_onnx(model, X.shape[1], args.output.replace(".pt", ".onnx"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the NatyaNetra Mudra MLP Classifier")
    parser.add_argument("--data",         type=str,   default="landmarks_dataset.npz",
                        help="Path to .npz file from prepare_dataset.py")
    parser.add_argument("--output",       type=str,   default="mudra_model.pt",
                        help="Output checkpoint path")
    parser.add_argument("--epochs",       type=int,   default=150)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--export_onnx",  action="store_true",
                        help="Also export to ONNX after training")
    args = parser.parse_args()
    main(args)
