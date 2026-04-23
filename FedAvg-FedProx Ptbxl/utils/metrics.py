import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)


CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]


def evaluate(model, dataset, device, batch_size=64):
    """Returns accuracy and macro F1 on a dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds = model(X).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


def full_report(model, dataset, device, batch_size=64):
    """Prints full classification report + confusion matrix."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds = model(X).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES, zero_division=0
    ))
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    return all_labels, all_preds
