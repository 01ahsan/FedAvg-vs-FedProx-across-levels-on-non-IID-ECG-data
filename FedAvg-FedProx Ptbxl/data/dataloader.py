import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PTBXLDataset(Dataset):
    """
    Loads preprocessed PTB-XL splits.
    Files expected:
        {split}_signals.npy   → shape (N, 5000, 12)
        {split}_labels.npy    → shape (N,)
    """
    def __init__(self, data_root, split="train"):
        assert split in ("train", "val", "test")

        X = np.load(os.path.join(data_root, f"{split}_signals.npy"), mmap_mode="r")
        y = np.load(os.path.join(data_root, f"{split}_labels.npy"))

        # (N, 5000, 12) → (N, 12, 5000) for PyTorch Conv1d
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_all_splits(data_root):
    train = PTBXLDataset(data_root, "train")
    val   = PTBXLDataset(data_root, "val")
    test  = PTBXLDataset(data_root, "test")
    return train, val, test