import numpy as np
from torch.utils.data import Subset

def dirichlet_partition(dataset, num_clients, alpha=0.3, seed=42):
    """
    Partition dataset across FL clients using Dirichlet distribution.
    Lower alpha = more non-IID (each client gets fewer classes).

    Returns: list of index arrays, one per client
    """
    np.random.seed(seed)
    labels = dataset.y.numpy()
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        np.random.shuffle(cls_idx)

        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(cls_idx)).astype(int)
        # Fix rounding error so no indices are lost
        proportions[-1] = len(cls_idx) - proportions[:-1].sum()

        splits = np.split(cls_idx, np.cumsum(proportions)[:-1])
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    return [np.array(idx) for idx in client_indices]


def get_client_subsets(dataset, client_indices):
    return [Subset(dataset, idx) for idx in client_indices]


def print_partition_stats(client_indices, dataset, num_classes=5):
    """Print how many samples of each class each client has."""
    labels = dataset.y.numpy()
    print(f"\n{'Client':<10}", end="")
    class_names = ["NORM", "MI", "STTC", "CD", "HYP"]
    for c in class_names:
        print(f"{c:>8}", end="")
    print(f"{'Total':>8}")
    print("-" * 54)

    for i, idx in enumerate(client_indices):
        client_labels = labels[idx]
        print(f"Client {i:<4}", end="")
        for cls in range(num_classes):
            count = (client_labels == cls).sum()
            print(f"{count:>8}", end="")
        print(f"{len(idx):>8}")
