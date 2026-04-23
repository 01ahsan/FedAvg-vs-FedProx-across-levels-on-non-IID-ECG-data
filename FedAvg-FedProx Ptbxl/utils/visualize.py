import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
os.makedirs("results", exist_ok=True)


def plot_accuracy_f1(history, num_rounds, alpha, save=True):
    """Two-panel plot: accuracy and macro-F1 across rounds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"fedavg": "#E63946", "fedprox": "#457B9D"}
    lines  = {"fedavg": "--",      "fedprox": "-"}

    for algo in ["fedavg", "fedprox"]:
        accs = [h["acc"] for h in history[algo]]
        f1s  = [h["f1"]  for h in history[algo]]
        rounds = range(1, num_rounds + 1)
        label = algo.upper()

        axes[0].plot(rounds, accs, linestyle=lines[algo],
                     color=colors[algo], linewidth=2, label=label)
        axes[1].plot(rounds, f1s,  linestyle=lines[algo],
                     color=colors[algo], linewidth=2, label=label)

    for ax, title, ylabel in zip(
        axes,
        [f"Test Accuracy (α={alpha})", f"Macro F1-Score (α={alpha})"],
        ["Accuracy", "Macro F1"]
    ):
        ax.set_xlabel("Communication Rounds", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    plt.suptitle(
        "FedAvg vs FedProx on PTB-XL ECG Dataset",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    if save:
        path = f"results/accuracy_f1_alpha{alpha}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


def plot_confusion_matrix(labels, preds, algo_name, alpha, save=True):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(
        f"{algo_name.upper()} Confusion Matrix (α={alpha})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    if save:
        path = f"results/cm_{algo_name}_alpha{alpha}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


def plot_client_distribution(client_indices, dataset, num_classes=5, save=True):
    """Stacked bar showing class distribution per client."""
    labels = dataset.y.numpy()
    data = np.zeros((len(client_indices), num_classes))
    for i, idx in enumerate(client_indices):
        for cls in range(num_classes):
            data[i, cls] = (labels[idx] == cls).sum()

    data_pct = data / data.sum(axis=1, keepdims=True) * 100
    colors = ["#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#264653"]

    fig, ax = plt.subplots(figsize=(12, 5))
    bottoms = np.zeros(len(client_indices))
    for cls in range(num_classes):
        ax.bar(range(len(client_indices)), data_pct[:, cls],
               bottom=bottoms, label=CLASS_NAMES[cls],
               color=colors[cls], edgecolor="white", linewidth=0.5)
        bottoms += data_pct[:, cls]

    ax.set_xlabel("Client ID", fontsize=12)
    ax.set_ylabel("Class Distribution (%)", fontsize=12)
    ax.set_title("Non-IID Data Distribution Across FL Clients", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(client_indices)))
    ax.set_xticklabels([f"C{i}" for i in range(len(client_indices))])
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    if save:
        plt.savefig("results/client_distribution.png", dpi=150, bbox_inches="tight")
        print("Saved → results/client_distribution.png")
    plt.show()
