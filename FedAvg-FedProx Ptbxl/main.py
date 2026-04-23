import os, copy, random, yaml
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from data.dataloader import load_all_splits
from data.partition import dirichlet_partition, get_client_subsets, print_partition_stats
from models.ecg_cnn import ECGCNN
from client.client import FLClient
from algorithms.fedavg import aggregate
from utils.metrics import evaluate, full_report
from utils.visualize import plot_accuracy_f1, plot_confusion_matrix, plot_client_distribution

os.makedirs("results", exist_ok=True)

# ── Load config ────────────────────────────────────────────────────────────────
with open("experiments/config.yaml") as f:
    cfg = yaml.safe_load(f)

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(cfg["seed"])
np.random.seed(cfg["seed"])
torch.manual_seed(cfg["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# ── Load data ──────────────────────────────────────────────────────────────────
print("\nLoading PTB-XL splits...")
train_data, val_data, test_data = load_all_splits(cfg["data_root"])
print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

# ── Partition across clients ───────────────────────────────────────────────────
alpha = cfg["alpha"]
client_indices = dirichlet_partition(
    train_data, cfg["num_clients"], alpha=alpha, seed=cfg["seed"]
)
client_subsets = get_client_subsets(train_data, client_indices)

print(f"\nNon-IID Partition (α={alpha}):")
print_partition_stats(client_indices, train_data, cfg["num_classes"])
plot_client_distribution(client_indices, train_data, cfg["num_classes"])

clients = [
    FLClient(i, client_subsets[i], device)
    for i in range(cfg["num_clients"])
]

# ── FL Training Loop ───────────────────────────────────────────────────────────
all_results = {}
summary_rows = []

for algo in ["fedavg", "fedprox"]:
    mu = cfg["mu"] if algo == "fedprox" else 0.0
    print(f"\n{'='*60}")
    print(f"  Algorithm: {algo.upper()}   μ={mu}   α={alpha}")
    print(f"{'='*60}")

    global_model = ECGCNN(num_classes=cfg["num_classes"]).to(device)
    history = []
    best_f1 = 0.0
    best_weights = None

    for rnd in tqdm(range(cfg["num_rounds"]), desc=algo.upper()):
        local_weights, local_sizes = [], []

        # ── Client training ──
        for client in clients:
            local_model = copy.deepcopy(global_model)
            weights, loss, acc = client.train(
                local_model,
                epochs=cfg["local_epochs"],
                lr=cfg["lr"],
                batch_size=cfg["batch_size"],
                mu=mu,
                global_model=global_model
            )
            local_weights.append(weights)
            local_sizes.append(len(client.subset))

        # ── Aggregation ──
        global_model = aggregate(global_model, local_weights, local_sizes)

        # ── Evaluation on val set ──
        val_acc, val_f1 = evaluate(global_model, val_data, device)
        history.append({"acc": val_acc, "f1": val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = copy.deepcopy(global_model.state_dict())

        if (rnd + 1) % 10 == 0:
            tqdm.write(
                f"  Round {rnd+1:>3}/{cfg['num_rounds']} | "
                f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
            )

    # ── Final test evaluation ──
    print(f"\n[{algo.upper()}] Loading best checkpoint (Val F1={best_f1:.4f})")
    global_model.load_state_dict(best_weights)
    test_acc, test_f1 = evaluate(global_model, test_data, device)
    print(f"[{algo.upper()}] TEST Accuracy: {test_acc:.4f} | TEST Macro F1: {test_f1:.4f}")

    # ── Full classification report ──
    labels, preds = full_report(global_model, test_data, device)
    plot_confusion_matrix(labels, preds, algo, alpha)

    # ── Save model ──
    model_path = f"results/{algo}_best.pth"
    torch.save(best_weights, model_path)
    print(f"Model saved → {model_path}")

    all_results[algo] = history
    summary_rows.append({
        "Algorithm": algo.upper(),
        "Alpha": alpha,
        "Rounds": cfg["num_rounds"],
        "Test Accuracy": round(test_acc, 4),
        "Test Macro F1": round(test_f1, 4),
        "Best Val F1": round(best_f1, 4),
    })

# Plots, Summary 
plot_accuracy_f1(all_results, cfg["num_rounds"], alpha)

df = pd.DataFrame(summary_rows)
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(df.to_string(index=False))
df.to_csv("results/summary.csv", index=False)
print("\nSaved → results/summary.csv")
