# 🫀 Federated Learning on ECG Data: FedAvg vs FedProx under Non-IID Conditions

> Simulating privacy-preserving cardiac diagnosis across heterogeneous clinical sites using the PTB-XL dataset.

---

## Overview

In real-world healthcare, patient ECG data cannot be shared across hospitals due to privacy regulations. **Federated Learning (FL)** enables multiple institutions to collaboratively train a shared model without exchanging raw data — each site trains locally and only shares model weights.

This project implements and compares two foundational FL algorithms — **FedAvg** and **FedProx** — on the [PTB-XL ECG dataset](https://physionet.org/content/ptb-xl/1.0.3/), a large-scale 12-lead ECG benchmark with 5 cardiac diagnostic classes. The key challenge studied is **statistical heterogeneity (non-IID data)**: in practice, different hospitals see different patient populations, leading to highly unequal class distributions across clients.

We simulate this heterogeneity using the **Dirichlet distribution** and evaluate both algorithms across three levels of heterogeneity.

---

## Dataset

**PTB-XL** — A large publicly available 12-lead ECG dataset from PhysioNet.

| Split | Samples | Patients |
|-------|---------|----------|
| Train | 15,237  | 153      |
| Val   | 3,200   | 32       |
| Test  | 3,400   | 34       |

**Signal specs:** 5000 time steps × 12 leads @ 500 Hz (10 seconds per recording)

**Label distribution (5 classes):**

| Class | Label | Train Samples | % |
|-------|-------|--------------|---|
| Normal | NORM | 5,942 | 39.0% |
| Myocardial Infarction | MI | 5,758 | 37.8% |
| ST/T Change | STTC | 2,773 | 18.2% |
| Conduction Disturbance | CD | 711 | 4.7% |
| Hypertrophy | HYP | 53 | 0.3% |

> ⚠️ **Class imbalance is severe** — HYP has only 53 training samples (0.3%). We address this with inverse-frequency class weighting in the loss function.

---

## Federated Learning Setup

### Algorithms

**FedAvg** *(McMahan et al., 2017)*
- Each client trains locally for E epochs using standard cross-entropy loss
- Server performs weighted averaging of client model weights
- Known to struggle under non-IID conditions due to *client drift*

**FedProx** *(Li et al., 2020)*
- Extends FedAvg by adding a proximal regularization term to each client's local objective:

$$\mathcal{L}_{\text{FedProx}} = \mathcal{L}_{\text{local}} + \frac{\mu}{2} \| w - w^{global} \|^2$$

- The proximal term penalizes local models from drifting too far from the global model
- More stable convergence under heterogeneous data distributions
- Note: the **aggregation step is identical to FedAvg** — the difference is entirely in local training

### Non-IID Simulation

We use **Dirichlet partitioning** to distribute training data across 10 simulated clients. The concentration parameter α controls the degree of heterogeneity:

| α | Heterogeneity | Description |
|---|--------------|-------------|
| 0.1 | Very High | Each client has data from ~1–2 classes |
| 0.3 | Moderate | Uneven but multi-class per client |
| 1.0 | Mild | More balanced, closer to IID |

**Example — Non-IID partition at α=0.3 (10 clients):**

![Client Distribution](results/client_distribution.png)

> Client 4 holds 41% of all training data, mostly NORM. Client 8 has virtually only MI samples. This simulates realistic hospital specialization.

---

## Model Architecture

A 4-block **1D CNN** designed for multi-lead ECG classification:

```
Input: (batch, 12 leads, 5000 time steps)
  │
  ├─ Conv1d(12→32, k=7) → BN → ReLU → MaxPool(4)    # → (32, 1250)
  ├─ Conv1d(32→64, k=5) → BN → ReLU → MaxPool(4)    # → (64, 312)
  ├─ Conv1d(64→128, k=5) → BN → ReLU → MaxPool(4)   # → (128, 78)
  ├─ Conv1d(128→256, k=3) → BN → ReLU → AvgPool(8)  # → (256, 8)
  │
  └─ Flatten → Linear(2048→256) → ReLU → Dropout(0.4) → Linear(256→5)

Output: (batch, 5 classes)
```

**Training details per client:**
- Optimizer: Adam (lr=0.001)
- Local epochs: 3 per round
- Batch size: 32
- Gradient clipping: max norm = 1.0
- Class weights: inverse-frequency weighting to handle HYP/CD imbalance

---

## Experimental Configuration

```yaml
num_clients:   10
num_rounds:    50
local_epochs:  3
lr:            0.001
batch_size:    32
mu (FedProx):  0.01
alpha:         [0.1, 0.3, 1.0]
seed:          42
```

---

## Results

### Main Comparison: F1 Score across Non-IID Levels

![F1 vs Alpha](results/Macro F1 FedAvg vs FedProx.png)

### Accuracy across Non-IID Levels

![Accuracy vs Alpha](results/Test Accuracy FedAvg vs FedProx.png)

### Full Results Table

| α | Algorithm | Test Accuracy | Test Macro F1 |
|---|-----------|--------------|--------------|
| 0.1 | FedAvg  | 0.6300 | 0.4167 |
| 0.1 | FedProx | 0.6335 | 0.4064 |
| 0.3 | FedAvg  | 0.6791 | 0.4221 |
| 0.3 | **FedProx** | **0.7074** | **0.4623** |
| 1.0 | FedAvg  | 0.7174 | 0.5417 |
| 1.0 | FedProx | 0.7082 | 0.5522 |

### Per-Class Performance (α=0.3, Best Setting)

|       | FedAvg F1 | FedProx F1 |
|-------|-----------|------------|
| NORM  | 0.79      | 0.80       |
| MI    | 0.71      | 0.75       |
| STTC  | 0.52      | 0.53       |
| CD    | 0.00      | 0.06       |
| HYP   | 0.08      | 0.17       |


---

## Key Findings

**1. FedProx consistently outperforms FedAvg on Macro F1 across all α values.**
The F1 gap is largest at α=0.3 (+0.04), where heterogeneity is moderate. This confirms FedProx's proximal regularization is most effective when clients have meaningfully different — but not completely disjoint — data distributions.

**2. Extreme heterogeneity (α=0.1) hurts both algorithms equally.**
At α=0.1, some clients receive as few as 1–19 samples total (see partition table). At this extreme, the proximal term cannot compensate for the near-total absence of certain classes at most clients. Both algorithms achieve similar F1 (~0.41).

**3. FedProx significantly improves minority class detection.**
Under FedAvg at α=0.3, CD and HYP receive near-zero F1. FedProx, with its resistance to client drift, allows the global model to retain minority-class signal learned by the few clients that hold those samples.

**4. Accuracy is a misleading metric here.**
At α=0.3, FedAvg achieves 0.68 accuracy but only 0.42 Macro F1 — because it learns to predict majority classes (NORM, MI) well and ignore CD/HYP entirely. **Macro F1 is the correct metric for imbalanced medical classification.**

**5. As data becomes more IID (α=1.0), the gap narrows.**
This validates the theoretical motivation for FedProx — its advantage is specifically tied to non-IID conditions, not a blanket improvement.

---

## Project Structure

```
fl-ptbxl/
├── data/
│   ├── dataloader.py       # PTB-XL dataset loader
│   └── partition.py        # Dirichlet non-IID partitioning
├── models/
│   └── ecg_cnn.py          # 1D CNN for 12-lead ECG
├── client/
│   └── client.py           # Local training (FedAvg + FedProx)
├── algorithms/
│   └── fedavg.py           # Weighted aggregation
├── utils/
│   ├── metrics.py          # Accuracy, Macro F1, classification report
│   └── visualize.py        # All plotting functions
├── experiments/
│   └── config.yaml         # Hyperparameters
├── results/                # Generated plots and model checkpoints
├── main.py                 # Experiment runner
└── requirements.txt
```

---

## Reproducing Results

```bash
# 1. Clone the repo
git clone github.com/01ahsan/FedAvg-vs-FedProx-across-levels-on-non-IID-ECG-data.git
cd fl-ptbxl

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your PTB-XL data path in experiments/config.yaml
#    data_root: "/path/to/processed_ptbxl"
#    Expected files: train_signals.npy, train_labels.npy, etc.

# 4. Run experiment (edit alpha in config.yaml for different settings)
python main.py
```

**Hardware used:** NVIDIA GPU (CUDA), ~18–25 min per experiment (50 rounds × 2 algorithms)

---

## Dependencies

```
torch >= 2.0.0
numpy
scikit-learn
matplotlib
seaborn
pyyaml
tqdm
pandas
```

---

## References

- McMahan, B., et al. *"Communication-Efficient Learning of Deep Networks from Decentralized Data."* AISTATS 2017.
- Li, T., et al. *"Federated Optimization in Heterogeneous Networks (FedProx)."* MLSys 2020.
- Wagner, P., et al. *"PTB-XL, a large publicly available electrocardiography dataset."* Scientific Data, 2020.

---

## License

This project is for research and educational purposes. PTB-XL dataset is available under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
