import torch
import torch.nn as nn
from torch.utils.data import DataLoader



class FLClient:
    def __init__(self, client_id, subset, device):
        self.id = client_id
        self.subset = subset
        self.device = device

        # Compute class weights here
        labels = torch.tensor([
            self.subset.dataset.y[i].item()
            for i in self.subset.indices
        ])
        class_counts = torch.bincount(labels, minlength=5).float()
        class_counts = class_counts.clamp(min=1)

        weights = (1.0 / class_counts)
        weights = weights / weights.sum() * 5
        self.weights = weights.to(self.device)

    def train(self, model, epochs, lr, batch_size,
              mu=0.0, global_model=None):
        """
        Train local model.
        mu=0   → FedAvg (plain cross-entropy)
        mu>0   → FedProx (adds proximal regularization term)
        """
        model.train()
        loader = DataLoader(
            self.subset, batch_size=batch_size,
            shuffle=True, num_workers=0, pin_memory=False
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=self.weights)


        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(epochs):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                logits = model(X)
                loss = criterion(logits, y)

                if mu > 0.0 and global_model is not None:
                    prox_term = 0.0
                    for w_local, w_global in zip(
                        model.parameters(), global_model.parameters()
                    ):
                        prox_term += (w_local - w_global.detach()).norm(2) ** 2
                    loss += (mu / 2.0) * prox_term

                loss.backward()
                # Gradient clipping — used for ECG signals
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * y.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0
        return model.state_dict(), avg_loss, train_acc
