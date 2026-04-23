import torch
import torch.nn as nn

class ECGCNN(nn.Module):
    """
    1D CNN for 12-lead ECG classification.
    Input shape: (batch, 12, 5000)
    Output: (batch, num_classes)
    """
    def __init__(self, num_classes=5, num_leads=12):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(num_leads, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),          # → (32, 1250)

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),          # → (64, 312)

            # Block 3
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),          # → (128, 78)

            # Block 4
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),  # → (256, 8) regardless of input length
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
