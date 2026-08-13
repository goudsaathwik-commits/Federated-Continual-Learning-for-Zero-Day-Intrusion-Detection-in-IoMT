import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularIDSBackbone(nn.Module):
    """
    Primary Deep Neural Network Backbone for Tabular IoMT Intrusion Detection.
    Features a Multi-Layer Perceptron (MLP) architecture with LayerNorm/BatchNorm,
    Dropout regularization, and Residual Connections tailored for tabular network flow metrics.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list = [256, 128, 64], dropout: float = 0.2):
        super(TabularIDSBackbone, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Input projection layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Block 1
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classification Head (Logits output)
        self.head = nn.Linear(hidden_dims[2], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw class logits g(x).
        """
        out = self.input_layer(x)
        out = self.block1(out)
        out = self.block2(out)
        logits = self.head(out)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
