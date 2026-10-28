"""Stage 4 & Phase Six: Baseline Model Architectures and Calibration.

Implements:
1. Majority Class Dummy Baseline
2. Balanced Logistic Regression (Linear Baseline)
3. Random Forest Classifier
4. Gradient Boosting Classifier (GBDT)
5. Compact Neural Baseline (MLP)
6. Temporal BiGRU Classifier (for student sequence tracks)
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV


class StudentBiGRU(nn.Module):
    """Compact Temporal BiGRU sequence model for student trajectory features."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1, num_classes: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.gru(x)
        # Take mean-pooling across temporal dimension
        pooled = torch.mean(out, dim=1)
        logits = self.fc(pooled)
        return logits


def get_baseline_models(seed: int = 42) -> Dict[str, BaseEstimator]:
    """Returns initialized scikit-learn compatible baseline classifiers."""
    return {
        "Majority Class Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (L2)": LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1
        ),
        "Gradient Boosting (GBDT)": HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=5,
            class_weight="balanced",
            random_state=seed
        ),
        "Compact MLP Neural Baseline": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            random_state=seed
        )
    }
