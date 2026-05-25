from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from survival_utils import hazard_to_survival, logits_to_hazard, survival_to_time


class PredictionHead(nn.Module):
    """MLP prediction head for discrete-time survival prediction."""

    def __init__(
        self,
        input_dim: int,
        task: str = "survival",
        num_classes: int = 2,
        num_time_bins: int = 15,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        norm: str = "batchnorm",
    ):
        super().__init__()

        if task != "survival":
            raise ValueError("SSL3D survival integration only supports task='survival'.")
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be >= 2 for the survival head.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0.")

        self.input_dim = input_dim
        self.task = task
        self.num_classes = num_classes
        self.num_time_bins = num_time_bins
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.norm = norm.lower()

        hidden_dim2 = hidden_dim // 2

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = self._make_norm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.norm2 = self._make_norm(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim2, num_time_bins)

    def _make_norm(self, dim: int) -> nn.Module:
        if self.norm in {"batchnorm", "batch_norm", "batch"}:
            return nn.BatchNorm1d(dim)
        if self.norm in {"layernorm", "layer_norm", "layer"}:
            return nn.LayerNorm(dim)
        if self.norm in {"none", "identity", ""}:
            return nn.Identity()
        raise ValueError("norm must be one of: batchnorm, layernorm, none.")

    def _activate_block(self, x: torch.Tensor, norm: nn.Module, dropout: nn.Module) -> torch.Tensor:
        if isinstance(norm, nn.BatchNorm1d) and self.training and x.shape[0] == 1:
            x = F.relu(x)
        else:
            x = F.relu(norm(x))
        return dropout(x)

    def forward(
        self,
        features: torch.Tensor,
        task: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        if task is not None and task != "survival":
            raise ValueError("SurvivalHead only supports survival predictions.")

        if isinstance(features, dict):
            features = list(features.values())[0]
            if features.ndim == 3:
                features = features.squeeze(1)

        if features.ndim == 1:
            features = features.unsqueeze(0)

        x = self._activate_block(self.fc1(features), self.norm1, self.dropout1)
        x = self._activate_block(self.fc2(x), self.norm2, self.dropout2)

        logits = self.fc3(x)
        hazard = logits_to_hazard(logits)
        survival = hazard_to_survival(hazard)

        return {
            "logits": logits,
            "hazard": hazard,
            "survival": survival,
            "survival_time": survival_to_time(survival),
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "PredictionHead",
            "input_dim": self.input_dim,
            "task": self.task,
            "num_classes": self.num_classes,
            "num_time_bins": self.num_time_bins,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_rate,
            "norm": self.norm,
        }


class SurvivalHead(PredictionHead):
    """Convenience class for survival prediction."""

    def __init__(
        self,
        input_dim: int,
        num_time_bins: int = 15,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        norm: str = "batchnorm",
    ):
        super().__init__(
            input_dim=input_dim,
            task="survival",
            num_classes=2,
            num_time_bins=num_time_bins,
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm=norm,
        )
