from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from survival_utils import hazard_to_survival, logits_to_hazard, survival_to_time


_VALID_LOSSES = ("nll", "cox", "deephit", "soft_logrank", "pmf", "mtlr", "bcesurv", "pchazard", "weibull")


def _nll_surv(phi: torch.Tensor) -> torch.Tensor:
    return torch.cumprod(1 - torch.sigmoid(phi), dim=1)


def _pmf_surv(phi: torch.Tensor) -> torch.Tensor:
    return 1 - F.softmax(phi, dim=1).cumsum(1)


def _bce_surv(phi: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(phi)


def _mtlr_surv(phi: torch.Tensor) -> torch.Tensor:
    # Replicate pycox MTLR.predict_pmf + PMFBase.predict_surv:
    #   g = cumsum_reverse(phi)                  [B, K]
    #   pmf = pad_col(g).softmax(1)[:, :-1]     [B, K]
    #   surv = 1 - pmf.cumsum(1)                [B, K]
    from pycox.models.utils import cumsum_reverse, pad_col
    g = cumsum_reverse(phi, dim=1)
    pmf = pad_col(g).softmax(1)[:, :-1]
    return 1 - pmf.cumsum(1)


LOGITS_TO_SURVIVAL = {
    "nll": _nll_surv,
    "pmf": _pmf_surv,
    "deephit": _pmf_surv,
    "bcesurv": _bce_surv,
    "mtlr": _mtlr_surv,
}


def logits_to_survival(name: str, phi: torch.Tensor) -> torch.Tensor:
    """Derive a survival curve from raw logits for the given loss name."""
    return LOGITS_TO_SURVIVAL[name](phi)


class PredictionHead(nn.Module):
    """MLP prediction head for discrete-time survival prediction.

    Emits a stable output dict regardless of which loss is active; the
    ``survival`` field is computed in the loss-appropriate way.
    """

    def __init__(
        self,
        input_dim: int,
        task: str = "survival",
        num_classes: int = 2,
        num_time_bins: int = 15,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        norm: str = "batchnorm",
        survival_loss_name: str = "nll",
    ):
        super().__init__()

        if task != "survival":
            raise ValueError("SSL3D survival integration only supports task='survival'.")
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be >= 2 for the survival head.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0.")
        if survival_loss_name not in _VALID_LOSSES:
            raise ValueError(
                f"survival_loss_name must be one of {_VALID_LOSSES}, "
                f"got {survival_loss_name!r}."
            )

        self.input_dim = input_dim
        self.task = task
        self.num_classes = num_classes
        self.num_time_bins = num_time_bins
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.norm = norm.lower()
        self.survival_loss_name = survival_loss_name

        hidden_dim2 = hidden_dim // 2

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = self._make_norm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.norm2 = self._make_norm(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout)

        # Three parallel terminal projections. Only one receives gradient
        # under any given training run; the other two are along for the ride.
        self.fc_hazard = nn.Linear(hidden_dim2, num_time_bins)
        self.fc_pmf = nn.Linear(hidden_dim2, num_time_bins)
        self.fc_risk = nn.Linear(hidden_dim2, 1)
        self.fc_weibull = nn.Linear(hidden_dim2, 2)

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

    def _survival_for_active_loss(
        self,
        hazard: torch.Tensor,
        hazard_logits: torch.Tensor,
        pmf_logits: torch.Tensor,
        weibull_params: torch.Tensor,
    ) -> torch.Tensor:
        name = self.survival_loss_name
        if name in ("pmf", "deephit"):
            return logits_to_survival(name, pmf_logits)
        if name in ("bcesurv", "mtlr", "nll"):
            return logits_to_survival(name, hazard_logits)
        if name == "weibull":
            from torchsurv.loss import weibull as _wb
            times = torch.arange(
                1, self.num_time_bins + 1,
                device=weibull_params.device,
                dtype=weibull_params.dtype,
            )
            surv = _wb.survival_function_weibull(weibull_params, times)
            return surv.clamp(0.0, 1.0)
        # cox / soft_logrank / pchazard: fall back to cumprod(1-hazard)
        return hazard_to_survival(hazard)

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

        hazard_logits = self.fc_hazard(x)
        pmf_logits = self.fc_pmf(x)
        risk = self.fc_risk(x).squeeze(-1)
        weibull_params = self.fc_weibull(x)
        # p_high is the soft high-risk membership used by the soft_logrank loss
        # and by inference's KM grouping. Derived from the same fc_risk scalar
        # that Cox trains; under soft_logrank, gradient flows through sigmoid
        # back into fc_risk.
        p_high = torch.sigmoid(risk)

        hazard = logits_to_hazard(hazard_logits)
        pmf = F.softmax(pmf_logits.float(), dim=1)
        survival = self._survival_for_active_loss(hazard, hazard_logits, pmf_logits, weibull_params)

        # Stable dict: every key is present in every mode, but only some are
        # trained/meaningful for the active loss.
        #   nll          -> logits, hazard, survival, survival_time meaningful
        #   deephit      -> pmf, survival, survival_time meaningful
        #   cox          -> risk meaningful (as log hazard ratio)
        #   soft_logrank -> risk meaningful (as logit); p_high = sigmoid(risk)
        #   weibull      -> weibull_params [B,2], survival (analytic Weibull)
        return {
            "logits": hazard_logits,
            "hazard": hazard,
            "pmf_logits": pmf_logits,
            "pmf": pmf,
            "risk": risk,
            "weibull_params": weibull_params,
            "survival": survival,
            "survival_time": survival_to_time(survival),
            "p_high": p_high,
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
            "survival_loss_name": self.survival_loss_name,
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
        survival_loss_name: str = "nll",
    ):
        super().__init__(
            input_dim=input_dim,
            task="survival",
            num_classes=2,
            num_time_bins=num_time_bins,
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm=norm,
            survival_loss_name=survival_loss_name,
        )
