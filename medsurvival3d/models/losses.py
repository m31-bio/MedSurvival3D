"""Survival loss functions, criteria, and the loss-dispatch map."""
import torch
import torch.nn as nn
from medsurvival3d.utils.survival_labels import interval_frac


def logits_to_hazard(logits: torch.Tensor) -> torch.Tensor:
    """Convert hazard logits to hazard probabilities."""
    return torch.sigmoid(logits)


def hazard_to_survival(hazard: torch.Tensor) -> torch.Tensor:
    """Convert hazard probabilities to a survival function."""
    return torch.cumprod(1.0 - hazard, dim=1)


def survival_to_time(survival: torch.Tensor) -> torch.Tensor:
    """Convert a survival function to expected discrete survival time."""
    return torch.sum(survival, dim=1)


def soft_logrank_loss(
    p_high: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """Directional Mantel-Cox log-rank signal on soft high-risk membership.

    Encourages high-risk patients to have more observed events than expected.
    Returns ``-mean_over_event_times(o_high - e_high)`` so lower = better.
    """
    p_high = p_high.float().view(-1)
    time = time.to(device=p_high.device, dtype=torch.float32).view(-1)
    event = event.to(device=p_high.device, dtype=torch.float32).view(-1)

    # Grad-preserving zero used as the no-op / no-event fallback.
    zero = p_high.sum() * 0.0

    event_mask = event > 0.5
    if not torch.any(event_mask):
        return zero

    unique_event_times = torch.unique(time[event_mask])
    if unique_event_times.numel() == 0:
        return zero

    signal = zero
    for t in unique_event_times:
        at_risk = time >= t
        n_total = at_risk.sum()
        if n_total <= 0:
            continue
        events_at_t = at_risk & (time == t) & event_mask
        d_total = events_at_t.sum()
        if d_total <= 0:
            continue
        n_high = p_high[at_risk].sum()
        o_high = p_high[events_at_t].sum()
        e_high = d_total.float() * n_high / n_total.float()
        signal = signal + (o_high - e_high)

    return -signal / float(unique_event_times.numel())


def group_balance_penalty(
    p_high: torch.Tensor,
    min_frac: float = 0.20,
    max_frac: float = 0.80,
) -> torch.Tensor:
    """Weak range-clamp penalty that fires only when ``p_high.mean()`` exits
    ``[min_frac, max_frac]``. Returns a non-negative scalar."""
    p_high = p_high.float().view(-1)
    mean_high = p_high.mean()
    too_small = torch.relu(torch.tensor(min_frac, device=p_high.device) - mean_high)
    too_large = torch.relu(mean_high - torch.tensor(max_frac, device=p_high.device))
    return too_small.pow(2) + too_large.pow(2)


class NLLSurvLoss(nn.Module):
    """pycox logistic-hazard NLL. Input: raw hazard logits, int time bins, event."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        from pycox.models.loss import NLLLogistiHazardLoss
        self._loss = NLLLogistiHazardLoss(reduction=reduction)

    def forward(self, logits, time, event):
        logits = logits.float()  # pycox scatter requires logits/events same dtype (16-mixed AMP)
        idx = time.to(torch.int64).view(-1)
        ev = event.to(torch.float32).view(-1)
        return self._loss(logits, idx, ev)


class CoxPHLoss(nn.Module):
    """Cox partial likelihood via torchsurv (Efron ties). Input: risk, time, event."""

    def __init__(self, reduction: str = "mean", ties_method: str = "efron"):
        super().__init__()
        from torchsurv.loss import cox
        self._fn = cox.neg_partial_log_likelihood
        self.reduction = reduction
        self.ties_method = ties_method

    def forward(self, risk, time, event):
        log_hz = risk.float().view(-1)
        ev = event.to(torch.bool).view(-1)
        if ev.sum() == 0:
            return log_hz.sum() * 0.0
        t = time.to(torch.float32).view(-1)
        return self._fn(log_hz, ev, t, ties_method=self.ties_method,
                        reduction=self.reduction, checks=False)


class DeepHitLoss(nn.Module):
    """Single-event DeepHit via pycox. Input: raw pmf logits, int time bins, event.

    total = alpha * NLL + (1 - alpha) * ranking. No calibration term (pycox).
    """
    def __init__(self, alpha: float = 0.2, sigma: float = 0.1):
        super().__init__()
        from pycox.models.loss import DeepHitSingleLoss
        self._loss = DeepHitSingleLoss(alpha=float(alpha), sigma=float(sigma))

    def forward(self, pmf_logits, time, event):
        from pycox.models.data import pair_rank_mat
        pmf_logits = pmf_logits.float()  # pycox DeepHit rank matmul requires fp32 (16-mixed AMP)
        idx = time.to(torch.int64).view(-1)
        ev = event.to(torch.int64).view(-1)
        rank_mat = torch.as_tensor(
            pair_rank_mat(idx.detach().cpu().numpy(), ev.detach().cpu().numpy()),
            dtype=pmf_logits.dtype, device=pmf_logits.device,
        )
        return self._loss(pmf_logits, idx, ev.to(pmf_logits.dtype), rank_mat)


class PMFLoss(nn.Module):
    """pycox PMF NLL. Input: raw pmf logits, int time bins, event."""
    def __init__(self):
        super().__init__()
        from pycox.models.loss import NLLPMFLoss
        self._loss = NLLPMFLoss()

    def forward(self, pmf_logits, time, event):
        return self._loss(pmf_logits, time.to(torch.int64).view(-1), event.to(torch.float32).view(-1))


class MTLRLoss(nn.Module):
    """pycox MTLR NLL. Input: raw logits [B,K], int time bins, event."""
    def __init__(self):
        super().__init__()
        from pycox.models.loss import NLLMTLRLoss
        self._loss = NLLMTLRLoss()

    def forward(self, logits, time, event):
        return self._loss(logits, time.to(torch.int64).view(-1), event.to(torch.float32).view(-1))


class BCESurvLoss(nn.Module):
    """pycox BCESurv loss. Input: raw logits [B,K], int time bins, event."""
    def __init__(self):
        super().__init__()
        from pycox.models.loss import BCESurvLoss as _BCE
        self._loss = _BCE()

    def forward(self, logits, time, event):
        return self._loss(logits, time.to(torch.int64).view(-1), event.to(torch.float32).view(-1))


class PCHazardLoss(nn.Module):
    """pycox piecewise-constant hazard NLL. Input: logits, time bins, event, interval_frac."""
    def __init__(self):
        super().__init__()
        from pycox.models.loss import NLLPCHazardLoss
        self._loss = NLLPCHazardLoss()

    def forward(self, logits, time, event, interval_frac):
        return self._loss(logits, time.to(torch.int64).view(-1),
                          event.to(torch.float32).view(-1),
                          interval_frac.to(logits.dtype).view(-1))


class WeibullLoss(nn.Module):
    """Parametric Weibull AFT via torchsurv. Input: log_params [B,2], time, event."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        from torchsurv.loss import weibull
        self._fn = weibull.neg_log_likelihood_weibull
        self.reduction = reduction

    def forward(self, log_params, time, event):
        ev = event.to(torch.bool).view(-1)
        if ev.sum() == 0:
            return log_params.sum() * 0.0
        return self._fn(log_params, ev, time.float().view(-1),
                        reduction=self.reduction, checks=False)


class SoftLogRankLoss(nn.Module):
    """Differentiable Mantel-Cox log-rank loss + weak group-balance penalty.

    ``forward`` takes ``p_high`` in [0, 1], continuous ``time``, and binary
    ``event`` and returns ``(total, components)`` where ``components`` exposes
    the individual terms and diagnostic statistics for logging.
    """

    def __init__(
        self,
        lambda_balance: float = 0.01,
        min_frac: float = 0.20,
        max_frac: float = 0.80,
    ):
        super().__init__()
        if lambda_balance < 0.0:
            raise ValueError("lambda_balance must be non-negative.")
        if not 0.0 <= min_frac <= max_frac <= 1.0:
            raise ValueError("require 0 <= min_frac <= max_frac <= 1.")
        self.lambda_balance = float(lambda_balance)
        self.min_frac = float(min_frac)
        self.max_frac = float(max_frac)

    def forward(
        self,
        p_high: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logrank = soft_logrank_loss(p_high, time, event)
        balance = group_balance_penalty(
            p_high,
            min_frac=self.min_frac,
            max_frac=self.max_frac,
        )
        total = logrank + self.lambda_balance * balance

        with torch.no_grad():
            p_h = p_high.detach().float().view(-1)
            components = {
                "logrank": logrank.detach(),
                "balance": balance.detach(),
                "p_high_mean": p_h.mean(),
                "p_high_min": p_h.min() if p_h.numel() > 0 else torch.tensor(0.0, device=p_h.device),
                "p_high_max": p_h.max() if p_h.numel() > 0 else torch.tensor(0.0, device=p_h.device),
                "fraction_high_hard": (p_h >= 0.5).float().mean(),
            }
        return total, components


class CompositeSurvivalLoss(nn.Module):
    """Container for a weighted sum of survival losses.

    Holds the member criteria (so their parameters register), their names, the
    parallel weights, and the ``primary`` member name. The weighted summation
    and the ``y_hat -> input`` mapping live in ``base_model._survival_loss``;
    this class is only the data holder.
    """

    def __init__(self, members, names, weights, primary):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.names = list(names)
        self.weights = list(weights)
        self.primary = primary


def _reject_legacy_cox_loss_lambda(kwargs):
    if "cox_loss_lambda" in kwargs and kwargs["cox_loss_lambda"] is not None:
        raise ValueError(
            "`cox_loss_lambda` is no longer supported. Use "
            "`model.survival_loss: {name: cox}` to train Cox alone or "
            "`{name: nll}` for plain NLL."
        )


_SINGLE_LOSS_NAMES = (
    "nll", "cox", "deephit", "soft_logrank",
    "pmf", "mtlr", "bcesurv", "weibull", "pchazard",
)


def _parse_composite(cfg):
    """Validate a composite ``survival_loss`` block and normalise it.

    Returns ``(components, primary)`` where ``components`` is a list of
    ``{"name", "cfg", "weight"}`` dicts in config order (``cfg`` is the original
    per-component mapping, carrying that loss's own options) and ``primary`` is
    the lowercased name of the component that drives metrics/inference.

    Validates without constructing any loss objects, so it is dependency-free.
    Raises ``ValueError`` on any malformed block.
    """
    if "components" not in cfg or not cfg["components"]:
        raise ValueError(
            "composite survival_loss requires a non-empty 'components' list."
        )
    if "primary" not in cfg or cfg["primary"] is None:
        raise ValueError(
            "composite survival_loss requires a 'primary' naming one component."
        )

    components = []
    seen = set()
    for entry in cfg["components"]:
        if "name" not in entry:
            raise ValueError("each composite component needs a 'name'.")
        nm = str(entry["name"]).lower()
        if nm == "composite":
            raise ValueError(
                "composite components cannot themselves be 'composite' (no nesting)."
            )
        if nm not in _SINGLE_LOSS_NAMES:
            raise ValueError(
                f"unknown composite component name: {nm!r}. Expected one of: "
                + ", ".join(_SINGLE_LOSS_NAMES) + "."
            )
        if nm in seen:
            raise ValueError(f"duplicate composite component name: {nm!r}.")
        seen.add(nm)
        weight = entry.get("weight", 1.0)
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            raise ValueError(
                f"composite component {nm!r} has non-numeric weight: {weight!r}."
            )
        if weight < 0:
            raise ValueError(
                f"composite component {nm!r} has negative weight: {weight}."
            )
        components.append({"name": nm, "cfg": entry, "weight": weight})

    primary = str(cfg["primary"]).lower()
    if primary not in seen:
        raise ValueError(
            f"composite 'primary' {primary!r} is not among components: {sorted(seen)}."
        )
    return components, primary


def _build_single_criterion(name, cfg):
    """Construct one survival-loss module from its (lowercased) name and config.

    ``cfg`` is the per-loss mapping carrying that loss's options. Shared by the
    single-loss path and each member of a composite.
    """
    if name == "nll":
        return NLLSurvLoss(reduction=cfg.get("reduction", "mean"))
    if name == "cox":
        return CoxPHLoss(reduction=cfg.get("reduction", "mean"))
    if name == "deephit":
        return DeepHitLoss(
            alpha=cfg.get("alpha", 0.2),
            sigma=cfg.get("sigma", 0.1),
        )
    if name == "soft_logrank":
        return SoftLogRankLoss(
            lambda_balance=float(cfg.get("lambda_balance", 0.01)),
            min_frac=float(cfg.get("min_frac", 0.20)),
            max_frac=float(cfg.get("max_frac", 0.80)),
        )
    if name == "pmf":
        return PMFLoss()
    if name == "mtlr":
        return MTLRLoss()
    if name == "bcesurv":
        return BCESurvLoss()
    if name == "weibull":
        return WeibullLoss(reduction=cfg.get("reduction", "mean"))
    if name == "pchazard":
        return PCHazardLoss()
    raise ValueError(
        f"Unknown survival_loss.name: {name!r}. Expected one of: "
        + ", ".join(_SINGLE_LOSS_NAMES) + "."
    )


def build_survival_criterion(cfg, num_time_bins: int):
    """Return (name, criterion) for a survival_loss config block.

    ``cfg`` is the OmegaConf/dict block under ``model.survival_loss``, or
    ``None`` (in which case the default ``{name: 'nll'}`` is used).

    Supported names and their hyperparameters:
      - ``nll``          — pycox logistic-hazard NLL. Opts: reduction (default "mean").
      - ``cox``          — Cox partial likelihood (Efron ties). Opts: reduction (default "mean").
      - ``deephit``      — Single-event DeepHit. Opts: alpha (default 0.2), sigma (default 0.1).
      - ``soft_logrank`` — Differentiable log-rank + balance penalty.
                           Opts: lambda_balance (0.01), min_frac (0.20), max_frac (0.80).
      - ``pmf``          — pycox PMF NLL. No extra opts.
      - ``mtlr``         — pycox MTLR NLL. No extra opts.
      - ``bcesurv``      — pycox BCESurv loss. No extra opts.
      - ``weibull``      — Parametric Weibull AFT. Opts: reduction (default "mean").
      - ``pchazard``     — pycox piecewise-constant hazard NLL. No extra opts.
      - ``composite``    — weighted sum of any of the above. Requires a
                           ``components`` list (each a single-loss block plus an
                           optional ``weight``, default 1.0) and a ``primary``
                           naming the member that drives metrics/inference.
    """
    if cfg is None:
        cfg = {"name": "nll"}
    # OmegaConf nodes behave dict-like enough for our purposes.
    if not hasattr(cfg, "get") or not hasattr(cfg, "__contains__"):
        raise ValueError(
            "survival_loss must be a mapping with a 'name' key (e.g. "
            "`survival_loss: {name: deephit}`); got " + repr(cfg) + "."
        )
    if "name" not in cfg:
        raise ValueError(
            "survival_loss block must specify 'name' (one of: nll, cox, deephit)."
        )
    name = str(cfg["name"]).lower()

    if name == "composite":
        components, primary = _parse_composite(cfg)
        members = [_build_single_criterion(c["name"], c["cfg"]) for c in components]
        names = [c["name"] for c in components]
        weights = [c["weight"] for c in components]
        return name, CompositeSurvivalLoss(members, names, weights, primary)

    return name, _build_single_criterion(name, cfg)


_SURVIVAL_LOSS_TAGS = {
    "nll": "NLL",
    "cox": "CoxPH",
    "deephit": "DeepHit",
    "soft_logrank": "SoftLogRank",
    "weibull": "Weibull",
    "pmf": "PMF",
    "mtlr": "MTLR",
    "bcesurv": "BCESurv",
    "pchazard": "PCHazard",
    "composite": "Composite",
}


def call_one_loss(name, criterion, y_hat, time_bin, event, continuous_time, bin_edges):
    """Run one survival criterion -> (loss_tensor, components). Pure; no module state."""
    if name == "nll":
        return criterion(y_hat["logits"], time_bin, event), {}
    if name == "cox":
        return criterion(y_hat["risk"], continuous_time, event), {}
    if name == "deephit":
        return criterion(y_hat["pmf_logits"], time_bin, event), {}
    if name == "soft_logrank":
        total, components = criterion(y_hat["p_high"], continuous_time, event)
        return total, components
    if name == "pmf":
        return criterion(y_hat["pmf_logits"], time_bin, event), {}
    if name in ("mtlr", "bcesurv"):
        return criterion(y_hat["logits"], time_bin, event), {}
    if name == "weibull":
        return criterion(y_hat["weibull_params"], continuous_time, event), {}
    if name == "pchazard":
        frac = interval_frac(continuous_time, time_bin, bin_edges)
        return criterion(y_hat["logits"], time_bin, event, frac), {}
    raise ValueError(f"Unexpected survival_loss_name: {name!r}")
