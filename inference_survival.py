import csv
import json
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import hydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
except ModuleNotFoundError:
    hydra = None
    instantiate = None
    OmegaConf = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from parsing_utils import make_omegaconf_resolvers
except ModuleNotFoundError:
    make_omegaconf_resolvers = None

import matplotlib.pyplot as plt
from survival_utils import (
    concordance_index,
    hazard_to_survival,
    integrated_brier_score,
    logits_to_hazard,
    survival_to_time,
)


def _torch_load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _as_path(value):
    return Path(os.path.expanduser(str(value))).resolve()


def _resolve_device(device_cfg):
    if device_cfg is None or str(device_cfg).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_cfg))


def _load_training_cfg(exp_dir, splits_json, fold):
    if OmegaConf is None:
        raise ModuleNotFoundError("Hydra/OmegaConf is required for survival inference.")
    cfg = OmegaConf.load(exp_dir / "config.yaml")
    cfg.model.pretrained = False
    cfg.model.chpt_path = None
    cfg.data.module.fold = int(fold)
    cfg.data.module.split_file = str(_as_path(splits_json))
    cfg.data.module.test_transforms = None
    if "trainer" in cfg:
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1
        cfg.trainer.strategy = "auto"
        cfg.trainer.pop("logger", None)
        cfg.trainer.pop("callbacks", None)
    return cfg


def _fold_dir(exp_dir, fold):
    candidates = [exp_dir / f"fold_{fold}", exp_dir / str(fold)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _checkpoint_score_from_name(path):
    text = str(path)
    patterns = [
        r"(?:Val[/_\- ]?C[-_ ]?index|val[/_\- ]?c[-_ ]?index)[=:_\-]?([0-9]*\.?[0-9]+)",
        r"(?:C[-_ ]?index|c[-_ ]?index)[=:_\-]?([0-9]*\.?[0-9]+)",
    ]
    scores = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            try:
                scores.append(float(match.group(1)))
            except ValueError:
                pass
    return scores[-1] if scores else None


def _checkpoint_score_from_metadata(path):
    checkpoint = _torch_load_checkpoint(path, map_location="cpu")
    callbacks = checkpoint.get("callbacks", {})
    scores = []
    for state in callbacks.values():
        if not isinstance(state, dict):
            continue
        for key, value in state.items():
            if "c-index" not in str(key).lower() and "c_index" not in str(key).lower():
                continue
            if torch.is_tensor(value):
                value = value.detach().cpu().item()
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                scores.append(float(value))
        monitor = str(state.get("monitor", "")).lower()
        value = state.get("best_model_score")
        if ("c-index" in monitor or "c_index" in monitor) and value is not None:
            if torch.is_tensor(value):
                value = value.detach().cpu().item()
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                scores.append(float(value))
    return max(scores) if scores else None


def select_best_checkpoint(exp_dir, fold, checkpoint_glob):
    root = _fold_dir(exp_dir, fold)
    paths = sorted(root.rglob(checkpoint_glob))
    if not paths:
        raise FileNotFoundError(
            f"No checkpoints matching {checkpoint_glob!r} found under {root}"
        )

    scored = []
    missing = []
    for path in paths:
        score = _checkpoint_score_from_name(path)
        if score is None:
            score = _checkpoint_score_from_metadata(path)
        if score is None:
            missing.append(str(path))
        else:
            scored.append((score, path))

    if not scored:
        raise RuntimeError(
            "Could not determine validation C-index for any checkpoint. "
            "Expected a filename such as 'epoch=12-Val_C-index=0.7345.ckpt' "
            "or ModelCheckpoint callback metadata with best_model_score monitored "
            f"on Val/C-index. Checked: {missing}"
        )
    return max(scored, key=lambda item: item[0])


def _bin_columns(labels, num_bins):
    labels = list(labels or [])
    if len(labels) == num_bins:
        return [str(label) for label in labels]
    return [f"bin_{idx}" for idx in range(num_bins)]


def _landmark_indices(landmark_years, upper_bounds, num_bins=None):
    bounds = [None if value is None else float(value) for value in upper_bounds]
    valid = [
        (idx, value)
        for idx, value in enumerate(bounds)
        if value is not None and (num_bins is None or idx < num_bins)
    ]
    result = {}
    for year in landmark_years:
        year = float(year)
        exact = [idx for idx, value in valid if math.isclose(value, year)]
        if exact:
            result[year] = exact[0]
            continue
        earlier = [(idx, value) for idx, value in valid if value < year]
        if earlier:
            result[year] = max(earlier, key=lambda item: item[1])[0]
        elif valid:
            result[year] = min(valid, key=lambda item: abs(item[1] - year))[0]
    return result


def _move_target_to_device(target, device):
    if isinstance(target, dict):
        return {key: value.to(device) for key, value in target.items()}
    return target.to(device)


def _tensor_to_numpy(tensor):
    return tensor.detach().cpu().float().numpy()


def _resolve_survival_loss_name(training_cfg):
    """Read survival_loss.name from a checkpoint's training config, defaulting to 'nll'."""
    cfg = training_cfg.model.get("survival_loss", {"name": "nll"})
    return str(cfg.get("name", "nll")).lower()


def run_split_inference(model, dataloader, dataset, device, split, fold):
    outputs = {
        "logits": [],
        "hazard": [],
        "pmf": [],
        "survival": [],
        "survival_time": [],
        "risk": [],
    }
    time_bins = []
    times = []
    events = []
    seen = 0

    model.eval()
    with torch.no_grad():
        for images, target in dataloader:
            images = images.to(device)
            target = _move_target_to_device(target, device)
            y_hat = model(images)
            batch_size = images.shape[0]
            for key in outputs:
                outputs[key].append(_tensor_to_numpy(y_hat[key]))
            time_bins.append(_tensor_to_numpy(target["time_bin"]).reshape(-1))
            times.append(_tensor_to_numpy(target.get("time", target["time_bin"])).reshape(-1))
            events.append(_tensor_to_numpy(target["event"]).reshape(-1))
            seen += batch_size

    logits = np.concatenate(outputs["logits"], axis=0)
    hazard = np.concatenate(outputs["hazard"], axis=0)
    pmf = np.concatenate(outputs["pmf"], axis=0)
    survival = np.concatenate(outputs["survival"], axis=0)
    survival_time = np.concatenate(outputs["survival_time"], axis=0).reshape(-1)
    risk_per_sample = np.concatenate(outputs["risk"], axis=0).reshape(-1)
    patient_ids = list(dataset.img_files[:seen])

    return {
        "patient_id": patient_ids,
        "split": split,
        "fold": fold,
        "time_bin": np.concatenate(time_bins).astype(int),
        "time": np.concatenate(times),
        "event": np.concatenate(events).astype(int),
        "logits": logits,
        "hazards": hazard,
        "pmf": pmf,
        "survival": survival,
        "predicted_survival_time": survival_time,
        # Cox-mode "risk" is the head scalar; NLL/DeepHit fall back to -∑survival.
        "risk_scalar": risk_per_sample,
        "risk": -survival.sum(axis=1),
    }


def _mean_se(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))


def km_survival_at(times, events, horizon):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    if len(times) == 0:
        return float("nan"), float("nan")

    survival = 1.0
    greenwood = 0.0
    for event_time in sorted(set(times[events & (times <= horizon)])):
        at_risk = np.sum(times >= event_time)
        observed = np.sum((times == event_time) & events)
        if at_risk <= 0:
            continue
        survival *= 1.0 - observed / at_risk
        if at_risk > observed:
            greenwood += observed / (at_risk * (at_risk - observed))
    se = survival * math.sqrt(greenwood) if greenwood > 0 else 0.0
    return float(survival), float(se)


def km_step_curve(times, events):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    if len(times) == 0:
        return np.array([0.0]), np.array([np.nan])
    event_times = sorted(set(times[events]))
    x = [0.0]
    y = [1.0]
    survival = 1.0
    for event_time in event_times:
        at_risk = np.sum(times >= event_time)
        observed = np.sum((times == event_time) & events)
        if at_risk <= 0:
            continue
        x.extend([event_time, event_time])
        y.extend([survival, survival * (1.0 - observed / at_risk)])
        survival = y[-1]
    x.append(float(np.max(times)))
    y.append(survival)
    return np.asarray(x), np.asarray(y)


def write_matrix_csv(path, patient_ids, matrix, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id"] + columns)
        for patient_id, row in zip(patient_ids, matrix):
            writer.writerow([patient_id] + [float(value) for value in row])


def write_predictions(outputs, out_dir, bin_columns, landmark_map, cutoff, survival_loss_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    risk_groups = np.where(outputs["risk"] >= cutoff, "high", "low")
    header = [
        "patient_id",
        "split",
        "fold",
        "time",
        "time_bin",
        "event",
        "predicted_survival_time",
        "risk",
        "risk_group",
    ]
    years = sorted(landmark_map)

    has_curve = survival_loss_name in ("nll", "deephit")
    if has_curve:
        header.extend([
            f"pred_risk_{int(year) if year.is_integer() else year:g}y" for year in years
        ])

    with (out_dir / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, patient_id in enumerate(outputs["patient_id"]):
            row = [
                patient_id,
                outputs["split"],
                outputs["fold"],
                float(outputs["time"][idx]),
                int(outputs["time_bin"][idx]),
                int(outputs["event"][idx]),
                (float(outputs["predicted_survival_time"][idx]) if has_curve else float("nan")),
                float(outputs["risk"][idx]),
                risk_groups[idx],
            ]
            if has_curve:
                row.extend([
                    float(1.0 - outputs["survival"][idx, landmark_map[year]]) for year in years
                ])
            writer.writerow(row)

    if survival_loss_name == "nll":
        write_matrix_csv(out_dir / "logits.csv", outputs["patient_id"], outputs["logits"], bin_columns)
        write_matrix_csv(out_dir / "hazards.csv", outputs["patient_id"], outputs["hazards"], bin_columns)
        write_matrix_csv(out_dir / "survival_curves.csv", outputs["patient_id"], outputs["survival"], bin_columns)
    elif survival_loss_name == "deephit":
        write_matrix_csv(out_dir / "logits.csv", outputs["patient_id"], outputs["logits"], bin_columns)
        write_matrix_csv(out_dir / "pmf.csv", outputs["patient_id"], outputs["pmf"], bin_columns)
        write_matrix_csv(out_dir / "survival_curves.csv", outputs["patient_id"], outputs["survival"], bin_columns)
    elif survival_loss_name == "cox":
        # Cox produces only a scalar risk; no curve outputs.
        pass
    else:
        raise ValueError(f"Unknown survival_loss_name: {survival_loss_name!r}")


def compute_metrics(outputs, survival_loss_name):
    metrics = {
        "c_index": concordance_index(outputs["time"], outputs["risk"], outputs["event"]),
        "n_patients": len(outputs["patient_id"]),
        "n_events": int(np.sum(outputs["event"])),
    }
    if survival_loss_name in ("nll", "deephit"):
        metrics["ibs"] = integrated_brier_score(
            torch.as_tensor(outputs["survival"]),
            torch.as_tensor(outputs["time_bin"]),
            torch.as_tensor(outputs["event"]),
        )
    else:
        metrics["ibs"] = float("nan")
    return metrics


def plot_km_high_low(outputs, cutoff, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    high = outputs["risk"] >= cutoff
    plt.figure(figsize=(6, 4))
    for label, mask, color in [
        ("Low risk", ~high, "#1f77b4"),
        ("High risk", high, "#d62728"),
    ]:
        if np.sum(mask) == 0:
            continue
        x, y = km_step_curve(outputs["time"][mask], outputs["event"][mask])
        plt.step(x, y, where="post", label=f"{label} (n={np.sum(mask)})", color=color)
    plt.xlabel("Time (years)")
    plt.ylabel("Kaplan-Meier survival")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def landmark_rows(outputs, landmark_map, cutoff):
    rows = []
    masks = {
        "all": np.ones(len(outputs["patient_id"]), dtype=bool),
        "low": outputs["risk"] < cutoff,
        "high": outputs["risk"] >= cutoff,
    }
    for group, mask in masks.items():
        for year, idx in sorted(landmark_map.items()):
            pred_risk = 1.0 - outputs["survival"][mask, idx]
            pred_mean, pred_se = _mean_se(pred_risk)
            km_survival, km_survival_se = km_survival_at(
                outputs["time"][mask],
                outputs["event"][mask],
                year,
            )
            rows.append(
                {
                    "group": group,
                    "year": int(year) if float(year).is_integer() else year,
                    "predicted_mean_risk": pred_mean,
                    "predicted_se": pred_se,
                    "km_risk": float(1.0 - km_survival) if not math.isnan(km_survival) else float("nan"),
                    "km_se": km_survival_se,
                    "n_patients": int(np.sum(mask)),
                    "n_events": int(np.sum(outputs["event"][mask])),
                }
            )
    return rows


def write_landmark_risks(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "group",
        "year",
        "predicted_mean_risk",
        "predicted_se",
        "km_risk",
        "km_se",
        "n_patients",
        "n_events",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def plot_landmark_bars(rows, path):
    plot_rows = [row for row in rows if row["group"] == "all"]
    years = [str(row["year"]) for row in plot_rows]
    x = np.arange(len(years))
    width = 0.35
    pred = [row["predicted_mean_risk"] for row in plot_rows]
    pred_se = [row["predicted_se"] for row in plot_rows]
    km = [row["km_risk"] for row in plot_rows]
    km_se = [row["km_se"] for row in plot_rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, pred, width, yerr=pred_se, label="Predicted", color="#4c78a8", capsize=4)
    plt.bar(x + width / 2, km, width, yerr=km_se, label="KM actual", color="#f58518", capsize=4)
    plt.xticks(x, [f"{year}y" for year in years])
    plt.ylabel("Cumulative recurrence risk")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def load_model(training_cfg, checkpoint_path, device):
    if instantiate is None:
        raise ModuleNotFoundError("Hydra is required to instantiate the survival model.")
    model = instantiate(training_cfg.model)
    checkpoint = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def prediction_dataloader(datamodule, dataset):
    num_workers = int(getattr(datamodule, "num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=datamodule.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def run_fold(cfg, exp_dir, pred_dir, fold, device, metrics_rows, pooled_val_risks):
    if instantiate is None:
        raise ModuleNotFoundError("Hydra is required to instantiate the survival data module.")
    score, checkpoint_path = select_best_checkpoint(exp_dir, fold, cfg.checkpoint_glob)
    training_cfg = _load_training_cfg(exp_dir, cfg.splits_json, fold)
    model = load_model(training_cfg, checkpoint_path, device)
    datamodule = instantiate(training_cfg.data).module
    datamodule.setup("predict")

    num_bins = int(training_cfg.model.num_time_bins)
    survival_loss_name = _resolve_survival_loss_name(training_cfg)
    bin_columns = _bin_columns(cfg.time_bin_labels, num_bins)
    landmark_map = _landmark_indices(
        cfg.landmark_years,
        cfg.time_bin_upper_bounds_years,
        num_bins=num_bins,
    )

    fold_outputs = {}
    for split, dataloader, dataset in [
        ("val", prediction_dataloader(datamodule, datamodule.val_dataset), datamodule.val_dataset),
        ("test", prediction_dataloader(datamodule, datamodule.test_dataset), datamodule.test_dataset),
    ]:
        outputs = run_split_inference(model, dataloader, dataset, device, split, fold)
        if survival_loss_name == "cox":
            outputs["risk"] = outputs["risk_scalar"]
        if split == "val":
            cutoff = float(np.median(outputs["risk"]))
            pooled_val_risks.extend(outputs["risk"].tolist())
            cutoff_source = "fold_val_median"
            cutoff_path = pred_dir / f"fold_{fold}" / "risk_cutoff.json"
            cutoff_path.parent.mkdir(parents=True, exist_ok=True)
            cutoff_path.write_text(json.dumps({"cutoff": cutoff}, indent=2))
        else:
            cutoff = fold_outputs["val_cutoff"]
            cutoff_source = "fold_val_median"

        out_dir = pred_dir / f"fold_{fold}" / split
        write_predictions(outputs, out_dir, bin_columns, landmark_map, cutoff, survival_loss_name)
        if survival_loss_name != "cox":
            plot_km_high_low(outputs, cutoff, out_dir / "km_high_low.png")
            lmk_rows = landmark_rows(outputs, landmark_map, cutoff)
            write_landmark_risks(lmk_rows, out_dir / "landmark_risks.csv")
            plot_landmark_bars(lmk_rows, out_dir / "landmark_risk_bars.png")

        metrics = compute_metrics(outputs, survival_loss_name)
        metrics_rows.append(
            {
                "scope": "fold",
                "fold": fold,
                "split": split,
                "c_index": metrics["c_index"],
                "ibs": metrics["ibs"],
                "n_patients": metrics["n_patients"],
                "n_events": metrics["n_events"],
                "cutoff_source": cutoff_source,
                "cutoff": cutoff,
            }
        )
        fold_outputs[f"{split}_outputs"] = outputs
        if split == "val":
            fold_outputs["val_cutoff"] = cutoff

    return {
        "fold": fold,
        "checkpoint_path": checkpoint_path,
        "checkpoint_val_c_index": score,
        "training_cfg": training_cfg,
        "bin_columns": bin_columns,
        "landmark_map": landmark_map,
        **fold_outputs,
    }


def make_ensemble_outputs(fold_results, survival_loss_name):
    base = fold_results[0]["test_outputs"]

    if survival_loss_name == "cox":
        risk = np.stack(
            [result["test_outputs"]["risk"] for result in fold_results],
            axis=0,
        ).mean(axis=0)
        return {
            "patient_id": base["patient_id"],
            "split": "test",
            "fold": "ensemble",
            "time_bin": base["time_bin"],
            "time": base["time"],
            "event": base["event"],
            "risk": risk,
            "predicted_survival_time": np.full_like(risk, fill_value=np.nan, dtype=float),
        }

    if survival_loss_name == "deephit":
        pmf = np.stack(
            [result["test_outputs"]["pmf"] for result in fold_results],
            axis=0,
        ).mean(axis=0)
        survival = (1.0 - np.cumsum(pmf, axis=1)).clip(0.0, 1.0)
        survival_time = _tensor_to_numpy(survival_to_time(torch.as_tensor(survival))).reshape(-1)
        return {
            "patient_id": base["patient_id"],
            "split": "test",
            "fold": "ensemble",
            "time_bin": base["time_bin"],
            "time": base["time"],
            "event": base["event"],
            "pmf": pmf,
            "survival": survival,
            "predicted_survival_time": survival_time,
            "risk": -survival.sum(axis=1),
        }

    # nll (default) — unchanged behaviour
    logits = np.stack([result["test_outputs"]["logits"] for result in fold_results], axis=0).mean(axis=0)
    hazards = _tensor_to_numpy(logits_to_hazard(torch.as_tensor(logits)))
    survival = _tensor_to_numpy(hazard_to_survival(torch.as_tensor(hazards)))
    survival_time = _tensor_to_numpy(survival_to_time(torch.as_tensor(survival))).reshape(-1)
    return {
        "patient_id": base["patient_id"],
        "split": "test",
        "fold": "ensemble",
        "time_bin": base["time_bin"],
        "time": base["time"],
        "event": base["event"],
        "logits": logits,
        "hazards": hazards,
        "survival": survival,
        "predicted_survival_time": survival_time,
        "risk": -survival.sum(axis=1),
    }


def write_metrics_csv(metrics_rows, path):
    header = [
        "scope",
        "fold",
        "split",
        "c_index",
        "ibs",
        "n_patients",
        "n_events",
        "cutoff_source",
        "cutoff",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(metrics_rows)


def _inference_impl(cfg):
    exp_dir = _as_path(cfg.exp_dir)
    pred_dir = _as_path(cfg.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)

    metrics_rows = []
    pooled_val_risks = []
    fold_results = []
    for fold in cfg.folds:
        fold_results.append(
            run_fold(
                cfg=cfg,
                exp_dir=exp_dir,
                pred_dir=pred_dir,
                fold=int(fold),
                device=device,
                metrics_rows=metrics_rows,
                pooled_val_risks=pooled_val_risks,
            )
        )

    if len(fold_results) > 1:
        pooled_cutoff = float(np.median(np.asarray(pooled_val_risks, dtype=float)))
        cutoff_path = pred_dir / "ensemble" / "oof_validation_cutoff.json"
        cutoff_path.parent.mkdir(parents=True, exist_ok=True)
        cutoff_path.write_text(json.dumps({"pooled_oof_cutoff": pooled_cutoff}, indent=2))

        survival_loss_name = _resolve_survival_loss_name(fold_results[0]["training_cfg"])

        ensemble = make_ensemble_outputs(fold_results, survival_loss_name)
        out_dir = pred_dir / "ensemble" / "test"
        write_predictions(
            ensemble,
            out_dir,
            fold_results[0]["bin_columns"],
            fold_results[0]["landmark_map"],
            pooled_cutoff,
            survival_loss_name,
        )
        if survival_loss_name != "cox":
            plot_km_high_low(ensemble, pooled_cutoff, out_dir / "km_high_low.png")
            lmk_rows = landmark_rows(ensemble, fold_results[0]["landmark_map"], pooled_cutoff)
            write_landmark_risks(lmk_rows, out_dir / "landmark_risks.csv")
            plot_landmark_bars(lmk_rows, out_dir / "landmark_risk_bars.png")
        metrics = compute_metrics(ensemble, survival_loss_name)
        metrics_rows.append(
            {
                "scope": "ensemble",
                "fold": "all",
                "split": "test",
                "c_index": metrics["c_index"],
                "ibs": metrics["ibs"],
                "n_patients": metrics["n_patients"],
                "n_events": metrics["n_events"],
                "cutoff_source": "pooled_oof_validation_median",
                "cutoff": pooled_cutoff,
            }
        )

    write_metrics_csv(metrics_rows, pred_dir / "metrics.csv")


if hydra is not None:
    inference = hydra.main(
        version_base=None,
        config_path="./cli_configs",
        config_name="inference_survival",
    )(_inference_impl)
else:
    def inference(_cfg=None):
        raise ModuleNotFoundError("Hydra/OmegaConf is required for survival inference.")


if __name__ == "__main__":
    if make_omegaconf_resolvers is None:
        raise ModuleNotFoundError("OmegaConf is required for survival inference.")
    make_omegaconf_resolvers()
    inference()
