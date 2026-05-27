import math
import warnings

import lightning as L
import torch
import torch.nn as nn
import wandb
from madgrad import MADGRAD
from timm.optim import RMSpropTF
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    Precision,
    Recall,
)
from torchmetrics.aggregation import CatMetric
from metrics.balanced_accuracy import BalancedAccuracy
from augmentation.mixup import mixup_criterion, mixup_data
from metrics.conf_mat import ConfusionMatrix
from regularization.sam import SAM
from survival_utils import (
    _reject_legacy_cox_loss_lambda,
    build_survival_criterion,
    concordance_index,
    integrated_brier_score,
    integrated_brier_score_ipcw,
    time_dependent_auc,
)


_SURVIVAL_LOSS_TAGS = {
    "nll": "NLL",
    "cox": "CoxPH",
    "deephit": "DeepHit",
    "soft_logrank": "SoftLogRank",
}


class BaseModel(L.LightningModule):
    def __init__(
            self,
            task,
            metric_computation_mode,
            result_plot,
            metrics,
            num_classes,
            name,
            lr,
            weight_decay,
            optimizer,
            nesterov,
            sam,
            adaptive_sam,
            scheduler,
            T_max,
            warmstart,
            epochs,
            mixup,
            mixup_alpha,
            label_smoothing,
            stochastic_depth,
            resnet_dropout,
            squeeze_excitation,
            apply_shakedrop,
            undecay_norm,
            zero_init_residual,
            input_dim,
            input_channels,
            pretrained,
            *args,
            **kwargs
    ):
        super(BaseModel, self).__init__()

        # Task
        self.task = task

        # Metrics
        self.metric_computation_mode = metric_computation_mode
        self.result_plot_setting = result_plot
        metrics_dict = {}

        self.subtask = kwargs.get("subtask")

        if self.subtask == "multiclass":
            metric_task = "multiclass"
        elif self.subtask == "multilabel":
            metric_task = "multilabel"
        else:
            metric_task = None

        if self.task == "Classification":
            if metric_task is None:
                raise ValueError(
                    "Classification requires subtask to be 'multiclass' or 'multilabel'."
                )
            if "acc" in metrics:
                metrics_dict["Accuracy"] = Accuracy(
                    task=metric_task,
                    num_classes=num_classes,
                    num_labels=num_classes,
                )
            if "balanced_acc" in metrics:
                if "balanced_acc" in metrics:
                    metrics_dict["Balanced_Accuracy"] = BalancedAccuracy(
                        task=metric_task,
                        num_classes=num_classes,
                    )

            if "f1" in metrics:
                metrics_dict["F1"] = F1Score(
                    average="macro",
                    num_classes=num_classes,
                    task=metric_task,
                    num_labels=num_classes,
                )
            if "f1_per_class" in metrics:
                metrics_dict["F1_per_class"] = F1Score(
                    average=None,
                    num_classes=num_classes,
                    task=metric_task,
                    num_labels=num_classes,
                )
            if "pr" in metrics:
                metrics_dict["Precision"] = Precision(
                    average="macro",
                    num_classes=num_classes,
                    task=metric_task,
                    num_labels=num_classes,
                )
                metrics_dict["Recall"] = Recall(
                    average="macro",
                    num_classes=num_classes,
                    task=metric_task,
                    num_labels=num_classes,
                )
            if "top5acc" in metrics:
                metrics_dict["Accuracy_top5"] = Accuracy(
                    task=metric_task,
                    num_classes=num_classes,
                    top_k=5,
                    num_labels=num_classes,
                )
            if "auroc" in metrics:
                metrics_dict["AUROC"] = AUROC(
                    average="macro",
                    task=metric_task,
                    num_classes=num_classes,
                    num_labels=num_classes,
                )
            if "ap" in metrics:
                metrics_dict["AP"] = AveragePrecision(
                    task=metric_task,
                    num_classes=num_classes,
                    num_labels=num_classes,
                )

        elif self.task == "Regression":
            if "mse" in metrics:
                metrics_dict["MSE"] = MeanSquaredError()
            if "mae" in metrics:
                metrics_dict["MAE"] = MeanAbsoluteError()
        elif self.task == "Survival":
            if metrics:
                warnings.warn(
                    "Configured survival metrics are ignored; loss, C-index, and predictions are logged."
                )
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        if self.result_plot_setting in ["val", "all"]:
            if self.task == "Classification":
                self.val_conf_mat = ConfusionMatrix(num_classes=num_classes)
            elif self.task == "Regression":
                self.val_pred_list = []
                self.val_label_list = []
        if self.result_plot_setting == "all":
            if self.task == "Classification":
                self.train_conf_mat = ConfusionMatrix(num_classes=num_classes)
            elif self.task == "Regression":
                self.train_pred_list = []
                self.train_label_list = []

        self.save_preds = True if kwargs["save_preds"] else False
        if self.save_preds:
            self.val_preds = CatMetric(dist_sync_on_step=False)
            self.val_labels = CatMetric(dist_sync_on_step=False)
            self.val_indices = CatMetric(dist_sync_on_step=False)

        self.has_metrics = len(metrics_dict) > 0
        metrics = MetricCollection(metrics_dict)
        self.train_metrics = metrics.clone(prefix="Train/")
        self.val_metrics = metrics.clone(prefix="Val/")

        # Training Args
        self.name = name
        # self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.nesterov = nesterov
        self.sam = sam
        self.adaptive_sam = adaptive_sam
        self.scheduler = scheduler
        self.T_max = T_max
        self.warmstart = warmstart
        self.warmstart2 = kwargs["warmstart2"]
        self.epochs = epochs
        self.pretrained = pretrained

        # Regularization techniques
        self.mixup = mixup
        if self.task == "Survival" and self.mixup:
            raise ValueError("mixup=True is not supported for Survival tasks.")
        self.mixup_alpha = mixup_alpha  # 0.2
        self.label_smoothing = label_smoothing  # 0.1
        self.stochastic_depth = (
            stochastic_depth  # 0.1 (with higher resolution maybe 0.2)
        )
        self.resnet_dropout = resnet_dropout  # 0.5
        self.se = squeeze_excitation
        self.apply_shakedrop = apply_shakedrop
        self.undecay_norm = undecay_norm
        self.zero_init_residual = zero_init_residual

        # Finetuning method
        self.finetuning_method = kwargs["finetune_method"]

        # Data and Dataloading
        self.input_dim = input_dim
        self.input_channels = input_channels
        self.num_classes = num_classes

        # switch to manual optimization for Sharpness Aware Minimization
        if self.sam:
            self.automatic_optimization = False

        # Loss
        if self.task == "Classification":
            if self.subtask == "multiclass":
                self.criterion = nn.CrossEntropyLoss(
                    label_smoothing=self.label_smoothing
                )
            elif self.subtask == "multilabel":
                self.criterion = nn.BCEWithLogitsLoss()
        elif self.task == "Regression":
            self.criterion = nn.MSELoss()
        elif self.task == "Survival":
            _reject_legacy_cox_loss_lambda(kwargs)

            self.num_time_bins = int(kwargs.get("num_time_bins", num_classes))

            survival_loss_cfg = kwargs.get("survival_loss")
            if survival_loss_cfg is None:
                survival_loss_cfg = {"name": "nll"}
            self.survival_loss_name, self.criterion = build_survival_criterion(
                survival_loss_cfg, num_time_bins=self.num_time_bins,
            )

            default_cut_points_years = (
                [1.0, 2.0, 3.0, 5.0]
                if self.num_time_bins == 5
                else [float(i) for i in range(1, self.num_time_bins)]
            )
            cut_points_years = self._survival_year_values(
                kwargs.get("survival_cut_points_years"),
                kwargs.get("survival_cut_points_months"),
                default=default_cut_points_years,
            )
            if len(cut_points_years) != self.num_time_bins - 1:
                raise ValueError(
                    "survival_cut_points_years must contain num_time_bins - 1 "
                    f"values. Got {len(cut_points_years)} cut points for "
                    f"{self.num_time_bins} bins."
                )
            landmark_years = self._survival_year_values(
                kwargs.get("survival_landmark_years"),
                None,
                default=cut_points_years,
            )
            self.register_buffer(
                "survival_cut_points_years",
                torch.tensor(cut_points_years, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "survival_landmark_years",
                torch.tensor(landmark_years, dtype=torch.float32),
                persistent=False,
            )
            self.survival_landmark_labels = [
                self._format_survival_landmark_label(value)
                for value in landmark_years
            ]
            self.train_survival_risks = []
            self.train_survival_curves = []
            self.train_survival_time_bins = []
            self.train_survival_continuous_times = []
            self.train_survival_events = []
            self.val_survival_risks = []
            self.val_survival_curves = []
            self.val_survival_time_bins = []
            self.val_survival_continuous_times = []
            self.val_survival_events = []
            self.survival_smoothing_alpha = float(
                kwargs.get(
                    "survival_smoothing_alpha",
                    kwargs.get("survival_ema_alpha", 0.05),
                )
            )
            self.survival_smoothing_alpha = min(
                max(self.survival_smoothing_alpha, 0.0),
                1.0,
            )
            self.survival_metric_ema = {
                "Train/loss": None,
                "Val/loss": None,
                "Train/C-index": None,
                "Val/C-index": None,
            }
            self.train_survival_losses = []
            self.val_survival_losses = []

    def forward(self, x):
        pass

    def _survival_year_values(self, values, month_values=None, default=None):
        if values is None and month_values is not None:
            values = [float(value) / 12.0 for value in month_values]
        if values is None:
            values = default or []
        return [float(value) for value in values]

    def _format_survival_landmark_label(self, value):
        value = float(value)
        if value.is_integer():
            return f"{int(value)}y"
        return f"{value:g}y".replace(".", "p")

    def _time_to_survival_bin(self, continuous_time):
        cut_points = self.survival_cut_points_years.to(continuous_time.device)
        time_bin = torch.bucketize(
            continuous_time.float(),
            cut_points,
            right=False,
        )
        return time_bin.clamp(0, self.num_time_bins - 1).long()

    def _unpack_survival_targets(self, y):
        if isinstance(y, dict):
            event = y["event"]
            if "time" in y:
                continuous_time = y["time"]
                time_bin = None
            elif "time_years" in y:
                continuous_time = y["time_years"]
                time_bin = None
            elif "time_months" in y:
                continuous_time = y["time_months"].float() / 12.0
                time_bin = None
            elif "time_bin" in y:
                continuous_time = y["time_bin"]
                time_bin = y["time_bin"].to(
                    device=self.device,
                    dtype=torch.long,
                ).view(-1)
            else:
                raise KeyError(
                    "Survival targets must contain 'time', 'time_years', "
                    "'time_months', or legacy 'time_bin'."
                )
        else:
            time_bin, event = y
            continuous_time = time_bin
            time_bin = time_bin.to(device=self.device, dtype=torch.long).view(-1)

        continuous_time = continuous_time.to(
            device=self.device,
            dtype=torch.float32,
        ).view(-1)
        if time_bin is None:
            time_bin = self._time_to_survival_bin(continuous_time)

        return (
            time_bin,
            event.to(device=self.device, dtype=torch.float32).view(-1),
            continuous_time,
        )

    def _survival_label_tensor(self, time_bin, event):
        return torch.stack([time_bin.float(), event.float()], dim=1)

    def _survival_loss(self, y_hat, y):
        time_bin, event, continuous_time = self._unpack_survival_targets(y)
        name = self.survival_loss_name

        if name == "nll":
            loss = self.criterion(y_hat["logits"], time_bin, event)
            loss_parts = {"total": loss, name: loss}
        elif name == "cox":
            loss = self.criterion(y_hat["risk"], continuous_time, event)
            loss_parts = {"total": loss, name: loss}
        elif name == "deephit":
            loss = self.criterion(y_hat["pmf"], time_bin, event)
            loss_parts = {"total": loss, name: loss}
        elif name == "soft_logrank":
            total, components = self.criterion(
                y_hat["p_high"],
                continuous_time,
                event,
            )
            loss_parts = {"total": total, name: total, **components}
        else:
            raise ValueError(f"Unexpected survival_loss_name: {name!r}")

        return loss_parts, time_bin, event, continuous_time

    def _update_survival_metric_buffers(
        self,
        split,
        y_hat,
        time_bin,
        event,
        continuous_time,
    ):
        if self.survival_loss_name in ("cox", "soft_logrank"):
            # Cox: risk = log hazard ratio. soft_logrank: risk = stratification
            # logit (sigmoid → p_high). Both are monotonic continuous risk
            # scores suitable for C-index. No calibrated survival curve.
            risk = y_hat["risk"].detach()
            survival = None
        else:
            risk = -y_hat["survival_time"].detach()
            survival = y_hat["survival"].detach()
        if split == "train":
            self.train_survival_risks.append(risk)
            if survival is not None:
                self.train_survival_curves.append(survival)
            self.train_survival_time_bins.append(time_bin.detach())
            self.train_survival_continuous_times.append(continuous_time.detach())
            self.train_survival_events.append(event.detach())
        elif split == "val":
            self.val_survival_risks.append(risk)
            if survival is not None:
                self.val_survival_curves.append(survival)
            self.val_survival_time_bins.append(time_bin.detach())
            self.val_survival_continuous_times.append(continuous_time.detach())
            self.val_survival_events.append(event.detach())

    def _to_metric_float(self, value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().mean().cpu().item())
        return float(value)

    def _log_smoothed_survival_metric(self, metric_name, value, prog_bar=False):
        if getattr(self.trainer, "sanity_checking", False):
            return
        value = self._to_metric_float(value)
        previous = self.survival_metric_ema[metric_name]
        if previous is None:
            smoothed_value = value
        else:
            alpha = self.survival_smoothing_alpha
            smoothed_value = alpha * value + (1.0 - alpha) * previous
        self.survival_metric_ema[metric_name] = smoothed_value
        self.log(
            f"{metric_name}_smoothed",
            smoothed_value,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            logger=True,
            sync_dist=True,
        )

    def _log_smoothed_survival_loss(self, split):
        if split == "train":
            losses = self.train_survival_losses
            metric_name = "Train/loss"
        else:
            losses = self.val_survival_losses
            metric_name = "Val/loss"

        if not losses:
            return

        mean_loss = torch.stack(losses).mean()
        self._log_smoothed_survival_metric(metric_name, mean_loss, prog_bar=True)
        losses.clear()

    def _log_survival_metrics(self, split):
        if split == "train":
            risks = self.train_survival_risks
            survival_curves = self.train_survival_curves
            time_bins = self.train_survival_time_bins
            continuous_times = self.train_survival_continuous_times
            events = self.train_survival_events
            c_index_metric_name = "Train/C-index"
            brier_metric_name = "Train/Brier"
            brier_ipcw_metric_name = "Train/Brier-IPCW"
        else:
            risks = self.val_survival_risks
            survival_curves = self.val_survival_curves
            time_bins = self.val_survival_time_bins
            continuous_times = self.val_survival_continuous_times
            events = self.val_survival_events
            c_index_metric_name = "Val/C-index"
            brier_metric_name = "Val/Brier"
            brier_ipcw_metric_name = "Val/Brier-IPCW"

        if not risks:
            return

        all_time_bins = torch.cat(time_bins)
        all_continuous_times = torch.cat(continuous_times)
        all_events = torch.cat(events)
        c_index = concordance_index(
            all_continuous_times,
            torch.cat(risks),
            all_events,
        )
        self.log(
            c_index_metric_name,
            c_index,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self._log_smoothed_survival_metric(
            c_index_metric_name,
            c_index,
            prog_bar=True,
        )

        if survival_curves:
            all_survival_curves = torch.cat(survival_curves)
            brier = integrated_brier_score(
                all_survival_curves,
                all_time_bins,
                all_events,
            )
            brier_ipcw = integrated_brier_score_ipcw(
                all_survival_curves,
                all_time_bins,
                all_events,
            )
            landmark_aucs = time_dependent_auc(
                all_survival_curves,
                all_continuous_times,
                all_events,
                self.survival_landmark_years,
                self.survival_cut_points_years,
            )
            valid_landmark_aucs = []
            self.log(
                brier_metric_name,
                brier,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                brier_ipcw_metric_name,
                brier_ipcw,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            for landmark, label in zip(
                self.survival_landmark_years.detach().cpu().tolist(),
                self.survival_landmark_labels,
            ):
                auc = landmark_aucs.get(float(landmark), float("nan"))
                if math.isnan(auc):
                    continue
                valid_landmark_aucs.append(auc)
                self.log(
                    f"{split.capitalize()}/AUC@{label}",
                    auc,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
            if valid_landmark_aucs:
                self.log(
                    f"{split.capitalize()}/mean_AUC_landmarks",
                    sum(valid_landmark_aucs) / len(valid_landmark_aucs),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

        risks.clear()
        survival_curves.clear()
        time_bins.clear()
        continuous_times.clear()
        events.clear()

    def training_step(self, batch, batch_idx):

        x, y = batch

        if self.task == "Survival":
            y_hat = self(x)

            if self.sam:
                opt = self.optimizers()
                loss_parts, time_bin, event, continuous_time = self._survival_loss(
                    y_hat,
                    y,
                )
                loss = loss_parts["total"]
                self.manual_backward(loss)
                opt.first_step(zero_grad=True)

                second_loss_parts, _, _, _ = self._survival_loss(self(x), y)
                second_loss = second_loss_parts["total"]
                self.manual_backward(second_loss)
                opt.second_step(zero_grad=True)
            else:
                loss_parts, time_bin, event, continuous_time = self._survival_loss(
                    y_hat,
                    y,
                )
                loss = loss_parts["total"]

            self._update_survival_metric_buffers(
                "train",
                y_hat,
                time_bin,
                event,
                continuous_time,
            )
            self.train_survival_losses.append(loss.detach())

            self.log(
                "Train/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"Train/{_SURVIVAL_LOSS_TAGS[self.survival_loss_name]}Loss",
                loss_parts[self.survival_loss_name],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

            if torch.isnan(y_hat["logits"]).any():
                print("######################################### Model predicts NaNs!")

            return loss

        if self.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(x, y, alpha=self.mixup_alpha)
            y_hat = self(inputs)

        else:
            y_hat = self(x)
            if self.num_classes == 1:
                y_hat = y_hat.view(-1)

        if x.shape[0] == 1 and len(y_hat.shape) == 1:
            # for cases where batch size is 1 and y_hat doesn't have a batch dim
            y_hat = y_hat.unsqueeze(0)

        if self.sam:
            opt = self.optimizers()

            # first forward-backward pass
            if self.mixup:
                loss = mixup_criterion(self.criterion, y_hat, targets_a, targets_b, lam)
            else:
                loss = self.criterion(y_hat, y)
            self.manual_backward(loss)
            opt.first_step(zero_grad=True)

            # second forward-backward pass
            if self.mixup:
                self.manual_backward(
                    mixup_criterion(
                        self.criterion, self(inputs), targets_a, targets_b, lam
                    )
                )
            else:
                if self.num_classes == 1:
                    self.manual_backward(self.criterion(self(x).view(-1), y))
                else:
                    self.manual_backward(self.criterion(self(x), y))
            opt.second_step(zero_grad=True)

        else:
            if self.mixup:
                loss = mixup_criterion(self.criterion, y_hat, targets_a, targets_b, lam)
            else:
                loss = self.criterion(
                    y_hat, y.float() if self.subtask == "multilabel" else y
                )

        self.log(
            "Train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if torch.isnan(y_hat).any():
            print("######################################### Model predicts NaNs!")

        # save metrics
        if self.has_metrics and self.metric_computation_mode == "stepwise":
            metrics_res = self.train_metrics(y_hat, y)
            if "Train/F1_per_class" in metrics_res.keys():
                for i, value in enumerate(metrics_res["Train/F1_per_class"]):
                    metrics_res["Train/F1_class_{}".format(i)] = (
                        value if not torch.isnan(value) else 0.0
                    )
                del metrics_res["Train/F1_per_class"]
            self.log_dict(
                metrics_res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        elif self.has_metrics and self.metric_computation_mode == "epochwise":
            if self.task == "Classification":
                if self.subtask == "multilabel":
                    self.train_metrics.update(torch.sigmoid(y_hat.detach()), y)
                elif self.subtask == "multiclass":
                    self.train_metrics.update(F.softmax(y_hat.detach(), dim=-1), y)
            else:
                self.train_metrics.update(y_hat.detach(), y)

        if hasattr(self, "train_conf_mat"):
            self.train_conf_mat.update(y_hat, y)
        if hasattr(self, "train_pred_list"):
            self.train_pred_list.extend(y_hat)
            self.train_label_list.extend(y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if self.task == "Survival":
            val_loss_parts, time_bin, event, continuous_time = self._survival_loss(
                y_hat,
                y,
            )
            val_loss = val_loss_parts["total"]
            self._update_survival_metric_buffers(
                "val",
                y_hat,
                time_bin,
                event,
                continuous_time,
            )
            self.val_survival_losses.append(val_loss.detach())
            self.log(
                "Val/loss",
                val_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"Val/{_SURVIVAL_LOSS_TAGS[self.survival_loss_name]}Loss",
                val_loss_parts[self.survival_loss_name],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

            if hasattr(self, "val_preds"):
                actual_batch_size = x.size(0)
                start_idx = batch_idx * self.trainer.val_dataloaders.batch_size
                idx = torch.arange(
                    start_idx, start_idx + actual_batch_size, device=self.device
                )
                if self.survival_loss_name in ("cox", "soft_logrank"):
                    self.val_preds.update(y_hat["risk"].detach())
                else:
                    self.val_preds.update(y_hat["survival_time"].detach())
                self.val_labels.update(
                    self._survival_label_tensor(time_bin, event).detach()
                )
                self.val_indices.update(idx)

            return val_loss

        if self.num_classes == 1:
            y_hat = y_hat.view(-1)

        val_loss = self.criterion(
            y_hat, y.float() if self.subtask == "multilabel" else y
        )
        self.log(
            "Val/loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,  # True if self.trainer.num_devices > 1 else False,
        )

        # save metrics
        if self.has_metrics and self.metric_computation_mode == "stepwise":
            metrics_res = self.val_metrics(y_hat, y)
            if "Val/F1_per_class" in metrics_res.keys():
                for i, value in enumerate(metrics_res["Val/F1_per_class"]):
                    metrics_res["Val/F1_class_{}".format(i)] = (
                        value if not torch.isnan(value) else 0.0
                    )
                del metrics_res["Val/F1_per_class"]
            self.log_dict(
                metrics_res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,  # True if self.trainer.num_devices > 1 else False,
            )
        elif self.has_metrics and self.metric_computation_mode == "epochwise":
            if self.task == "Classification":
                if self.subtask == "multilabel":
                    self.val_metrics.update(torch.sigmoid(y_hat.detach()), y)
                elif self.subtask == "multiclass":
                    self.val_metrics.update(F.softmax(y_hat.detach(), dim=-1), y)
            else:
                self.val_metrics.update(y_hat.detach(), y)


        if hasattr(self, "val_conf_mat"):
            self.val_conf_mat.update(y_hat, y)
        if hasattr(self, "val_preds"):
            """self.val_pred_list.extend(y_hat.detach().cpu())
            self.val_label_list.extend(y.detach().cpu())"""
            actual_batch_size = x.size(0)  # dynamic size (works for last batch)
            start_idx = batch_idx * self.trainer.val_dataloaders.batch_size
            idx = torch.arange(
                start_idx, start_idx + actual_batch_size, device=self.device
            )

            self.val_preds.update(y_hat.detach())
            self.val_labels.update(y.detach())
            self.val_indices.update(idx)

    def predict_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)

        if self.task == "Survival":
            time_bin, event, _ = self._unpack_survival_targets(y)
            return self._survival_label_tensor(time_bin, event), y_hat

        if self.num_classes == 1:
            y_hat = y_hat.view(-1)

        # self.predictions.append(y_hat)
        return y, y_hat

    def on_validation_epoch_end(self) -> None:
        if self.task == "Survival":
            self._log_smoothed_survival_loss("val")
            self._log_survival_metrics("val")

        if self.has_metrics and self.metric_computation_mode == "epochwise":
            metrics_res = self.val_metrics.compute()
            if "Val/F1_per_class" in metrics_res.keys():
                for i, value in enumerate(metrics_res["Val/F1_per_class"]):
                    metrics_res["Val/F1_class_{}".format(i)] = (
                        value if not torch.isnan(value) else 0.0
                    )
                del metrics_res["Val/F1_per_class"]
            self.log_dict(
                metrics_res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,  # True if self.trainer.num_devices > 1 else False,
            )

            self.val_metrics.reset()

        if hasattr(self, "val_conf_mat"):
            self.val_conf_mat.save_state(self, "val")
            self.val_conf_mat.reset()
        if hasattr(self, "val_preds"):
            """# Stack tensors along batch dim
            val_preds = torch.stack(self.val_pred_list, dim=0).to(self.device)
            val_labels = torch.stack(self.val_label_list, dim=0).to(self.device)
            # print(len(self.val_pred_list), val_preds.shape)
            # Gather from all GPUs
            preds_all = self.all_gather(val_preds)
            preds_all = preds_all.view(-1, *preds_all.shape[2:])
            labels_all = self.all_gather(val_labels)
            labels_all = labels_all.view(-1, *labels_all.shape[2:])
            # print(preds_all.shape)"""

            preds_all = self.val_preds.compute()  # shape: [N_total, C]
            labels_all = self.val_labels.compute()
            indices = self.val_indices.compute()

            if self.trainer.is_global_zero:
                # Sort by original index to preserve dataset order
                sorted_idx = torch.argsort(indices)
                preds_all = preds_all[sorted_idx]
                labels_all = labels_all[sorted_idx]
                if self.task == "Regression":
                    data = [[x, y] for (x, y) in zip(labels_all, preds_all)]
                    table = wandb.Table(
                        data=data, columns=["Ground Truth", "Prediction"]
                    )
                    wandb.log(
                        {
                            "Val Scatterplot": wandb.plot.scatter(
                                table,
                                "Ground Truth",
                                "Prediction",
                                "Validation Scatterplot",
                            )
                        }
                    )
                if self.save_preds:

                    if self.task == "Survival":
                        if self.survival_loss_name == "soft_logrank":
                            columns = [
                                "GT_time_bin",
                                "GT_event",
                                "Pred_logit",
                                "Pred_p_high",
                                "Pred_group",
                            ]
                            data = []
                            for x, y_val in zip(labels_all, preds_all):
                                logit = float(y_val.item())
                                p = 1.0 / (1.0 + math.exp(-logit))
                                data.append([
                                    x[0].item(),
                                    x[1].item(),
                                    logit,
                                    p,
                                    "high" if p >= 0.5 else "low",
                                ])
                        elif self.survival_loss_name == "cox":
                            columns = [
                                "GT_time_bin",
                                "GT_event",
                                "Pred_risk",
                            ]
                            data = [
                                [x[0].item(), x[1].item(), y.item()]
                                for x, y in zip(labels_all, preds_all)
                            ]
                        else:
                            columns = [
                                "GT_time_bin",
                                "GT_event",
                                "Pred_survival_time",
                                "Pred_risk",
                            ]
                            data = [
                                [x[0].item(), x[1].item(), y.item(), -y.item()]
                                for x, y in zip(labels_all, preds_all)
                            ]
                        table = wandb.Table(data=data, columns=columns)
                        wandb.log({"Val Predictions": table})
                    elif self.task == "Classification":
                        columns = (
                                      (["GT_" + str(i) for i in range(len(labels_all[0]))])
                                      if self.subtask == "multilabel"
                                      else ["GT"]
                                  ) + ["Pred_" + str(i) for i in range(len(preds_all[0]))]
                        data = [
                            (
                                    (x.tolist() if self.subtask == "multilabel" else [x])
                                    + (
                                        F.softmax(y, dim=-1)
                                        if self.subtask == "multiclass"
                                        else torch.sigmoid(y)
                                    ).tolist()
                            )
                            for x, y in zip(labels_all, preds_all)
                        ]
                        table = wandb.Table(data=data, columns=columns)
                        wandb.log({"Val Predictions": table})
                    else:
                        raise NotImplementedError

            # reset
            self.val_preds.reset()
            self.val_labels.reset()
            self.val_indices.reset()

    def on_train_epoch_end(self) -> None:
        if self.task == "Survival":
            self._log_smoothed_survival_loss("train")
            self._log_survival_metrics("train")

        if self.has_metrics and self.metric_computation_mode == "epochwise":
            metrics_res = self.train_metrics.compute()
            if "Train/F1_per_class" in metrics_res.keys():
                for i, value in enumerate(metrics_res["Train/F1_per_class"]):
                    metrics_res["Train/F1_class_{}".format(i)] = (
                        value if not torch.isnan(value) else 0.0
                    )
                del metrics_res["Train/F1_per_class"]

            self.log_dict(
                metrics_res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,  # True if self.trainer.num_devices > 1 else False,
            )

            self.train_metrics.reset()

        if hasattr(self, "train_conf_mat"):
            self.train_conf_mat.save_state(self, "train")
            self.train_conf_mat.reset()
        if hasattr(self, "train_pred_list"):
            data = [
                [x, y] for (x, y) in zip(self.train_label_list, self.train_pred_list)
            ]
            table = wandb.Table(data=data, columns=["Ground Truth", "Prediction"])
            wandb.log(
                {
                    "Train Scatterplot": wandb.plot.scatter(
                        table, "Ground Truth", "Prediction", "Train Scatterplot"
                    )
                }
            )
            # reset
            self.train_pred_list = []
            self.train_label_list = []

    def on_train_start(self):
        # from models.preact_resnet import PreActBlock, PreActBottleneck
        # from models.pyramidnet import BasicBlock as BasicBlock_pyramid
        # from models.pyramidnet import Bottleneck as Bottleneck_pyramid
        # from models.resnet import BasicBlock, Bottleneck
        # from models.wide_resnet import BasicBlock as Wide_BasicBlock
        # from models.wide_resnet import Bottleneck as Wide_Bottleneck

        if not self.pretrained:
            print("Initializing weights")
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    # nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1e-3)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            # TODO
            # if self.zero_init_residual:
            #     if "PreAct" in self.name:
            #         for m in self.modules():
            #             if isinstance(m, PreActBottleneck):
            #                 nn.init.constant_(m.conv3.weight, 0)
            #             elif isinstance(m, PreActBlock):
            #                 nn.init.constant_(m.conv2.weight, 0)
            #
            #     elif "ResNet" in self.name or "WRN" in self.name:
            #         for m in self.modules():
            #             if isinstance(m, Bottleneck) or isinstance(m, Wide_Bottleneck):
            #                 nn.init.constant_(m.bn3.weight, 0)
            #             elif isinstance(m, BasicBlock) or isinstance(
            #                     m, Wide_BasicBlock
            #             ):
            #                 nn.init.constant_(m.bn2.weight, 0)
            #
            #     elif "Pyramid" in self.name:
            #         for m in self.modules():
            #             if isinstance(m, Bottleneck_pyramid):
            #                 nn.init.constant_(m.bn4.weight, 0)
            #             elif isinstance(m, BasicBlock_pyramid):
            #                 nn.init.constant_(m.bn3.weight, 0)

    def configure_optimizers(self):
        # leave bias and params of batch norm undecayed as in https://arxiv.org/pdf/1812.01187.pdf (Bag of tricks)
        if self.undecay_norm:
            model_params = []
            norm_params = []
            for name, p in self.named_parameters():
                if p.requires_grad:
                    if "norm" in name or "bias" in name or "bn" in name:
                        norm_params += [p]
                    else:
                        model_params += [p]
            params = [
                {"params": model_params},
                {"params": norm_params, "weight_decay": 0},
            ]
        else:
            params = self.parameters()

        if self.finetuning_method == "full_sawtooth":
            # Separate encoder and prediction head parameters.
            encoder_params = []
            head_params = []

            for name, param in self.named_parameters():
                if "encoder" in name:
                    encoder_params.append(param)
                elif "cls_head" in name or "survival_head" in name:
                    head_params.append(param)

        if not self.sam:
            if self.optimizer == "SGD":
                if self.finetuning_method == "full_sawtooth":
                    optimizer = torch.optim.SGD(
                        [
                            {
                                "params": head_params,
                                "lr": self.lr,
                                "momentum": 0.9,
                                "weight_decay": self.weight_decay,
                                "nesterov": self.nesterov,
                                "name": "cls_head",
                            },
                            {
                                "params": encoder_params,
                                "lr": self.lr,
                                "momentum": 0.9,
                                "weight_decay": self.weight_decay,
                                "nesterov": self.nesterov,
                                "name": "encoder",
                            },
                        ]
                    )

                else:
                    optimizer = torch.optim.SGD(
                        params,
                        lr=self.lr,
                        momentum=0.9,
                        weight_decay=self.weight_decay,
                        nesterov=self.nesterov,
                    )
            elif self.optimizer == "Adam":
                if self.finetuning_method == "full_sawtooth":
                    optimizer = torch.optim.Adam(
                        [
                            {
                                "params": head_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "cls_head",
                            },
                            {
                                "params": encoder_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "encoder",
                            },
                        ]
                    )

                else:
                    optimizer = torch.optim.Adam(
                        params, lr=self.lr, weight_decay=self.weight_decay
                    )
            elif self.optimizer == "AdamW":

                if self.finetuning_method == "full_sawtooth":
                    optimizer = torch.optim.AdamW(
                        [
                            {
                                "params": head_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "cls_head",
                            },
                            {
                                "params": encoder_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "encoder",
                            },
                        ]
                    )

                else:
                    optimizer = torch.optim.AdamW(
                        params, lr=self.lr, weight_decay=self.weight_decay
                    )
            elif self.optimizer == "Rmsprop":

                if self.finetuning_method == "full_sawtooth":
                    optimizer = RMSpropTF(
                        [
                            {
                                "params": head_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "cls_head",
                            },
                            {
                                "params": encoder_params,
                                "lr": self.lr,
                                "weight_decay": self.weight_decay,
                                "name": "encoder",
                            },
                        ]
                    )

                else:
                    optimizer = RMSpropTF(
                        params, lr=self.lr, weight_decay=self.weight_decay
                    )
            elif self.optimizer == "Madgrad":

                if self.finetuning_method == "full_sawtooth":
                    optimizer = MADGRAD(
                        [
                            {
                                "params": head_params,
                                "lr": self.lr,
                                "momentum": 0.9,
                                "weight_decay": self.weight_decay,
                                "name": "cls_head",
                            },
                            {
                                "params": encoder_params,
                                "lr": self.lr,
                                "momentum": 0.9,
                                "weight_decay": self.weight_decay,
                                "name": "encoder",
                            },
                        ]
                    )

                else:
                    optimizer = MADGRAD(
                        params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
                    )

        else:
            # ASAM paper suggests 10x larger rho for adaptive SAM than in normal SAM
            rho = 0.5 if self.adaptive_sam else 0.05

            if self.optimizer == "SGD":
                base_optimizer = torch.optim.SGD
                optimizer = SAM(
                    params,
                    base_optimizer,
                    adaptive=self.adaptive_sam,
                    lr=self.lr,
                    momentum=0.9,
                    weight_decay=self.weight_decay,
                    nesterov=self.nesterov,
                    rho=rho,
                )
            elif self.optimizer == "Madgrad":
                base_optimizer = MADGRAD
                optimizer = SAM(
                    params,
                    base_optimizer,
                    adaptive=self.adaptive_sam,
                    lr=self.lr,
                    momentum=0.9,
                    weight_decay=self.weight_decay,
                    rho=rho,
                )
            elif self.optimizer == "Adam":
                base_optimizer = torch.optim.Adam
                optimizer = SAM(
                    params,
                    base_optimizer,
                    adaptive=self.adaptive_sam,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    rho=rho,
                )
            elif self.optimizer == "AdamW":
                base_optimizer = torch.optim.AdamW
                optimizer = SAM(
                    params,
                    base_optimizer,
                    adaptive=self.adaptive_sam,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    rho=rho,
                )
            elif self.optimizer == "Rmsprop":
                base_optimizer = RMSpropTF
                optimizer = SAM(
                    params,
                    base_optimizer,
                    adaptive=self.adaptive_sam,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    rho=rho,
                )

        if not self.scheduler:
            return [optimizer]
        else:
            if self.scheduler == "CosineAnneal" and self.warmstart == 0:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.T_max
                )
            elif self.scheduler == "CosineAnneal" and self.warmstart > 0:
                if self.finetuning_method == "full_sawtooth":
                    scheduler = CosineAnnealingLR_DoubleWarmstart(
                        optimizer,
                        T_max=self.T_max,
                        warmstart1=self.warmstart,
                        warmstart2=self.warmstart2,
                    )
                else:
                    scheduler = CosineAnnealingLR_Warmstart(
                        optimizer,
                        T_max=self.T_max,
                        warmstart=self.warmstart,
                    )
            elif self.scheduler == "Step":
                # decays every 1/4 epochs
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=self.epochs // 4, gamma=0.1
                )
            elif self.scheduler == "MultiStep":
                # decays lr with 0.1 after half of epochs and 3/4 of epochs
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, [self.epochs // 2, self.epochs * 3 // 4]
                )

            return [optimizer], [scheduler]


class CosineAnnealingLR_Warmstart(_LRScheduler):
    """
    Same as CosineAnnealingLR but includes a warmstart option that will gradually increase the LR
    for the amount of specified warmup epochs as described in https://arxiv.org/pdf/1706.02677.pdf
    """

    def __init__(
            self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False, warmstart=0
    ):
        self.T_max = T_max - warmstart  # do not consider warmstart epochs for T_max
        self.eta_min = eta_min
        self.warmstart = warmstart
        self.T = 0

        super(CosineAnnealingLR_Warmstart, self).__init__(
            optimizer, last_epoch,
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
                UserWarning,
            )

        # Warmstart
        if self.last_epoch < self.warmstart:
            addrates = [(lr / (self.warmstart + 1)) for lr in self.base_lrs]
            updated_lr = [
                addrates[i] * (self.last_epoch + 1)
                for i, group in enumerate(self.optimizer.param_groups)
            ]

            return updated_lr

        else:
            if self.T == 0:
                self.T += 1
                return self.base_lrs
            elif (self.T - 1 - self.T_max) % (2 * self.T_max) == 0:
                updated_lr = [
                    group["lr"]
                    + (base_lr - self.eta_min)
                    * (1 - math.cos(math.pi / self.T_max))
                    / 2
                    for base_lr, group in zip(
                        self.base_lrs, self.optimizer.param_groups
                    )
                ]

                self.T += 1
                return updated_lr

            updated_lr = [
                (1 + math.cos(math.pi * self.T / self.T_max))
                / (1 + math.cos(math.pi * (self.T - 1) / self.T_max))
                * (group["lr"] - self.eta_min)
                + self.eta_min
                for group in self.optimizer.param_groups
            ]

            self.T += 1
            return updated_lr


class CosineAnnealingLR_DoubleWarmstart(_LRScheduler):
    """
    CosineAnnealingLR with two consecutive warmup phases.

    - Warmup 1: Increases LR from 0 to base LR, **only for `cls_head`**.
    - Warmup 2: Increases LR from 0 to base LR, **for both `cls_head` and `encoder`**.
    - Cosine Annealing: Decays LR **for both `cls_head` and `encoder`**.
    """

    def __init__(
            self,
            optimizer,
            T_max,
            eta_min=0,
            last_epoch=-1,
            verbose=False,
            warmstart1=0,
            warmstart2=0,
    ):
        self.warmstart1 = warmstart1
        self.warmstart2 = warmstart2
        self.eta_min = eta_min
        self.T_max = T_max - (warmstart1 + warmstart2)  # Effective decay period
        self.T = 0  # Internal counter

        # Identify param groups: assume "cls_head" and "encoder" are named properly in optimizer param_groups
        self.cls_head_group = None
        self.encoder_group = None

        for param_group in optimizer.param_groups:
            if param_group.get("name") == "cls_head":
                self.cls_head_group = param_group
            elif param_group.get("name") == "encoder":
                self.encoder_group = param_group

        if self.cls_head_group is None:
            raise ValueError("Optimizer must have a parameter group named 'cls_head'.")
        if self.encoder_group is None:
            raise ValueError("Optimizer must have a parameter group named 'encoder'.")

        super(CosineAnnealingLR_DoubleWarmstart, self).__init__(
            optimizer, last_epoch,
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
                UserWarning,
            )

        warmup_total = self.warmstart1 + self.warmstart2

        # First warmup phase (only `cls_head` is trained)
        if self.last_epoch < self.warmstart1:
            warmup_factor = (self.last_epoch + 1) / self.warmstart1
            updated_lr = []

            for group in self.optimizer.param_groups:
                if group is self.cls_head_group:
                    updated_lr.append(group["initial_lr"] * warmup_factor)
                else:  # Keep encoder frozen
                    updated_lr.append(0)

            return updated_lr

        # Second warmup phase (both `cls_head` and `encoder` are trained)
        elif self.last_epoch < warmup_total:
            warmup_factor = (self.last_epoch - self.warmstart1 + 1) / self.warmstart2
            updated_lr = [
                group["initial_lr"] * warmup_factor
                for group in self.optimizer.param_groups
            ]
            return updated_lr

        # Cosine annealing phase (both `cls_head` and `encoder`)
        else:
            epoch_cosine = self.last_epoch - warmup_total  # Shifted epoch count
            updated_lr = [
                self.eta_min
                + (group["initial_lr"] - self.eta_min)
                * 0.5
                * (1 + math.cos(math.pi * epoch_cosine / self.T_max))
                for group in self.optimizer.param_groups
            ]
            return updated_lr


class ModelConstructor(BaseModel):
    def __init__(self, model, **kwargs):
        super(ModelConstructor, self).__init__(**kwargs)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out
