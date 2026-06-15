"""LR schedulers and optimizer construction for survival training."""
import math
import warnings
from torch.optim.lr_scheduler import _LRScheduler


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
