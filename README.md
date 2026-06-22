# SSL3D Survival

This repository is a fork of the original 3D medical image classification codebase
from the [Helmholtz Imaging image classification framework](https://github.com/MIC-DKFZ/image_classification).
It has been adapted for 3D survival modeling and can fine-tune checkpoints from
[nnssl](https://github.com/MIC-DKFZ/nnssl).

<sub>Copyright German Cancer Research Center (DKFZ) and contributors. Please make sure that your usage of this code is in compliance with its license.</sub>

## Installation

Install the requirements in a virtual environment:

```shell
pip install -r requirements.txt
```

You may need to adapt the CUDA versions for `torch` and `torchvision`.

## Running 3D Survival

> This fork trains **discrete-time survival** models, not the upstream
> classification task. The runnable surface is `main.py` (training) and
> `medsurvival3d/inference/survival.py` (inference), both driven by Hydra.
> **Always run from the repository root** so local package imports resolve.

`main.py` composes environment, model, and data config groups. Override any
value on the command line:

| Group | Options | Default |
|-------|---------|---------|
| `env=` | environment config name | project default |
| `model=` | model config name | project default |
| `data=` | dataset/loss config name | project default |

The `data=` choice selects **both the dataset and the survival loss** (e.g.
NLL/logistic-hazard, Cox, DeepHit, PMF, MTLR, BCE survival, Weibull, PC-Hazard,
composite, or soft-logrank).

## Train — fine-tune from an SSL checkpoint

The model config defaults to `pretrained: True`, so you **must** pass
`model.chpt_path` or the run crashes in `torch.load(None)`:

```shell
WANDB_MODE=offline python main.py \
  env=<env_config> \
  data=<data_config> \
  model.chpt_path=/path/to/checkpoint_final.pth \
  exp_dir=/your/writable/output/dir
```

## Train — from scratch (no checkpoint)

```shell
WANDB_MODE=offline python main.py \
  env=<env_config> \
  data=<data_config> \
  model.pretrained=False \
  exp_dir=/your/writable/output/dir
```

## Quick smoke test (one train + val step)

```shell
WANDB_MODE=offline python main.py \
  env=<env_config> \
  data=<data_config> \
  model.chpt_path=/path/to/checkpoint_final.pth \
  exp_dir=/your/writable/output/dir \
  data.cv.k=1 +trainer.fast_dev_run=true
```

### Overrides & gotchas

- **`exp_dir=`** — override it with a writable directory for checkpoints and
  W&B logs.
- **Progress bar** — disable it for non-interactive SSH runs with
  `trainer.enable_progress_bar=false trainer.callbacks.progressbar=null`.
- **Do not set `seed=`** — leave the default (`False`). A fixed seed enables
  `deterministic=True`, which crashes the non-deterministic CUDA `avg_pool3d` backward.
- **`WANDB_MODE=offline`** (or `trainer.logger.offline=true`) avoids a W&B login.
- **Cross-validation:** `data.cv.k=<folds>` controls the number of folds.
  `data.module.fold=<k>` picks a specific fold.
- Mixed precision (`trainer.precision='16-mixed'`) is the default.

## Inference / evaluation

A second Hydra entry point, `medsurvival3d/inference/survival.py`, loads the best
checkpoint per fold from a finished run's `exp_dir`, runs val+test, ensembles
folds, and writes `metrics.csv`
+ per-fold predictions. Required args: `exp_dir` (training output dir containing
`config.yaml`), `splits_json`, `pred_dir`:

```shell
python -m medsurvival3d.inference.survival \
  exp_dir=/your/training/run/dir \
  splits_json=/path/to/splits.json \
  pred_dir=/your/predictions/out \
  folds=[0,1,2,3,4]
```

> Run the inference CLI from the repository root so imports and Hydra config
> resolution work correctly.



**If you use this codebase, please cite:**
```
   @misc{Openmind,
   title={An OpenMind for 3D medical vision self-supervised learning},
   author={Tassilo Wald and Constantin Ulrich and Jonathan Suprijadi and Sebastian Ziegler and Michal Nohel and Robin Peretzke and Gregor Köhler and Klaus H. Maier-Hein},
   year={2025},
   eprint={2412.17041},
   archivePrefix={arXiv},
   primaryClass={cs.CV},
   url={https://arxiv.org/abs/2412.17041},
   }
```

