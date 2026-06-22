# 3D medical image classification repository
<sub>Copyright German Cancer Research Center (DKFZ) and contributors. Please make sure that your usage of this code is in compliance with its license.<sub>

Welcome to this 3D medical image classification repository. The repository builds up on the [IMAGE CLASSIFICATION FRAMEWORK BY HELMHOLTZ IMAGING](https://github.com/MIC-DKFZ/image_classification).
This repository was extended to allow fine-tuning checkpoints from this repository: [nnssl](https://github.com/MIC-DKFZ/nnssl). 
# Installation
## Requirements
Install the requirements in a [virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) by:

```shell
pip install -r requirements.txt
```

You might need to adapt the cuda versions for torch and torchvision.
Find a torch installation guide for your system [here](https://pytorch.org/get-started/locally/).


# Dataset preprocessing
Currently, preprocessing is highly dataset- and user-dependent. 
However in [this file](/datasets/preprocess_3D_data/datasets/template_brain_preprocessing.py) you can find examples of how a dataset can be preprocessed. 

For the SSL3D challenge we will resample all images towards a 1mm target spacing and then crop the center of the image with an 160 cubic block.  

# Including other datasets

For including your own dataset follow these steps:
1. In the ```dataset``` directory create a new file that implements the [torch dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files) class for your data. See [example](/datasets/RECvsT_1mm_cropped_160.py).
2. Additionally, create the [DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) for your dataset by writing a class that inherits from `BaseDataModule`. Write the `init` and `setup` functions for your dataset. The dataloaders are already defined by the `BaseDataModule`. An example could look like this:
    ```python
    from .base_datamodule import BaseDataModule

    class CustomDataModule(BaseDataModule):
      def __init__(self, **params):
          super(CustomDataModule, self).__init__(**params)

      def setup(self, stage: str):
          self.train_dataset = YourCustomPytorchDataset(
              data_path=self.data_path,
              split="train",
              transform=self.train_transforms,
          )
          self.val_dataset = YourCustomPytorchDataset(
              data_path=self.data_path,
              split="val",
              transform=self.test_transforms,
          )
    ```
   Note that the `__init__` function takes `**params` and passes them to the super init. By doing so the attributes `self.data_path`, `self.train_transforms` and `self.test_transforms` are already set automatically and can be used in the `setup` function. The `self.data_path` is a joined path consisting of the configs `data.module.data_root_dir` and `data.module.name`.
   Custom transforms can be added in `./augmentation/policies/<your-data>.py`. They need to inherit from the `BaseTransform` class. See the existing transforms for examples! 
3. Add a `<your-data>.yaml` file to the data config group, defining some data-specific variables.
    ```yaml
    # @package _global_
    data:
      module:
        _target_: datasets.RECvsT_1mm_cropped_160.RECvsT_1mm_cropped_160_DataModule
        name: RECvsT_1mm_cropped_160
        data_root_dir: ${data_dir}
        batch_size: 1
        train_transforms:
        _target_: augmentation.policies.batchgenerators.get_training_transforms
        patch_size: ${data.patch_size}
        rotation_for_DA: 0.523599
        mirror_axes: [0,1,2]
        do_dummy_2d_data_aug: False
        test_transforms: null
      cv:
        k:3

      num_classes: 2
      patch_size: [160, 160, 160]

    model:
      task: 'Classification'
      cifar_size: False
      input_channels: 2
      input_dim: 3
      input_shape: ${data.patch_size}
      optimizer: AdamW
      lr: 0.0001
      warmstart: 20
      weight_decay: 1e-2
      label_smoothing: 0.2
   
   trainer:
    logger:
      project: RECvsT_1mm_cropped_160
    accumulate_grad_batches: 48
    max_epochs: 400
    sync_batchnorm: True
   
   metrics:
    - 'f1'
    - 'balanced_acc'
    - 'ap'
    - 'auroc'
    ```
   The `data.module._target_` defines the path to your `DataModule`. Note that the first line of the file needs to be `# @package _global_` in order for Hydra to read the config properly.


# Running this fork (3D survival)

> This fork trains **discrete-time survival** models, not the upstream
> classification task. The runnable surface is `main.py` (training) and
> `medsurvival3d/inference/survival.py` (inference), both driven by Hydra configs
> in `cli_configs/`. **Always run from the repository root** so that
> `medsurvival3d`, `parsing_utils.py`, and `./cli_configs` resolve.

`main.py` (root config `cli_configs/train.yaml`) composes four config groups;
override any value on the command line:

| Group | Options | Default |
|-------|---------|---------|
| `env=` | `local`, `cluster` | `local` |
| `model=` | `resenc_survival` | `resenc_survival` |
| `data=` | `methylome_t1c_combined_{nll,cox,deephit,pmf,mtlr,bcesurv,weibull,pchazard,composite,soft_logrank}`, `methylome_t1c_combined_high_vs_low`, `methylome_t2w_combined_high_vs_low` | `…_soft_logrank` |

The `data=` choice selects **both the dataset and the survival loss** (e.g.
`…_nll` = NLL/logistic-hazard, `…_cox` = Cox, `…_deephit` = DeepHit, …).

## Train — fine-tune from an SSL checkpoint

The model config defaults to `pretrained: True`, so you **must** pass
`model.chpt_path` or the run crashes in `torch.load(None)`:

```shell
WANDB_MODE=offline python main.py \
  env=cluster \
  data=methylome_t1c_combined_nll \
  model.chpt_path=/path/to/S3D/checkpoint_final.pth \
  exp_dir=/your/writable/output/dir
```

## Train — from scratch (no checkpoint)

```shell
WANDB_MODE=offline python main.py \
  env=cluster \
  data=methylome_t1c_combined_nll \
  model.pretrained=False \
  exp_dir=/your/writable/output/dir
```

## Quick smoke test (one train + val step)

```shell
WANDB_MODE=offline python main.py \
  env=cluster \
  data=methylome_t1c_combined_nll \
  model.chpt_path=/path/to/S3D/checkpoint_final.pth \
  exp_dir=/your/writable/output/dir \
  data.cv.k=1 +trainer.fast_dev_run=true
```

### Overrides & gotchas (verified on the GPU workstation)

- **`exp_dir=`** — the bundled `env=local` / `env=cluster` point at DKFZ paths;
  override it with a writable directory (holds checkpoints + W&B logs). Dataset
  paths are absolute *inside* each `data=` config, so no `data_dir` override is needed.
- **`env=cluster`** already disables the progress bar, which is **required** over
  non-interactive SSH (the rich progress bar otherwise crashes with
  `IndexError: pop from empty list`). With `env=local`, add
  `trainer.enable_progress_bar=false trainer.callbacks.progressbar=null`.
- **Do not set `seed=`** — leave the default (`False`). A fixed seed enables
  `deterministic=True`, which crashes the non-deterministic CUDA `avg_pool3d` backward.
- **`WANDB_MODE=offline`** (or `trainer.logger.offline=true`) avoids a W&B login.
- **Cross-validation:** `data.cv.k=<folds>` (the per-loss configs set `k=5`;
  `train.yaml` defaults to 1 = no CV). `data.module.fold=<k>` picks the fold.
- Mixed precision (`trainer.precision='16-mixed'`) is the default.

## Inference / evaluation

A second Hydra entry point, `medsurvival3d/inference/survival.py` (config
`cli_configs/inference_survival.yaml`), loads the best checkpoint per fold from a
finished run's `exp_dir`, runs val+test, ensembles folds, and writes `metrics.csv`
+ per-fold predictions. Required args: `exp_dir` (training output dir containing
`config.yaml`), `splits_json`, `pred_dir`:

```shell
python -m medsurvival3d.inference.survival \
  exp_dir=/your/training/run/dir \
  splits_json=/path/to/splits_balanced_survival.json \
  pred_dir=/your/predictions/out \
  folds=[0,1,2,3,4]
```

> The training CLI above is verified end-to-end on the workstation. The inference
> **CLI entry** has so far only been exercised at the function level by the test
> suite — run it from the repo root, and if Hydra cannot locate the config, add
> `--config-dir cli_configs`.



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



