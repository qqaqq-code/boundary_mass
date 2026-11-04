# Data preparation

Install the training dependencies with `pip install -r requirements-train.txt` or `conda env create -f environment.yml`. Run commands from the project root.

## Datasets

CIFAR-100 is obtained through torchvision when its feature extractor runs. Tiny-ImageNet preparation creates the class-organized image folders:

```bash
python -m experiments.prepare_tiny_imagenet --data-root data/datasets
```

Dataset sources: [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html), [Tiny-ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip).

## Feature extraction

```bash
python extract_features.py --dataset cifar100 --backbone resnet50
python extract_features.py --dataset cifar100 --backbone vit_b_16
python extract_features.py --dataset cifar100 --backbone dino
python extract_features.py --dataset cifar100 --backbone siglip

python extract_features.py --dataset tiny_imagenet --backbone resnet50
python extract_features.py --dataset tiny_imagenet --backbone vit_b_16
python extract_features.py --dataset tiny_imagenet --backbone dino
python extract_features.py --dataset tiny_imagenet --backbone siglip
```

The DINO and SigLIP implementations use CUDA. Models are torchvision ResNet-50 `IMAGENET1K_V2`, ViT-B/16 `IMAGENET1K_V1`, `AI-ModelScope/dino-vitb16`, and `AI-ModelScope/siglip-base-patch16-224`.

Tiny-ImageNet comparisons use the common set of 80 classes and the shared `wnids.txt` class order. `utils/tiny_labels.py` maps ResNet's ImageFolder labels to that order when loading features.

## Input format

Feature files contain a dictionary with `features` and `labels` tensors. Use `train_features.pt` and `test_features.pt` for CIFAR-100, and `train_features.pt` and `val_features.pt` for Tiny-ImageNet.

| Dataset / backbone | Directory relative to `data/` |
|---|---|
| CIFAR-100 ResNet-50 | `training_runs/cifar100_resnet50_pretrained/feature_cache_resnet50_imagenetv2/` |
| CIFAR-100 ViT-B/16 | `training_runs/cifar100_vitb16_targeted/feature_cache_vitb16_imagenetv1/` |
| CIFAR-100 DINO / SigLIP | `features/cifar100/dino_ms/`, `features/cifar100/siglip_ms/` |
| Tiny-ImageNet ResNet-50 | `training_runs/tiny_imagenet_resnet50_targeted/feature_cache_resnet50_imagenetv2/` |
| Tiny-ImageNet ViT / DINO / SigLIP | `features/tiny_imagenet/ViT-B/`, `DINO-MS/`, `SigLIP-MS/` |

`BOUNDARY_DATA_DIR` selects another data root. `BOUNDARY_RUNS_DIR` selects the training-run directory. The experiment runner configures output tables and figures beneath `--output-dir`.

## Probe training

```bash
python train_cifar100_probe.py \
  --data-dir data/datasets/cifar100 \
  --feature-cache-dir data/training_runs/cifar100_resnet50_pretrained/feature_cache_resnet50_imagenetv2 \
  --backbone-model resnet50 --mode standard \
  --train-samples-per-class 10 --epochs 120 \
  --output-dir outputs/cifar100_probe --download

python train_imagefolder_probe.py --help
python train_encoder.py --help
python train_end_to_end.py --help
```
