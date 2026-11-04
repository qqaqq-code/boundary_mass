"""Prepare the frozen feature caches consumed by BoundaryMass experiments."""

import argparse
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cifar100", "tiny_imagenet"], required=True)
    parser.add_argument("--backbone", choices=["resnet50", "vit_b_16", "dino", "siglip"], required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    import torch
    from utils.paths import DATASETS_DIR, RUNS_DIR, MODELS_DIR
    torch.hub.set_dir(str(MODELS_DIR / "torch" / "hub"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "cifar100" and args.backbone in {"dino", "siglip"}:
        from torchvision.datasets import CIFAR100
        from experiments.extract_cifar100_features import extract_one, MODELS
        for train in [True, False]:
            CIFAR100(DATASETS_DIR / "cifar100", train=train, download=True)
        spec = next(s for s in MODELS if s["short"] == args.backbone + "_ms")
        print(extract_one(spec))
    elif args.dataset == "cifar100":
        import train_cifar100_probe as implementation
        settings = SimpleNamespace(backbone_model=args.backbone, weights="",
                                   data_dir=DATASETS_DIR / "cifar100", download=True)
        model, dim, transform, weights = implementation.build_backbone(settings, device)
        train, test = implementation.build_datasets(settings, transform)
        group, cache = {
            "resnet50": ("cifar100_resnet50_pretrained", "feature_cache_resnet50_imagenetv2"),
            "vit_b_16": ("cifar100_vitb16_targeted", "feature_cache_vitb16_imagenetv1"),
        }[args.backbone]
        output = RUNS_DIR / group / cache
        implementation.load_or_compute_feature_cache(
            model, train, test, device, args.batch_size, args.num_workers,
            output, dim, args.backbone, weights)
        print(output)
    elif args.backbone == "resnet50":
        import train_imagefolder_probe as implementation
        root = DATASETS_DIR / "tiny-imagenet-200"
        settings = SimpleNamespace(backbone_model="resnet50", weights="IMAGENET1K_V2",
                                   train_dir=root / "train", val_dir=root / "val_by_class")
        model, dim, transform = implementation.build_backbone(settings, device)
        train, val = implementation.build_datasets(settings, transform)
        output = RUNS_DIR / "tiny_imagenet_resnet50_targeted/feature_cache_resnet50_imagenetv2"
        implementation.load_or_compute_feature_cache(
            model, train, val, device, args.batch_size, args.num_workers,
            output, dim, len(train.classes))
        print(output)
    else:
        from experiments.tiny_imagenet import extract_torchvision_features, extract_modelscope_features
        name = {"vit_b_16": "ViT-B", "dino": "DINO-MS", "siglip": "SigLIP-MS"}[args.backbone]
        extract = extract_torchvision_features if args.backbone == "vit_b_16" else extract_modelscope_features
        print(extract(name))


if __name__ == "__main__":
    main()
