"""Extract Tiny-ImageNet caches using the paper's original extraction functions."""
import argparse
from experiments.tiny_imagenet import extract_torchvision_features, extract_modelscope_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('backbone', choices=['ViT-B', 'DINO-MS', 'SigLIP-MS'])
    args = parser.parse_args()
    if args.backbone == 'ViT-B':
        extract_torchvision_features(args.backbone)
    else:
        extract_modelscope_features(args.backbone)
