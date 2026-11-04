"""Align Tiny-ImageNet feature labels with the dataset's wnids.txt ordering."""

from .paths import DATASETS_DIR


def align_labels(blob, cache_path):
    labels = blob["labels"].numpy().astype("int64")
    if "resnet50" not in str(cache_path).lower() or "wnids.txt" in blob.get("label_space", ""):
        return labels
    classes = (DATASETS_DIR / "tiny-imagenet-200" / "wnids.txt").read_text().splitlines()
    classes = [name.strip() for name in classes if name.strip()]
    lookup = {name: i for i, name in enumerate(classes)}
    import numpy as np
    indices = np.asarray([lookup[name] for name in sorted(classes)], dtype="int64")
    return indices[labels]
