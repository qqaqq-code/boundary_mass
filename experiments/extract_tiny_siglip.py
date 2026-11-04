"Extract Tiny-ImageNet SigLIP vision features. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import PROJECT_ROOT, DATASETS_DIR, FEATURES_DIR, MODELS_DIR

import json
import sys
import time
from pathlib import Path

ROOT = PROJECT_ROOT
STUDY_ROOT = PROJECT_ROOT

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from utils.paths import resolve_foundation_checkpoint
from transformers import AutoImageProcessor, SiglipVisionModel

TINY_ROOT = DATASETS_DIR / "tiny-imagenet-200"
OUT = FEATURES_DIR / "tiny_imagenet" / "SigLIP-MS"
MID = "AI-ModelScope/siglip-base-patch16-224"
N_PAIRS = 40


def selected_labels() -> set[int]:
    out = set()
    for i in range(N_PAIRS):
        out.add(2 * i)
        out.add(2 * i + 1)
    return out


class TinyImageNetDataset(Dataset):
    def __init__(self, root: Path, split: str, keep_labels: set[int]):
        self.root = root
        self.wnids = [
            x.strip()
            for x in (root / "wnids.txt").read_text().splitlines()
            if x.strip()
        ]
        self.wnid_to_idx = {w: i for i, w in enumerate(self.wnids)}
        self.samples = []
        if split == "train":
            for label, wnid in enumerate(self.wnids):
                if label not in keep_labels:
                    continue
                for p in sorted((root / "train" / wnid / "images").glob("*.JPEG")):
                    self.samples.append((p, label))
        else:
            ann = {}
            for line in (root / "val" / "val_annotations.txt").read_text().splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    ann[parts[0]] = parts[1]
            for img, wnid in sorted(ann.items()):
                label = self.wnid_to_idx[wnid]
                if label in keep_labels:
                    self.samples.append((root / "val" / "images" / img, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        return Image.open(p).convert("RGB"), y


def collate(batch):
    imgs, labels = zip(*batch)
    return list(imgs), torch.tensor(labels, dtype=torch.long)


@torch.inference_mode()
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "train_features.pt").exists() and (OUT / "val_features.pt").exists():
        print(json.dumps({"status": "cached", "out": str(OUT)}, indent=2))
        return
    local = resolve_foundation_checkpoint(MID)
    proc = AutoImageProcessor.from_pretrained(local, local_files_only=True)
    model = (
        SiglipVisionModel.from_pretrained(local, local_files_only=True).eval().cuda()
    )
    keep = selected_labels()
    for split in ["train", "val"]:
        ds = TinyImageNetDataset(TINY_ROOT, split, keep)
        loader = DataLoader(
            ds,
            batch_size=192,
            shuffle=False,
            num_workers=4,
            collate_fn=collate,
            pin_memory=True,
        )
        feats, labs = [], []
        t0 = time.time()
        for i, (imgs, y) in enumerate(loader):
            inp = proc(images=imgs, return_tensors="pt")
            inp = {k: v.cuda(non_blocking=True) for k, v in inp.items()}
            out = model(**inp)
            feats.append(out.pooler_output.detach().cpu().float())
            labs.append(y)
            if (i + 1) % 25 == 0:
                print(
                    "SigLIP-MS",
                    split,
                    i + 1,
                    "/",
                    len(loader),
                    "elapsed",
                    round(time.time() - t0, 1),
                    flush=True,
                )
        F = torch.cat(feats)
        Y = torch.cat(labs)
        torch.save(
            {
                "features": F,
                "labels": Y,
                "modelscope_id": MID,
                "local_dir": local,
                "feature_dim": int(F.shape[1]),
                "selected_labels": sorted(keep),
            },
            OUT / f"{split}_features.pt",
        )
    print(
        json.dumps(
            {"status": "ok", "out": str(OUT), "feature_dim": int(F.shape[1])}, indent=2
        )
    )


if __name__ == "__main__":
    main()
