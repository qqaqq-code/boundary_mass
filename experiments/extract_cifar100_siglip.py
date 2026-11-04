"Extract CIFAR-100 SigLIP vision features. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import (
    PROJECT_ROOT,
    CODE_DIR,
    DATASETS_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    result_file,
)
import json, sys, time, traceback, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = PROJECT_ROOT
STUDY_ROOT = PROJECT_ROOT
ART = RESULTS_DIR
CACHE = FEATURES_DIR / "cifar100"
ART.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
# Use isolated newer transformers for SigLIP; keep conda torch/numpy.
sys.path.insert(0, str(CODE_DIR))
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from utils.paths import resolve_foundation_checkpoint
from transformers import AutoImageProcessor, SiglipVisionModel

DATA = DATASETS_DIR / "cifar100"
MID = "AI-ModelScope/siglip-base-patch16-224"
OUT = CACHE / "siglip_ms"


def collate(batch):
    imgs, labels = zip(*batch)
    return list(imgs), torch.tensor(labels, dtype=torch.long)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "train_features.pt").exists() and (OUT / "test_features.pt").exists():
        st = {
            "model": "siglip_ms",
            "model_id": MID,
            "status": "cached",
            "outdir": str(OUT),
        }
        (result_file("model_info/modelscope_siglip_status.json")).write_text(
            json.dumps(st, indent=2)
        )
        print(json.dumps(st, indent=2))
        return
    local = resolve_foundation_checkpoint(MID)
    proc = AutoImageProcessor.from_pretrained(local, local_files_only=True)
    model = (
        SiglipVisionModel.from_pretrained(local, local_files_only=True).eval().cuda()
    )
    dim = None
    for split, train in [("train", True), ("test", False)]:
        ds = datasets.CIFAR100(
            root=str(DATA), train=train, download=False, transform=None
        )
        loader = DataLoader(
            ds,
            batch_size=192,
            shuffle=False,
            num_workers=4,
            collate_fn=collate,
            pin_memory=True,
        )
        feats = []
        labs = []
        t0 = time.time()
        with torch.inference_mode():
            for i, (imgs, y) in enumerate(loader):
                inp = proc(images=imgs, return_tensors="pt")
                inp = {k: v.cuda(non_blocking=True) for k, v in inp.items()}
                out = model(**inp)
                z = out.pooler_output.detach().cpu().float()
                feats.append(z)
                labs.append(y)
                if (i + 1) % 25 == 0:
                    print(
                        "siglip_ms",
                        split,
                        i + 1,
                        "/",
                        len(loader),
                        "elapsed",
                        round(time.time() - t0, 1),
                        flush=True,
                    )
        F = torch.cat(feats, 0)
        Y = torch.cat(labs, 0)
        dim = int(F.shape[1])
        torch.save(
            {
                "features": F,
                "labels": Y,
                "modelscope_id": MID,
                "local_dir": local,
                "feature_dim": dim,
                "backend": "vendor_transformers_siglip_vision",
            },
            OUT / f"{split}_features.pt",
        )
    st = {
        "model": "siglip_ms",
        "model_id": MID,
        "status": "ok",
        "outdir": str(OUT),
        "feature_dim": dim,
    }
    (result_file("model_info/modelscope_siglip_status.json")).write_text(
        json.dumps(st, indent=2)
    )
    print(json.dumps(st, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st = {
            "model": "siglip_ms",
            "model_id": MID,
            "status": "failed",
            "error": repr(e),
            "traceback": traceback.format_exc()[-4000:],
        }
        (result_file("model_info/modelscope_siglip_status.json")).write_text(
            json.dumps(st, indent=2)
        )
        print(json.dumps(st, indent=2))
        raise
