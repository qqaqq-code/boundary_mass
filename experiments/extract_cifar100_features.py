"Extract CIFAR-100 features with DINO and SigLIP. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import (
    PROJECT_ROOT,
    DATASETS_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    result_file,
)
import json, os, time, traceback, warnings, sys
from pathlib import Path

warnings.filterwarnings("ignore")
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from utils.paths import resolve_foundation_checkpoint

ROOT = PROJECT_ROOT
STUDY_ROOT = PROJECT_ROOT
ART = RESULTS_DIR
CACHE = FEATURES_DIR / "cifar100"
ART.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
DATA = DATASETS_DIR / "cifar100"
MS_CACHE = MODELS_DIR
MODELS = [
    {"short": "dino_ms", "model_id": "AI-ModelScope/dino-vitb16", "backend": "base"},
    {
        "short": "siglip_ms",
        "model_id": "AI-ModelScope/siglip-base-patch16-224",
        "backend": "vendor",
    },
]


def collate(batch):
    imgs, labels = zip(*batch)
    return list(imgs), torch.tensor(labels, dtype=torch.long)


def pick_features(short, out):
    # For DINO ViTModel, pooler is newly initialized in transformers, so use CLS token.
    if short.startswith("dino") and hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "vision_model_output") and out.vision_model_output is not None:
        vo = out.vision_model_output
        if hasattr(vo, "pooler_output") and vo.pooler_output is not None:
            return vo.pooler_output
        if hasattr(vo, "last_hidden_state"):
            return vo.last_hidden_state[:, 0]
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]
    if isinstance(out, tuple):
        return out[0][:, 0]
    raise RuntimeError("unknown output fields: " + str(type(out)))


def extract_one(spec):
    short = spec["short"]
    model_id = spec["model_id"]
    backend = spec.get("backend", "base")
    outdir = CACHE / short
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "train_features.pt").exists() and (
        outdir / "test_features.pt"
    ).exists():
        return {
            "model": short,
            "model_id": model_id,
            "status": "cached",
            "outdir": str(outdir),
        }
    local = resolve_foundation_checkpoint(model_id)
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(
        local, local_files_only=True, trust_remote_code=True, use_fast=False
    )
    # SigLIP's full multimodal model requires text inputs; the paper uses its
    # vision tower's pooler_output, as in experiments/extract_cifar100_siglip.py.
    if short == "siglip_ms":
        from transformers import SiglipVisionModel
        model_class = SiglipVisionModel
    else:
        model_class = AutoModel
    model = (
        model_class.from_pretrained(local, local_files_only=True, trust_remote_code=True)
        .eval()
        .cuda()
    )
    dim = None
    for split, train in [("train", True), ("test", False)]:
        ds = datasets.CIFAR100(
            root=str(DATA), train=train, transform=None, download=False
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
                inputs = proc(images=imgs, return_tensors="pt")
                inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
                z = pick_features(short, model(**inputs)).detach().cpu().float()
                feats.append(z)
                labs.append(y)
                if (i + 1) % 25 == 0:
                    print(
                        short,
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
                "modelscope_id": model_id,
                "local_dir": local,
                "feature_dim": dim,
                "backend": backend,
            },
            outdir / f"{split}_features.pt",
        )
    del model
    torch.cuda.empty_cache()
    return {
        "model": short,
        "model_id": model_id,
        "status": "ok",
        "outdir": str(outdir),
        "feature_dim": dim,
        "backend": backend,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=[s['short'] for s in MODELS],
                        default=[s['short'] for s in MODELS])
    args = parser.parse_args()
    status = []
    for spec in MODELS:
        if spec['short'] not in args.models:
            continue
        try:
            print("START", spec, flush=True)
            st = extract_one(spec)
        except Exception as e:
            st = {"model": spec["short"], "model_id": spec["model_id"],
                  "status": "failed", "error": repr(e),
                  "traceback": traceback.format_exc()[-4000:]}
            print("FAILED", spec["short"], repr(e), flush=True)
        status.append(st)
        result_file(f"model_info/{spec['short']}_status.json").write_text(json.dumps(st, indent=2))
    # Single-model jobs have distinct status files so parallel workers cannot
    # overwrite each other's experiment trail.
    if len(args.models) == len(MODELS):
        result_file("model_info/modelscope_foundation_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    return int(any(s['status'] == 'failed' for s in status))


if __name__ == '__main__':
    raise SystemExit(main())
