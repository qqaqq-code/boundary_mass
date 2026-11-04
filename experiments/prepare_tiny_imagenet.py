"Prepare Tiny-ImageNet images and class-organized validation folders. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare Tiny-ImageNet for ImageFolder-based experiments."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-rebuild-val", action="store_true")
    return parser.parse_args()


def build_logger() -> logging.Logger:
    logger = logging.getLogger("prepare_tiny_imagenet")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def download_file(
    url: str, destination: Path, logger: logging.Logger, force: bool
) -> None:
    if destination.exists() and not force:
        logger.info("Archive already exists: %s", destination)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    logger.info("Downloading %s -> %s", url, destination)
    wget = shutil.which("wget")
    if wget is not None:
        result = subprocess.run([wget, "-O", str(tmp_path), url], check=False)
        if result.returncode != 0:
            raise RuntimeError(f"wget failed with return code {result.returncode}")
    else:
        with (
            urllib.request.urlopen(url, timeout=60) as response,
            tmp_path.open("wb") as f,
        ):
            shutil.copyfileobj(response, f, length=1024 * 1024)
    tmp_path.replace(destination)
    logger.info("Download complete (%s bytes)", destination.stat().st_size)


def extract_archive(
    archive_path: Path, data_root: Path, logger: logging.Logger
) -> Path:
    extracted_root = data_root / "tiny-imagenet-200"
    marker = extracted_root / ".extract_complete"
    if marker.exists():
        logger.info("Extraction already complete: %s", extracted_root)
        return extracted_root

    logger.info("Extracting %s into %s", archive_path, data_root)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(data_root)
    marker.write_text("ok\n", encoding="utf-8")
    logger.info("Extraction complete: %s", extracted_root)
    return extracted_root


def build_val_split(dataset_root: Path, logger: logging.Logger, force: bool) -> Path:
    val_root = dataset_root / "val"
    images_dir = val_root / "images"
    annotations_path = val_root / "val_annotations.txt"
    split_root = dataset_root / "val_by_class"
    marker = split_root / ".ready"

    if marker.exists() and not force:
        logger.info("Validation split already prepared: %s", split_root)
        return split_root

    if force and split_root.exists():
        shutil.rmtree(split_root)

    split_root.mkdir(parents=True, exist_ok=True)
    logger.info("Preparing validation split under %s", split_root)

    lines = annotations_path.read_text(encoding="utf-8").splitlines()
    copied = 0
    for line in lines:
        if not line.strip():
            continue
        image_name, class_name, *_ = line.split("\t")
        class_dir = split_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        src = images_dir / image_name
        dst = class_dir / image_name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    marker.write_text("ok\n", encoding="utf-8")
    logger.info("Validation split ready with %d copied images", copied)
    return split_root


def main() -> None:
    args = parse_args()
    logger = build_logger()

    data_root = args.data_root.resolve()
    archive_path = data_root / "tiny-imagenet-200.zip"
    download_file(args.url, archive_path, logger, force=args.force_download)
    dataset_root = extract_archive(archive_path, data_root, logger)
    val_split_root = build_val_split(dataset_root, logger, force=args.force_rebuild_val)

    logger.info("Train dir: %s", dataset_root / "train")
    logger.info("Val dir: %s", val_split_root)


if __name__ == "__main__":
    main()
