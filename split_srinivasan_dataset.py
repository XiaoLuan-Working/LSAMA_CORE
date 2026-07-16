"""Class-stratified train/val/test split for an OCT dataset.

This script is kept for datasets without patient identifiers. For patient-level
splitting, prefer split_srinivasan_by_patient.py.
"""

import os
import random
import shutil

from lasma_config import get_path


SOURCE_DIR = get_path("ext_srinivasan")
OUTPUT_DIR = get_path("ext_split")
SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42


def split_dataset():
    random.seed(RANDOM_SEED)

    classes = [
        item for item in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, item))
    ]
    print(f"Found classes: {classes}")

    for split in SPLIT_RATIO:
        for cls in classes:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    total_stats = {split: 0 for split in SPLIT_RATIO}

    for cls in classes:
        cls_dir = os.path.join(SOURCE_DIR, cls)
        files = sorted([
            filename for filename in os.listdir(cls_dir)
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        ])
        random.shuffle(files)

        n_train = int(len(files) * SPLIT_RATIO["train"])
        n_val = int(len(files) * SPLIT_RATIO["val"])
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split, split_files in splits.items():
            for filename in split_files:
                src = os.path.join(cls_dir, filename)
                dst = os.path.join(OUTPUT_DIR, split, cls, filename)
                shutil.copy2(src, dst)
            total_stats[split] += len(split_files)
            print(f"  {cls} -> {split}: {len(split_files)} images")

    print(f"\nSplit complete. Output directory: {OUTPUT_DIR}")
    print(
        f"Totals: train={total_stats['train']}, "
        f"val={total_stats['val']}, test={total_stats['test']}"
    )


if __name__ == "__main__":
    split_dataset()
