"""Merge LASMA augmented samples into the training split.

The validation and test splits are copied unchanged. By default, paths are
resolved from lasma_config.py and remain relative to this project folder unless
the user overrides them with environment variables.
"""

import os
import shutil
from tqdm import tqdm

from lasma_config import get_path


ORIG_DIR = get_path("dataset_root")
LASMA_DIR = os.path.join(get_path("lasma_aug"), "aug_10")
FINAL_DIR = get_path("lasma_merged")


def merge_dataset_for_training():
    print(f"Original dataset: {ORIG_DIR}")
    print(f"LASMA augmented:  {LASMA_DIR}")
    print(f"Output merged:    {FINAL_DIR}")

    if os.path.exists(FINAL_DIR):
        print("Cleaning old merged folder...")
        shutil.rmtree(FINAL_DIR)

    print("\n[Phase 1] Copying original dataset...")
    shutil.copytree(ORIG_DIR, FINAL_DIR, dirs_exist_ok=True)

    final_train = os.path.join(FINAL_DIR, "train")
    print(f"\n[Phase 2] Merging LASMA images into {final_train} ...")

    copy_count = 0
    for class_name in tqdm(os.listdir(LASMA_DIR)):
        class_dir = os.path.join(LASMA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        target_dir = os.path.join(final_train, class_name)
        os.makedirs(target_dir, exist_ok=True)

        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                src_img = os.path.join(class_dir, img_name)
                base, ext = os.path.splitext(img_name)
                new_img_name = f"{base}_lasma{ext}"
                dst_img = os.path.join(target_dir, new_img_name)
                shutil.copy2(src_img, dst_img)
                copy_count += 1

    print("=" * 60)
    print(f"Done. Injected {copy_count} LASMA images.")
    print(f"Merged dataset: {FINAL_DIR}")
    print("  train/: original images + LASMA augmentations")
    print("  val/:   copied unchanged")
    print("  test/:  copied unchanged")
    print("=" * 60)


if __name__ == "__main__":
    merge_dataset_for_training()
