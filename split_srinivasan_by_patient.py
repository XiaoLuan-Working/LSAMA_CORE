"""Patient-level split for the Srinivasan 2014 OCT dataset.

Input and output paths are configured in lasma_config.py. The default locations
are relative to this project folder and can be overridden with environment
variables.
"""

import os
import random
import shutil
from collections import defaultdict

from lasma_config import get_path


SRC_DIR = get_path("ext_srinivasan")
DST_DIR = get_path("ext_split")
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42


def split_by_patient():
    categories = ["AMD", "DME", "NORMAL"]

    for category in categories:
        src_cat = os.path.join(SRC_DIR, category)
        if not os.path.isdir(src_cat):
            print(f"[WARN] {src_cat} not found, skipping")
            continue

        patient_files = defaultdict(list)
        for filename in os.listdir(src_cat):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")):
                continue
            patient_id = filename.split("_")[0]
            patient_files[patient_id].append(filename)

        patients = list(patient_files.keys())
        random.seed(SEED)
        random.shuffle(patients)

        train_end = int(len(patients) * TRAIN_RATIO)
        val_end = int(len(patients) * (TRAIN_RATIO + VAL_RATIO))
        splits = {
            "train": patients[:train_end],
            "val": patients[train_end:val_end],
            "test": patients[val_end:],
        }

        print(f"\n{category}: {len(patients)} patients")
        for split_name, split_patients in splits.items():
            total_images = sum(len(patient_files[p]) for p in split_patients)
            print(f"  {split_name}: {len(split_patients)} patients, {total_images} images")

            dst_split = os.path.join(DST_DIR, split_name, category)
            os.makedirs(dst_split, exist_ok=True)

            for patient in split_patients:
                for filename in patient_files[patient]:
                    shutil.copy2(os.path.join(src_cat, filename), os.path.join(dst_split, filename))

    print(f"\nDone. Dataset saved to: {DST_DIR}")


if __name__ == "__main__":
    split_by_patient()
