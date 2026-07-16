import os
import time
import torch
import shutil

from generate_offline_dataset import generate_offline_dataset
from evaluate_augmented_dataset import evaluate_dataset_mhoa
from lasma_config import get_path

def main():
    print("=" * 50)
    print("      LASMA Auto Augmentation & Verification Pipeline")
    print("=" * 50 + "\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    orig_dir = get_path("train")
    offline_dir_base = get_path("lasma_aug")

    augment_times = 10

    print(f"\n>>>> [Phase 1/2] Starting offline LASMA augmentation (T={augment_times} resamples) <<<<")
    start_time = time.time()

    if os.path.exists(offline_dir_base):
        print(f"Cleaning old directory {offline_dir_base}.")
        shutil.rmtree(offline_dir_base)

    generate_offline_dataset(orig_dir, offline_dir_base, augment_times=augment_times, device=device)

    gen_time = (time.time() - start_time) / 60
    print(f"\n[Phase 1 complete] Augmentation took: {gen_time:.2f} minutes.\n")

    print(">>>> [Phase 2/2] Starting macro 2D-Entropy evaluation & selection <<<<")
    evaluate_dataset_mhoa(orig_dir, offline_dir_base)

    print("\n" + "=" * 50)
    print("      All done! Ready for classification model validation.")
    print("=" * 50)

if __name__ == "__main__":
    main()
