import os
import shutil
import random

def split_dataset(source_dir, dest_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    print("=" * 50)
    print("      Building Standard External Validation Set (Train / Val / Test)")
    print("=" * 50)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"

    if os.path.exists(dest_dir):
        print(f"[!] Detected existing output directory {dest_dir}, cleaning...")
        shutil.rmtree(dest_dir)

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, split))

    categories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    total_imgs = 0
    print(f"Found {len(categories)} disease categories: {categories}\n")

    for cat in categories:
        cat_dir = os.path.join(source_dir, cat)
        files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]

        random.seed(42)
        random.shuffle(files)

        num_files = len(files)
        train_end = int(num_files * train_ratio)
        val_end = train_end + int(num_files * val_ratio)

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        print(f"[{cat}] Total: {num_files} -> Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

        for split_name, split_files in splits.items():
            split_cat_dir = os.path.join(dest_dir, split_name, cat)
            os.makedirs(split_cat_dir, exist_ok=True)
            for f in split_files:
                shutil.copy2(os.path.join(cat_dir, f), os.path.join(split_cat_dir, f))
                total_imgs += 1

    print("\n" + "=" * 50)
    print(f"Done! Processed {total_imgs} images.")
    print(f"Standardized dataset saved to: {dest_dir}")
    print("=" * 50)

if __name__ == '__main__':
    from lasma_config import get_path
    SRC_DIR = get_path("ext_srinivasan")
    DST_DIR = get_path("ext_split")

    split_dataset(SRC_DIR, DST_DIR, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
