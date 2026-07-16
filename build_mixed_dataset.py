import os
import shutil
import random

from lasma_config import get_path

source_dir = get_path("lasma_merged_train")
target_base_dir = get_path("lasma_1750")
target_train_dir = os.path.join(target_base_dir, 'train')

def main():
    if os.path.exists(target_train_dir):
        shutil.rmtree(target_train_dir)
    os.makedirs(target_train_dir)

    categories = os.listdir(source_dir)
    total_sampled = 0

    print("Generating controlled 1750-image mixed dataset (50% real / 50% augmented)...")
    print("-" * 50)

    orig_total = 0
    cat_info = {}

    for cat in categories:
        cat_dir = os.path.join(source_dir, cat)
        if not os.path.isdir(cat_dir): continue
        all_files = os.listdir(cat_dir)
        aug_files = [f for f in all_files if '_lasma' in f]
        orig_files = [f for f in all_files if '_lasma' not in f]
        orig_total += len(orig_files)
        cat_info[cat] = {'orig': orig_files, 'aug': aug_files, 'cat_dir': cat_dir}

    target_total = 1750
    print(f"Original images: {orig_total}. Sampling {target_total} total with 50/50 real/augmented ratio per category.\n")

    cat_allocation = {}
    remaining_total = target_total

    for idx, (cat, info) in enumerate(cat_info.items()):
        orig_n = len(info['orig'])
        if idx == len(cat_info) - 1:
            cat_allocation[cat] = remaining_total
        else:
            alloc = int(target_total * (orig_n / orig_total))
            cat_allocation[cat] = alloc
            remaining_total -= alloc

    for cat in categories:
        if cat not in cat_info: continue

        info = cat_info[cat]
        orig_files = info['orig']
        aug_files = info['aug']

        cat_target_n = cat_allocation[cat]

        num_orig_to_sample = cat_target_n // 2
        num_aug_to_sample = cat_target_n - num_orig_to_sample

        if num_orig_to_sample > len(orig_files):
            diff = num_orig_to_sample - len(orig_files)
            num_orig_to_sample = len(orig_files)
            num_aug_to_sample += diff
        elif num_aug_to_sample > len(aug_files):
            diff = num_aug_to_sample - len(aug_files)
            num_aug_to_sample = len(aug_files)
            num_orig_to_sample += diff

        sampled_orig = random.sample(orig_files, num_orig_to_sample)
        sampled_aug = random.sample(aug_files, num_aug_to_sample)

        target_cat_dir = os.path.join(target_train_dir, cat)
        os.makedirs(target_cat_dir)

        for f in sampled_aug + sampled_orig:
            shutil.copy2(os.path.join(info['cat_dir'], f), os.path.join(target_cat_dir, f))

        print(f"[{cat}]: quota {cat_target_n} -> {len(sampled_orig)} real + {len(sampled_aug)} augmented")
        total_sampled += (len(sampled_orig) + len(sampled_aug))

    print("-" * 50)
    print(f"Done! New training set: {total_sampled} images (verified {target_total}).")

    for split in ['val', 'test']:
        src_split = source_dir.replace('train', split)
        tgt_split = os.path.join(target_base_dir, split)
        if os.path.exists(src_split):
            if os.path.exists(tgt_split):
                shutil.rmtree(tgt_split)
            shutil.copytree(src_split, tgt_split)
            print(f"Copied {split} set.")

    print(f"\nUpdate data path to: {target_base_dir}")

if __name__ == '__main__':
    main()
