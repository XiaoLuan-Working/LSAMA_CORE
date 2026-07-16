import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

def calculate_2d_entropy(img):
    neighbor_mean = cv2.blur(img, (3, 3))

    hist, _, _ = np.histogram2d(
        img.flatten(),
        neighbor_mean.flatten(),
        bins=[256, 256],
        range=[[0, 256], [0, 256]]
    )

    P = hist / np.sum(hist)

    P_non_zero = P[P > 0]
    entropy = -np.sum(P_non_zero * np.log2(P_non_zero))

    return entropy

def evaluate_dataset_mhoa(orig_dir, aug_dir_base):
    orig_paths = glob.glob(os.path.join(orig_dir, '**', '*.jpg'), recursive=True)
    if len(orig_paths) == 0:
        print(f"No original images found in {orig_dir}.")
        return

    orig_entropies = []

    print(f"Found {len(orig_paths)} original images, extracting 2D-Entropy feature space...")

    aug_dirs = sorted([d for d in os.listdir(aug_dir_base) if d.startswith('aug_')])
    aug_entropies_dict = {d: [] for d in aug_dirs}

    for opath in tqdm(orig_paths):
        o_img = cv2.imread(opath, cv2.IMREAD_GRAYSCALE)
        if o_img is None: continue
        orig_entropies.append(calculate_2d_entropy(o_img))

        rel_path = os.path.relpath(opath, orig_dir)
        for d in aug_dirs:
            a_path = os.path.join(aug_dir_base, d, rel_path)
            if os.path.exists(a_path):
                img_a = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
                if img_a is not None:
                    aug_entropies_dict[d].append(calculate_2d_entropy(img_a))

    mu_orig = np.mean(orig_entropies)
    V_orig = np.var(orig_entropies)
    target_V = (1 + 0.1) * V_orig

    print("\n" + "="*40)
    print("Original Dataset Feature Statistics")
    print(f"Mean  mu = {mu_orig:.4f}")
    print(f"Var   V  = {V_orig:.4f}")
    print(f"Target (1+Delta)V = {target_V:.4f}")

    def calc_loss(aug_entropies, name):
        if len(aug_entropies) == 0: return float('inf')
        mu_aug = np.mean(aug_entropies)
        V_aug = np.var(aug_entropies)

        loss_u = abs(mu_aug - mu_orig)
        loss_v = abs(V_aug - target_V)
        total_loss = loss_u + loss_v

        print("-" * 40)
        print(f"Augmented Dataset: {name}")
        print(f"Mean  mu' = {mu_aug:.4f}  (deviation: {loss_u:.4f})")
        print(f"Var   V'  = {V_aug:.4f}  (deviation: {loss_v:.4f})")
        print(f"MHOA Loss = |mu'-mu| + |V'-1.1V| = {total_loss:.4f}")
        return total_loss

    losses = []
    best_loss = float('inf')
    best_name = None

    for d in aug_dirs:
        l = calc_loss(aug_entropies_dict[d], d)
        losses.append(l)
        if l < best_loss:
            best_loss = l
            best_name = d

    print("=" * 40)
    print(f"Best augmented subset: {best_name}")
    print(f"Lowest MHOA loss: {best_loss:.4f}")

if __name__ == "__main__":
    from lasma_config import get_path
    orig_dir = get_path("train")
    offline_dir_base = get_path("lasma_aug")
    evaluate_dataset_mhoa(orig_dir, offline_dir_base)
