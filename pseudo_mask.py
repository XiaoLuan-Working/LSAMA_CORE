import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from lasma_config import get_path

def generate_improved_pseudo_mask(img_gray):
    blurred = cv2.medianBlur(img_gray, 7)

    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    retina_body = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel_large)
    _, retina_mask = cv2.threshold(retina_body, 30, 255, cv2.THRESH_BINARY)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel_small)

    _, grad_mask = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    lesion_kpts = cv2.bitwise_and(grad_mask, retina_mask)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    lesion_mask = cv2.morphologyEx(lesion_kpts, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(lesion_mask, connectivity=8)
    final_mask = np.zeros_like(lesion_mask)
    if num_labels > 1:
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 300:
                final_mask[labels == i] = 255

    return final_mask

def test_improved_mask_extractor():
    train_dir = get_path("train")
    amd_imgs = glob.glob(os.path.join(train_dir, "AMD", "*.jpg"))[:2]
    dme_imgs = glob.glob(os.path.join(train_dir, "DME", "*.jpg"))[:2]
    test_imgs = amd_imgs + dme_imgs

    if len(test_imgs) == 0:
        print(f"No test images found under {train_dir}. Set LASMA_DATA_ROOT or place data under data/oct_split.")
        return

    plt.figure(figsize=(12, 4 * len(test_imgs)))

    for i, img_path in enumerate(test_imgs):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))

        mask = generate_improved_pseudo_mask(img)

        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        overlay[mask == 255] = [0, 0, 255]
        blended = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.7, overlay, 0.3, 0)

        plt.subplot(len(test_imgs), 3, i * 3 + 1)
        plt.title(f"Original: {filename}")
        plt.imshow(img, cmap='gray')

        plt.subplot(len(test_imgs), 3, i * 3 + 2)
        plt.title("Extracted Pseudo Mask")
        plt.imshow(mask, cmap='gray')

        plt.subplot(len(test_imgs), 3, i * 3 + 3)
        plt.title("Overlay Check (Red = Locked Zone)")
        plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    output_dir = get_path("debug")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pseudo_mask_check.png")
    plt.savefig(output_path, dpi=300)
    print(f"Improved mask extraction test done, saved as {output_path}")

if __name__ == '__main__':
    test_improved_mask_extractor()
