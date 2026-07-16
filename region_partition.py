import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from lasma_config import get_path

def generate_mock_oct_data():
    image = np.zeros((256, 256), dtype=np.uint8)
    image[50:200, :] = 100
    image[100:150, :] = 150

    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(mask, (128, 120), (30, 15), 0, 0, 360, 255, -1)

    return image, mask

def build_tri_regions(mask, d_pixels=15):
    R1 = mask.copy()

    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=d_pixels)
    R2 = cv2.subtract(dilated_mask, mask)

    full_img = np.ones_like(mask) * 255
    R3 = full_img.copy()
    R3[R1 == 255] = 0
    R3[R2 == 255] = 0

    return R1, R2, R3

def extract_control_points(image, mask, grid_size=16):
    points = []
    h, w = image.shape

    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            x_coord = min(x, w - 1)
            y_coord = min(y, h - 1)
            points.append([x_coord, y_coord])

    for corner in [[0,0], [0, h-1], [w-1, 0], [w-1, h-1]]:
        if corner not in points:
            points.append(corner)

    edges = cv2.Canny(mask, 100, 200)
    y_edge, x_edge = np.where(edges > 0)
    for x, y in zip(x_edge[::5], y_edge[::5]):
        points.append([x, y])

    return np.array(points, dtype=np.float32)

def filter_control_points_by_mask(P_np, valid_mask_np):
    keep = []
    for pt in P_np:
        x = int(np.clip(pt[0], 0, valid_mask_np.shape[1] - 1))
        y = int(np.clip(pt[1], 0, valid_mask_np.shape[0] - 1))
        keep.append(valid_mask_np[y, x] > 0)
    keep = np.array(keep, dtype=bool)
    return P_np[keep], keep

def assign_region_weights(points, R1, R2, R3, alpha1=0.1, alpha2=0.5, alpha3=1.0):
    weights = []
    colors = []

    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        y = np.clip(y, 0, R1.shape[0]-1)
        x = np.clip(x, 0, R1.shape[1]-1)

        if R1[y, x] > 0:
            weights.append(alpha1)
            colors.append('red')
        elif R2[y, x] > 0:
            weights.append(alpha2)
            colors.append('yellow')
        else:
            weights.append(alpha3)
            colors.append('green')

    return np.array(weights), colors

if __name__ == "__main__":
    img_mock, mask_mock = generate_mock_oct_data()

    R1, R2, R3 = build_tri_regions(mask_mock, d_pixels=20)

    P = extract_control_points(img_mock, mask_mock, grid_size=20)

    weights, pt_colors = assign_region_weights(P, R1, R2, R3, alpha1=0.1, alpha2=0.5, alpha3=1.0)

    plt.figure(figsize=(15, 5))

    rgb_regions = np.zeros((img_mock.shape[0], img_mock.shape[1], 3), dtype=np.uint8)
    rgb_regions[R1 > 0] = [255, 0, 0]
    rgb_regions[R2 > 0] = [255, 255, 0]
    rgb_regions[R3 > 0] = [0, 255, 0]

    plt.subplot(1, 3, 1)
    plt.title("Simulated OCT & Mask")
    plt.imshow(img_mock, cmap='gray')
    plt.contour(mask_mock, colors='r', linewidths=1.5)

    plt.subplot(1, 3, 2)
    plt.title("Tri-Region (Red:R1, Yellow:R2, Green:R3)")
    plt.imshow(rgb_regions)

    plt.subplot(1, 3, 3)
    plt.title("Control Points P w/ Assigned Weights")
    plt.imshow(img_mock, cmap='gray')
    plt.scatter(P[:, 0], P[:, 1], c=pt_colors, s=15, edgecolors='white', linewidths=0.5)

    plt.tight_layout()
    output_dir = get_path("debug")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "region_partition_check.png")
    plt.savefig(output_path, dpi=300)
    print(f"Step 1 done. Visualization saved as {output_path}")
