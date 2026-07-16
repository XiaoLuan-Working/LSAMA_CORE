"""Generate an offline LASMA-augmented dataset."""

import os
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm

from region_partition import extract_control_points, build_tri_regions
from pseudo_mask import generate_improved_pseudo_mask
from lasma_optimizer import (
    build_local_masks,
    get_displacement_limits_and_boundaries,
    build_graph,
    get_lesion_control_indices,
    optimize_lasma_v2,
)
from lasma_config import LASMAConfig
from lasma_config import get_path

DEBUG_DIR = get_path("debug")


def ensure_debug_dir():
    os.makedirs(DEBUG_DIR, exist_ok=True)


def save_tri_regions_debug(R1, R2, R3):
    tri_vis = np.zeros((*R1.shape, 3), dtype=np.uint8)
    tri_vis[R1 > 0] = (0, 0, 255)  # Red: lesion core
    tri_vis[R2 > 0] = (0, 255, 0)  # Green: margin
    tri_vis[R3 > 0] = (255, 0, 0)  # Blue: non-lesion
    cv2.imwrite(os.path.join(DEBUG_DIR, "02_tri_regions.png"), tri_vis)


def save_control_points_debug(img, points, lesion_idx=None):
    """Save a control-point visualization."""
    point_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, pt in enumerate(points):
        color = (255, 0, 0) if (lesion_idx is not None and i in lesion_idx) else (0, 0, 255)
        cv2.circle(point_vis, (int(pt[0]), int(pt[1])), 2, color, -1)
    cv2.imwrite(os.path.join(DEBUG_DIR, "03_control_points.png"), point_vis)


def grid_to_flow(grid, H, W):
    """Compute flow offsets from a normalized sampling grid."""
    grid_np = grid.detach().cpu().squeeze().numpy()
    map_x = (grid_np[:, :, 0] + 1.0) * 0.5 * (W - 1)
    map_y = (grid_np[:, :, 1] + 1.0) * 0.5 * (H - 1)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    return map_x - xs, map_y - ys


def save_grid_debug(img, grid, step=16):
    """Save a deformation-grid visualization."""
    H, W = img.shape
    grid_np = grid.detach().cpu().squeeze().numpy()
    map_x = (grid_np[:, :, 0] + 1.0) * 0.5 * (W - 1)
    map_y = (grid_np[:, :, 1] + 1.0) * 0.5 * (H - 1)
    grid_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for y in range(0, H, step):
        pts = np.stack([map_x[y, ::step], map_y[y, ::step]], axis=1).astype(np.int32)
        cv2.polylines(grid_vis, [pts], False, (0, 255, 255), 1)

    for x in range(0, W, step):
        pts = np.stack([map_x[::step, x], map_y[::step, x]], axis=1).astype(np.int32)
        cv2.polylines(grid_vis, [pts], False, (0, 255, 255), 1)

    cv2.imwrite(os.path.join(DEBUG_DIR, "04_deformation_grid.png"), grid_vis)


def save_flow_heatmap_debug(flow_x, flow_y):
    mag = np.sqrt(flow_x**2 + flow_y**2)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(DEBUG_DIR, "05_flow_heatmap.png"), heatmap)


def save_mask_debug(mask, filename):
    cv2.imwrite(os.path.join(DEBUG_DIR, filename), mask)


def generate_offline_dataset(
    input_dir: str,
    output_dir: str,
    augment_times: int = 3,
    device: str = "cpu",
    resume: bool = False,
    config: LASMAConfig = None,
):
    """Generate offline LASMA augmentations for a dataset."""
    cfg = config if config is not None else LASMAConfig()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ensure_debug_dir()

    image_paths = glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True)
    print(f"Found {len(image_paths)} images, LASMA x{augment_times} (piecewise affine mode) ...")

    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (256, 256))
        H, W = img.shape
        rel_path = os.path.relpath(img_path, input_dir)

        try:
            mask = generate_improved_pseudo_mask(img)
            lesion_mask, preserve_mask, nonlesion_mask = build_local_masks(
                mask, margin_pixels=cfg.local_margin_pixels
            )
            save_mask_debug(lesion_mask * 255, "01_mask.png")
            save_mask_debug(preserve_mask * 255, "01a_preserve_mask.png")
            save_mask_debug(nonlesion_mask * 255, "01b_nonlesion_mask.png")

            R1, R2, R3 = build_tri_regions(mask, d_pixels=cfg.d_pixels)
            save_tri_regions_debug(R1, R2, R3)

            P_all = extract_control_points(img, mask, grid_size=cfg.grid_size)

            lesion_idx = get_lesion_control_indices(P_all, R1)
            save_control_points_debug(img, P_all, lesion_idx=set(lesion_idx.tolist()))

            limits, is_bound = get_displacement_limits_and_boundaries(
                P_all, R1, R2, R3, H, W, r1=cfg.r1, r2=cfg.r2, r3=cfg.r3
            )
            edges = build_graph(P_all)
        except Exception as e:
            print(f"Error extracting features for {img_path}: {e}")
            continue

        for i in range(augment_times):
            save_dir = os.path.join(output_dir, f"aug_{i+1}", os.path.dirname(rel_path))
            os.makedirs(save_dir, exist_ok=True)

            save_name = os.path.basename(img_path)
            if resume and os.path.exists(os.path.join(save_dir, save_name)):
                continue

            try:
                final_img, P_prime, debug = optimize_lasma_v2(
                    img,
                    limits,
                    is_bound,
                    P_all,
                    edges,
                    R1_np=R1,
                    config=cfg,
                    device=device,
                    return_debug=True,
                )

                grid = debug["grid"]
                flow_x, flow_y = grid_to_flow(grid, H, W)
                save_grid_debug(img, grid)
                save_flow_heatmap_debug(flow_x, flow_y)

                diff = cv2.absdiff(img, final_img)
                cv2.imwrite(os.path.join(DEBUG_DIR, "06_difference.png"), diff)
                cv2.imwrite(os.path.join(DEBUG_DIR, "07_final_aug.png"), final_img)

                cv2.imwrite(os.path.join(save_dir, save_name), final_img)
            except Exception as e:
                print(f"Error augmenting {img_path} aug_{i+1}: {e}")
                continue


if __name__ == "__main__":
    from lasma_config import get_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    INPUT_DATASET = get_path("train")
    OUTPUT_DATASET = get_path("lasma_aug")

    cfg = LASMAConfig(
        epochs=150,
        target_disp=3.5,
        w_drive=50,
        w_smooth=80,
        w_mag=2,
        noise_scale=0.15,
        grid_size=12,
    )
    generate_offline_dataset(INPUT_DATASET, OUTPUT_DATASET, augment_times=10, device=device, config=cfg)
    print("Offline dataset generation complete! All LASMA augmentations saved to disk.")
