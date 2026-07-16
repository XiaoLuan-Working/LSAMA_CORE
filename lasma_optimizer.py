"""Core LASMA optimizer using Delaunay piecewise affine warping."""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from typing import Optional
import glob
import os

from pseudo_mask import generate_improved_pseudo_mask
from region_partition import build_tri_regions, extract_control_points, filter_control_points_by_mask
from entropy_features import compute_distribution_loss

# =============================================================================
# =============================================================================


class PiecewiseAffineWarp:
    """Piecewise affine warp based on Delaunay triangulation."""

    def __init__(
        self,
        control_points_np: np.ndarray,
        H: int,
        W: int,
        device: str = "cpu",
        preserve_mask_np: Optional[np.ndarray] = None,
    ):
        """Store precomputed warp state."""
        self.device = device
        self.H = H
        self.W = W

        self.tri = Delaunay(control_points_np)
        tri_simplices = self.tri.simplices

        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        pixel_coords = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float64)  # (H*W, 2)

        tri_indices = self.tri.find_simplex(pixel_coords)

        outside_np = tri_indices < 0
        tri_indices[outside_np] = 0

        tri_verts = control_points_np[tri_simplices]  # (K, 3, 2)
        v0 = tri_verts[tri_indices, 0]  # (H*W, 2)
        v1 = tri_verts[tri_indices, 1]
        v2 = tri_verts[tri_indices, 2]

        e1_x = v1[:, 0] - v0[:, 0]  # (H*W,)
        e1_y = v1[:, 1] - v0[:, 1]
        e2_x = v2[:, 0] - v0[:, 0]
        e2_y = v2[:, 1] - v0[:, 1]

        denom = e1_x * e2_y - e1_y * e2_x
        denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)

        dp_x = pixel_coords[:, 0] - v0[:, 0]
        dp_y = pixel_coords[:, 1] - v0[:, 1]

        c1 = (e2_y * dp_x - e2_x * dp_y) / denom
        c2 = (e1_x * dp_y - e1_y * dp_x) / denom
        c0 = 1.0 - c1 - c2

        bary = np.stack([c0, c1, c2], axis=-1).astype(np.float32)  # (H*W, 3)

        if preserve_mask_np is not None:
            self.preserve_mask_flat = torch.tensor(
                preserve_mask_np.reshape(-1) > 0, dtype=torch.bool, device=device
            )
        else:
            self.preserve_mask_flat = torch.zeros(H * W, dtype=torch.bool, device=device)

        self.tri_vertices = torch.tensor(tri_simplices, dtype=torch.long, device=device)  # (K, 3)
        self.tri_map = torch.tensor(tri_indices, dtype=torch.long, device=device)  # (H*W,)
        self.bary_coords = torch.tensor(bary, dtype=torch.float32, device=device)  # (H*W, 3)
        self.outside_mask = torch.tensor(outside_np, dtype=torch.bool, device=device)  # (H*W,)

        self.identity_flat = torch.tensor(
            pixel_coords, dtype=torch.float32, device=device
        )  # (H*W, 2)

    def get_grid(self, P_current: torch.Tensor, image_shape: tuple) -> torch.Tensor:
        """Generate a normalized sampling grid from current control points."""
        v0 = P_current[self.tri_vertices[:, 0]]  # (K, 2)
        v1 = P_current[self.tri_vertices[:, 1]]
        v2 = P_current[self.tri_vertices[:, 2]]

        v0_pix = v0[self.tri_map]  # (H*W, 2)
        v1_pix = v1[self.tri_map]
        v2_pix = v2[self.tri_map]

        b0 = self.bary_coords[:, 0:1]
        b1 = self.bary_coords[:, 1:2]
        b2 = self.bary_coords[:, 2:3]
        mapped = b0 * v0_pix + b1 * v1_pix + b2 * v2_pix  # (H*W, 2)

        mapped[self.preserve_mask_flat] = self.identity_flat[self.preserve_mask_flat]

        mapped[self.outside_mask] = self.identity_flat[self.outside_mask]

        grid_x = mapped[:, 0] / max(self.W - 1, 1) * 2 - 1
        grid_y = mapped[:, 1] / max(self.H - 1, 1) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, self.H, self.W, 2)
        return grid


# =============================================================================
# =============================================================================


def build_graph(points: np.ndarray) -> list:
    """Build a Delaunay control-point adjacency graph."""
    tri = Delaunay(points)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = simplex[i], simplex[j]
                edges.add((min(u, v), max(u, v)))
    return list(edges)


def build_local_masks(lesion_mask_np: np.ndarray, margin_pixels: int = 8):
    """Build lesion, preserve, and non-lesion masks."""
    lesion_mask = (lesion_mask_np > 0).astype(np.uint8)
    kernel_size = margin_pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    preserve_mask = cv2.dilate(lesion_mask, kernel, iterations=1)
    preserve_mask = (preserve_mask > 0).astype(np.uint8)
    nonlesion_mask = (1 - preserve_mask).astype(np.uint8)
    return lesion_mask, preserve_mask, nonlesion_mask


def get_lesion_control_indices(P_np: np.ndarray, R1_np: np.ndarray) -> np.ndarray:
    """Return indices of control points inside the lesion-core mask."""
    lesion_idx = []
    H, W = R1_np.shape
    for i, pt in enumerate(P_np):
        x = int(np.clip(pt[0], 0, W - 1))
        y = int(np.clip(pt[1], 0, H - 1))
        if R1_np[y, x] > 0:
            lesion_idx.append(i)
    return np.array(lesion_idx, dtype=np.int64)


def get_displacement_limits_and_boundaries(
    P: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    R3: np.ndarray,
    H: int,
    W: int,
    r1: float = 2.0,
    r2: float = 5.0,
    r3: float = 10.0,
):
    """Assign displacement limits and image-border flags to control points."""
    limits = []
    is_boundary = []
    for pt in P:
        x, y = int(pt[0]), int(pt[1])
        x_c = np.clip(x, 0, W - 1)
        y_c = np.clip(y, 0, H - 1)

        if R1[y_c, x_c] > 0:
            limits.append(r1)
        elif R2[y_c, x_c] > 0:
            limits.append(r2)
        else:
            limits.append(r3)

        if x <= 5 or x >= W - 5 or y <= 5 or y >= H - 5:
            is_boundary.append(True)
        else:
            is_boundary.append(False)

    return np.array(limits, dtype=np.float32), np.array(is_boundary, dtype=bool)


def compute_image_gradients(tensor: torch.Tensor) -> torch.Tensor:
    """Compute Sobel gradient magnitude."""
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=tensor.device
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=tensor.device
    ).view(1, 1, 3, 3)
    gx = F.conv2d(tensor, kx, padding=1)
    gy = F.conv2d(tensor, ky, padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-8)


# =============================================================================
# =============================================================================


def optimize_lasma_v2(
    img_np: np.ndarray,
    limits_np: np.ndarray,
    is_boundary_np: np.ndarray,
    P_np: np.ndarray,
    edges: list,
    R1_np: Optional[np.ndarray] = None,
    config=None,
    epochs: Optional[int] = None,
    device: str = "cpu",
    return_debug: bool = False,
):
    """Optimize LASMA control point displacements."""
    from lasma_config import LASMAConfig, get_path

    cfg = config if config is not None else LASMAConfig()
    if epochs is None:
        epochs = cfg.epochs

    H, W_img = img_np.shape
    img_tensor = (
        torch.tensor(img_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) / 255.0
    )

    if R1_np is not None:
        _, preserve_mask_np, _ = build_local_masks(
            R1_np, margin_pixels=cfg.local_margin_pixels
        )
    else:
        preserve_mask_np = None

    warp = PiecewiseAffineWarp(
        P_np, H, W_img, device=device, preserve_mask_np=preserve_mask_np
    )

    if R1_np is not None:
        lesion_idx = get_lesion_control_indices(P_np, preserve_mask_np)
        lesion_ctrl_mask = torch.zeros(len(P_np), dtype=torch.bool, device=device)
        if len(lesion_idx) > 0:
            lesion_ctrl_mask[torch.tensor(lesion_idx, dtype=torch.long, device=device)] = True
    else:
        lesion_ctrl_mask = torch.zeros(len(P_np), dtype=torch.bool, device=device)

    P = torch.tensor(P_np, dtype=torch.float32, device=device)  # (N, 2)
    r_limits = torch.tensor(limits_np, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1)
    boundary_mask = torch.tensor(
        ~is_boundary_np, dtype=torch.float32, device=device
    ).unsqueeze(1)

    delta_P_raw = torch.nn.Parameter(torch.randn_like(P) * cfg.noise_scale)
    optimizer = torch.optim.Adam([delta_P_raw], lr=cfg.lr)

    edges_local = build_graph(P_np)
    edges_u = torch.tensor([e[0] for e in edges_local], dtype=torch.long, device=device)
    edges_v = torch.tensor([e[1] for e in edges_local], dtype=torch.long, device=device)
    orig_dist = torch.norm(P[edges_u] - P[edges_v], dim=1)

    if len(edges_local) > 0:
        all_src = torch.cat([edges_u, edges_v])
        all_dst = torch.cat([edges_v, edges_u])
        degree = torch.zeros(len(P), dtype=torch.float32, device=device)
        degree.scatter_add_(0, all_dst, torch.ones_like(all_dst, dtype=torch.float32))
        degree = degree.clamp(min=1.0)
    else:
        all_src = torch.tensor([], dtype=torch.long, device=device)
        all_dst = torch.tensor([], dtype=torch.long, device=device)
        degree = torch.ones(len(P), dtype=torch.float32, device=device)

    if R1_np is not None:
        lesion_mask_t = (
            torch.tensor((preserve_mask_np > 0).astype(np.float32), device=device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        lesion_pixel_count = torch.sum(lesion_mask_t) + 1e-8
    else:
        lesion_mask_t = None
        lesion_pixel_count = None

    for _epoch in range(epochs):
        optimizer.zero_grad()

        delta_norm = torch.norm(delta_P_raw, dim=1, keepdim=True) + 1e-8
        scale = torch.min(torch.ones_like(delta_norm), r_limits / delta_norm)
        delta_P = delta_P_raw * scale

        delta_P = delta_P * boundary_mask

        delta_P[lesion_ctrl_mask] = 0.0

        P_prime = P + delta_P

        grid = warp.get_grid(P_prime, (H, W_img))
        img_prime = F.grid_sample(img_tensor, grid, mode="bilinear", align_corners=True)

        L_dist = compute_distribution_loss(img_tensor, img_prime, Delta=cfg.delta_variance)

        new_dist = torch.norm(P_prime[edges_u] - P_prime[edges_v], dim=1)
        L_rel = torch.mean((new_dist - orig_dist) ** 2)

        if len(all_src) > 0:
            neighbor_sum = torch.zeros_like(delta_P)
            neighbor_sum.scatter_add_(
                0,
                all_dst.unsqueeze(1).expand_as(delta_P[all_src]),
                delta_P[all_src],
            )
            neighbor_mean = neighbor_sum / degree.unsqueeze(1)
            L_smooth = torch.mean(torch.sum((delta_P - neighbor_mean) ** 2, dim=1))
        else:
            L_smooth = torch.tensor(0.0, device=device)

        L_mag = torch.mean(delta_P**2)

        if lesion_mask_t is not None:
            L_lesion = (
                torch.sum(lesion_mask_t * torch.abs(img_prime - img_tensor)) / lesion_pixel_count
            )
        else:
            L_lesion = torch.tensor(0.0, device=device)

        driving_force = torch.abs(torch.mean(torch.norm(delta_P, dim=1)) - cfg.target_disp)

        loss = (
            cfg.w_dist * L_dist
            + cfg.w_rel * L_rel
            + cfg.w_smooth * L_smooth
            + cfg.w_mag * L_mag
            + cfg.w_lesion * L_lesion
            + cfg.w_drive * driving_force
        )

        loss.backward()
        optimizer.step()

    final_img = np.clip(
        np.round(img_prime.detach().cpu().squeeze().numpy() * 255.0), 0, 255
    ).astype(np.uint8)

    if return_debug:
        debug_info = {
            "grid": grid.detach().cpu(),
            "aug_result": final_img.copy(),
            "P": P.detach().cpu().numpy(),
            "P_prime": P_prime.detach().cpu().numpy(),
            "lesion_ctrl_mask": lesion_ctrl_mask.detach().cpu().numpy(),
        }
        return final_img, P_prime.detach().cpu().numpy(), debug_info

    return final_img, P_prime.detach().cpu().numpy()


# =============================================================================
# =============================================================================

if __name__ == "__main__":
    import argparse
    from lasma_config import LASMAConfig, get_path

    parser = argparse.ArgumentParser(description="Run a single-image LASMA self-test.")
    parser.add_argument("--img", type=str, default=None,
                        help="Path to an OCT image. If omitted, the script searches configured demo paths.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(get_path("debug"), "lasma_optimizer_output.png"),
    )
    args = parser.parse_args()

    print("=== LASMA piecewise affine self-test ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.img:
        img_path = args.img
    else:
        candidates = glob.glob("test1-LASMA/aug_1/*.jpg")  # Windows workspace
        if not candidates:
            candidates = glob.glob("../test1-LASMA/aug_1/*.jpg")  # one level up
        if not candidates:
            candidates = glob.glob(os.path.join(get_path("train"), "*.jpg"))
        if not candidates:
            print("ERROR: no test image found. Please pass --img explicitly.")
            print("Example: python lasma_optimizer.py --img /path/to/oct_image.jpg")
            exit(1)
        img_path = candidates[0]

    print(f"Input: {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: failed to read {img_path}")
        exit(1)
    img = cv2.resize(img, (256, 256))
    H, W = img.shape

    mask = generate_improved_pseudo_mask(img)
    R1, R2, R3 = build_tri_regions(mask, d_pixels=20)
    P_all = extract_control_points(img, mask, grid_size=20)

    limits, is_bound = get_displacement_limits_and_boundaries(
        P_all, R1, R2, R3, H, W, r1=0.5, r2=1.5, r3=3.0
    )
    edges = build_graph(P_all)

    lesion_idx = get_lesion_control_indices(P_all, R1)
    print(
        f"Control points: {len(P_all)} total, "
        f"{len(lesion_idx)} in lesion zone (locked)"
    )

    cfg = LASMAConfig(epochs=args.epochs)
    final_img, P_prime, dbg = optimize_lasma_v2(
        img, limits, is_bound, P_all, edges, R1_np=R1, config=cfg, device=device, return_debug=True
    )

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Control Point Displacement\n(orange arrows)")
    plt.imshow(img, cmap="gray", alpha=0.5)
    disp = P_prime - P_all
    plt.quiver(
        P_all[:, 0],
        P_all[:, 1],
        disp[:, 0],
        disp[:, 1],
        color="orange",
        scale_units="xy",
        angles="xy",
        scale=1,
        width=0.005,
    )

    plt.subplot(1, 3, 3)
    plt.title("LASMA v3 (Piecewise Affine)")
    plt.imshow(final_img, cmap="gray")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, dpi=300)
    print(f"Done. Saved {args.output}")
