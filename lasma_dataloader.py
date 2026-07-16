"""Online PyTorch DataLoader utilities for LASMA augmentation."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from region_partition import build_tri_regions, extract_control_points, generate_mock_oct_data
from lasma_optimizer import build_graph, get_displacement_limits_and_boundaries, optimize_lasma_v2
from lasma_config import LASMAConfig, get_path


class LASMA_Augmentation:
    """LASMA online augmentation transform."""

    def __init__(
        self,
        config: LASMAConfig = None,
        d_pixels: int = None,
        grid_size: int = None,
        r1: float = None,
        r2: float = None,
        r3: float = None,
        epochs: int = None,
        device: str = "cpu",
    ):
        """Initialize the transform and optionally override config values."""
        self.device = device

        if config is not None:
            self.cfg = config
        else:
            self.cfg = LASMAConfig()

        if d_pixels is not None:
            self.cfg.d_pixels = d_pixels
        if grid_size is not None:
            self.cfg.grid_size = grid_size
        if r1 is not None:
            self.cfg.r1 = r1
        if r2 is not None:
            self.cfg.r2 = r2
        if r3 is not None:
            self.cfg.r3 = r3
        if epochs is not None:
            self.cfg.epochs = epochs

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply LASMA augmentation to a single OCT image."""
        H, W = image.shape

        R1, R2, R3 = build_tri_regions(mask, d_pixels=self.cfg.d_pixels)

        P_all = extract_control_points(image, mask, grid_size=self.cfg.grid_size)

        limits, is_bound = get_displacement_limits_and_boundaries(
            P_all, R1, R2, R3, H, W,
            r1=self.cfg.r1, r2=self.cfg.r2, r3=self.cfg.r3,
        )

        edges = build_graph(P_all)

        augmented_image, _ = optimize_lasma_v2(
            img_np=image,
            limits_np=limits,
            is_boundary_np=is_bound,
            P_np=P_all,
            edges=edges,
            R1_np=R1,
            config=self.cfg,
            device=self.device,
        )

        return augmented_image


class MockOCTDataset(Dataset):
    """Mock OCT dataset for DataLoader pipeline testing."""

    def __init__(self, num_samples=4, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        self.base_img, self.base_mask = generate_mock_oct_data()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.base_img.copy()
        mask = self.base_mask.copy()

        img = cv2.add(
            img,
            np.random.randint(-10, 10, img.shape, dtype=np.int16).astype(np.uint8),
        )

        if self.transform:
            img_aug = self.transform(img, mask)
            return (
                torch.tensor(img_aug, dtype=torch.float32).unsqueeze(0),
                torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            )
        else:
            return (
                torch.tensor(img, dtype=torch.float32).unsqueeze(0),
                torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lasma_transform = LASMA_Augmentation(
        d_pixels=20,
        grid_size=25,
        r1=0.5,
        r2=1.5,
        r3=3.0,
        epochs=30,
        device=device,
    )

    dataset = MockOCTDataset(num_samples=4, transform=lasma_transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    print(f"Testing DataLoader batch generation on {device} (piecewise affine v3)...")

    for batch_idx, (imgs, masks) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}: loaded {imgs.shape[0]} augmented OCT images.")

        plt.figure(figsize=(10, 5))
        for i in range(imgs.shape[0]):
            plt.subplot(1, imgs.shape[0], i + 1)
            plt.title(f"Augmented Image {i + 1}")
            img_show = imgs[i].squeeze().numpy()
            mask_show = masks[i].squeeze().numpy()
            plt.imshow(img_show, cmap="gray")
            plt.contour(mask_show, colors="r", linewidths=0.5)

        plt.tight_layout()
        output_dir = get_path("debug")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "lasma_dataloader_batch.png")
        plt.savefig(output_path, dpi=300)
        print(f"Batch {batch_idx + 1} visualization saved to {output_path}")
        break
