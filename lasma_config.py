"""Central configuration for LASMA.

All default paths are relative to this project folder so the repository can be
uploaded to GitHub without machine-specific absolute paths. For real datasets,
override the defaults with environment variables such as LASMA_DATA_ROOT.
"""

import os
from dataclasses import dataclass


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get(
    "LASMA_DATA_ROOT",
    os.path.join(PROJECT_ROOT, "data", "oct_split"),
)
OUTPUT_ROOT = os.environ.get(
    "LASMA_OUTPUT_ROOT",
    os.path.join(PROJECT_ROOT, "outputs"),
)
EXTERNAL_DATA_ROOT = os.environ.get(
    "LASMA_EXTERNAL_DATA_ROOT",
    os.path.join(PROJECT_ROOT, "data", "external"),
)


PATHS = {
    # Dataset paths. These are the only paths users usually need to change.
    "dataset_root": DATA_ROOT,
    "dataset_7": DATA_ROOT,
    "train": os.path.join(DATA_ROOT, "train"),
    "val": os.path.join(DATA_ROOT, "val"),
    "test": os.path.join(DATA_ROOT, "test"),
    "ext_srinivasan": os.path.join(EXTERNAL_DATA_ROOT, "Srinivasan_2014"),
    "ext_split": os.path.join(PROJECT_ROOT, "data", "srinivasan_split"),

    # Generated outputs.
    "outputs": OUTPUT_ROOT,
    "debug": os.path.join(OUTPUT_ROOT, "debug"),
    "lasma_output": os.path.join(OUTPUT_ROOT, "lasma_output"),
    "lasma_aug": os.path.join(OUTPUT_ROOT, "lasma_aug"),
    "lasma_merged": os.path.join(OUTPUT_ROOT, "merged_dataset"),
    "lasma_merged_train": os.path.join(OUTPUT_ROOT, "merged_dataset", "train"),
    "lasma_1750": os.path.join(OUTPUT_ROOT, "LASMA_1750"),
    "vanilla_2064": os.path.join(OUTPUT_ROOT, "Vanilla_2064"),
    "srinivasan_train": os.path.join(DATA_ROOT, "train"),
    "srinivasan_val": os.path.join(DATA_ROOT, "val"),
    "srinivasan_test": os.path.join(DATA_ROOT, "test"),
    "srinivasan_aug": os.path.join(OUTPUT_ROOT, "Srinivasan_Aug"),
}


def get_path(key: str) -> str:
    """Return a configured path by key."""
    if key not in PATHS:
        raise KeyError(f"Unknown path key '{key}'. Available keys: {list(PATHS.keys())}")
    return PATHS[key]


@dataclass
class LASMAConfig:
    """Hyperparameters for LASMA augmentation."""

    # Region construction.
    d_pixels: int = 20
    local_margin_pixels: int = 8

    # Control point sampling.
    grid_size: int = 20

    # Region-specific displacement bounds in pixels.
    r1: float = 0.5
    r2: float = 1.5
    r3: float = 3.0

    # Optimization.
    epochs: int = 120
    lr: float = 0.1
    delta_variance: float = 0.1
    noise_scale: float = 0.1
    target_disp: float = 2.5

    # Loss weights.
    w_dist: float = 10.0
    w_rel: float = 50.0
    w_smooth: float = 80.0
    w_mag: float = 3.0
    w_lesion: float = 50.0
    w_drive: float = 30.0
