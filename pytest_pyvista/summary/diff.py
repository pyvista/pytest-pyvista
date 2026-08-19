"""Pixel-difference images for the summary report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

# Summed absolute channel difference above which a pixel counts as changed.
DIFF_PIXEL_THRESHOLD = 8

# Magenta stays distinguishable under the common colour-vision deficiencies.
DIFF_COLOR = (255, 0, 255)

# How much of the baseline's luminance survives behind the overlay.
_FADE = 0.35


@dataclass(frozen=True)
class DiffResult:
    """The outcome of diffing one baseline against one generated image."""

    image: Image.Image | None
    baseline_size: tuple[int, int]
    generated_size: tuple[int, int]

    @property
    def size_mismatch(self) -> bool:
        """Whether the two images differ in size, which leaves a pixel difference undefined."""
        return self.baseline_size != self.generated_size


def compute_diff_image(baseline_path: Path, generated_path: Path) -> DiffResult:
    """
    Build a difference image highlighting where two renders disagree.

    Changed pixels are painted in ``DIFF_COLOR`` over a faded greyscale copy of the
    baseline, so unchanged structure stays legible behind the overlay.

    Both source sizes are always reported, so that a caller can describe a mismatch in
    numbers. When the two differ a pixel difference is undefined, so ``image`` is ``None``.
    """
    with Image.open(baseline_path) as baseline_file, Image.open(generated_path) as generated_file:
        baseline = baseline_file.convert("RGB")
        generated = generated_file.convert("RGB")

    if baseline.size != generated.size:
        return DiffResult(None, baseline.size, generated.size)

    delta = np.abs(np.asarray(baseline, dtype=np.int16) - np.asarray(generated, dtype=np.int16)).sum(axis=2)
    changed = delta > DIFF_PIXEL_THRESHOLD

    faded = np.asarray(baseline.convert("L").convert("RGB"), dtype=np.float32)
    faded = faded * _FADE + 255.0 * (1.0 - _FADE)
    canvas = faded.astype(np.uint8)
    canvas[changed] = DIFF_COLOR

    return DiffResult(Image.fromarray(canvas), baseline.size, generated.size)
