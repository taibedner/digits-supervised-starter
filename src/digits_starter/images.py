"""Optional PNG/JPEG input; keep intensity conventions explicit."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def image_features(path, invert=False):
    """Resize to 8x8, map 0..255 to the dataset's 0..16, flatten to (1, 64)."""
    with Image.open(path) as image:
        image = image.convert("L")
        if invert:
            image = ImageOps.invert(image)
        image = image.resize((8, 8), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float64).reshape(1, 64) * (16.0 / 255.0)


def export_digit(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(np.asarray(image) * (255.0 / 16.0)).clip(0, 255).astype(np.uint8)
    with path.open("xb") as handle:
        Image.fromarray(pixels).save(handle, format="PNG")

