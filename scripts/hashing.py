"""Perceptual hashing shared by the split script and the leakage check."""

from pathlib import Path

import imagehash
from PIL import Image, ImageOps

# Max Hamming distance (out of 64 bits) at which two images count as copies.
HASH_THRESHOLD = 6


def image_hashes(p: Path) -> list:
    """pHash of the image and of its horizontal mirror, so flipped copies also match."""
    with Image.open(p) as im:
        im = im.convert("RGB")
        return [imagehash.phash(im), imagehash.phash(ImageOps.mirror(im))]
