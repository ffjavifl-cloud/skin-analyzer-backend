"""Offline/dev utility: convert .avif images in ./data to .jpg.

NOT part of the deployed service. Requires an extra dev dependency:
    pip install pillow-avif-plugin
Run from a folder that contains a `data/` directory of images.
"""
import os
import warnings
from PIL import Image
import pillow_avif  # noqa: F401  (registers AVIF support)

DATA_DIR = "data"
INPUT_EXT = ".avif"
OUTPUT_EXT = ".jpg"

for root, _dirs, files in os.walk(DATA_DIR):
    for fname in files:
        if fname.lower().endswith(INPUT_EXT):
            fpath = os.path.join(root, fname)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    img = Image.open(fpath).convert("RGB")
                new_path = os.path.join(root, fname[: -len(INPUT_EXT)] + OUTPUT_EXT)
                img.save(new_path, "JPEG")
                print(f"converted: {fpath} -> {new_path}")
            except Exception as e:
                print(f"error converting {fpath}: {e}")
