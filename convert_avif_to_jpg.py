import os
from PIL import Image
import pillow_avif  # Activar soporte para imágenes .avif
import warnings

DATA_DIR = "data"
INPUT_EXT = ".avif"
OUTPUT_EXT = ".jpg"

for root, dirs, files in os.walk(DATA_DIR):
    for fname in files:
        if fname.lower().endswith(INPUT_EXT):
            fpath = os.path.join(root, fname)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    img = Image.open(fpath).convert("RGB")
                new_name = fname.replace(INPUT_EXT, OUTPUT_EXT)
                new_path = os.path.join(root, new_name)
                img.save(new_path, "JPEG")
                print(f"✅ Convertido: {fpath} → {new_path}")
            except Exception as e:
                print(f"⚠️ Error al convertir {fpath}: {e}")
