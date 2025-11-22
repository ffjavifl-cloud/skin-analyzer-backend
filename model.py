import numpy as np
import cv2
from PIL import Image
import json
import os

# Cargar calibración si existe
calibration_path = os.path.join(os.path.dirname(__file__), "calibration.json")
if os.path.exists(calibration_path):
    with open(calibration_path, "r") as f:
        calibration = json.load(f)
else:
    calibration = {}

def pil_to_cv(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def normalize(value: float, low: float, high: float, param: str = None) -> float:
    if param and param in calibration:
        low = calibration[param].get("low", low)
        high = calibration[param].get("high", high)
    if high == low:
        return 0.0
    scaled = (value - low) / (high - low)
    return float(np.clip(scaled * 10.0, 0.0, 10.0))

def predict_scores(image: Image.Image, debug: bool = False) -> dict:
    img = pil_to_cv(image)
    h, w = img.shape[:2]

    # Redimensionar para estabilidad
    target = 640
    scale = target / max(h, w)
    img_resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Recorte central
    rh, rw = img_resized.shape[:2]
    pad = int(0.08 * max(rh, rw))
    r = img_resized[pad:rh - pad, pad:rw - pad]

    # Espacios de color
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Brightness
    brightness_val = float(np.mean(l_channel))
    brightness_score = 10.0 - normalize(brightness_val, 40, 180, "brightness")

    # Dryness (contraste local)
    l_norm = l_channel.astype(np.float32) / 255.0
    local_mean = cv2.GaussianBlur(l_norm, (0, 0), 3)
    local_var = cv2.GaussianBlur((l_norm - local_mean) ** 2, (0, 0), 3)
    dryness_metric = float(np.mean(local_var))
    dryness_score = normalize(dryness_metric, 0.0005, 0.01, "dryness")

    # Texture-pores (Laplaciano)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    pores_metric = float(np.mean(np.abs(lap)))
    texture_pores_score = normalize(pores_metric, 0.5, 5.0, "texture-pores")

    # Lines (bordes finos)
    edges_fine = cv2.Canny(gray, 30, 90)
    lines_metric = float(np.sum(edges_fine > 0)) / edges_fine.size
    lines_score = normalize(lines_metric, 0.005, 0.08, "lines")

    # Wrinkles (bordes duros)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.8)
    edges_hard = cv2.Canny(blur, 80, 160)
    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(edges_hard, cv2.MORPH_CLOSE, kernel, iterations=1)
    wrinkles_metric = float(np.sum(edges_closed > 0)) / edges_closed.size
    wrinkles_score = normalize(wrinkles_metric, 0.01, 0.12, "wrinkles")

    # Pigmentation (varianza + skewness)
    a = a_channel.astype(np.float32)
    b = b_channel.astype(np.float32)
    var_ab = float(np.var(a) + np.var(b))

    def skew(x):
        x = x.flatten().astype(np.float32)
        m = np.mean(x)
        s = np.std(x) + 1e-6
        return float(np.mean(((x - m) / s) ** 3))

    skew_ab = np.clip(abs(skew(a)) + abs(skew(b)), 0, 2)
    pigmentation_metric = var_ab / (255.0 ** 2) + 0.2 * skew_ab
    pigmentation_score = normalize(pigmentation_metric, 0.002, 0.12, "pigmentation")

    if debug:
        print("Brightness:", brightness_val)
        print("Dryness:", dryness_metric)
        print("Texture-pores:", pores_metric)
        print("Lines:", lines_metric)
        print("Wrinkles:", wrinkles_metric)
        print("Pigmentation:", pigmentation_metric)

    return {
        "brightness": round(brightness_score, 1),
        "dryness": round(dryness_score, 1),
        "lines": round(lines_score, 1),
        "pigmentation": round(pigmentation_score, 1),
        "texture-pores": round(texture_pores_score, 1),
        "wrinkles": round(wrinkles_score, 1),
    }
