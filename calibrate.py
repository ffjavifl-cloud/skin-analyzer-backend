"""Calibrated skin-metric pipeline (live scoring path).

Computes raw image metrics with OpenCV, normalises them to a 0-10 scale using
ranges from calibration.json, then applies a light clinical attenuation so
scores don't pin unrealistically at the extremes.
"""
import os
import json
import logging

import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger("skin-analyzer.calibrate")

# -----------------------------------------------------------------------------
# Image utilities
# -----------------------------------------------------------------------------
def pil_to_cv(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def safe_center_crop(img: np.ndarray, pad_ratio: float = 0.08) -> np.ndarray:
    h, w = img.shape[:2]
    pad = int(pad_ratio * max(h, w))
    y1, y2 = max(0, pad), max(1, h - pad)
    x1, x2 = max(0, pad), max(1, w - pad)
    if y2 - y1 < 16 or x2 - x1 < 16:
        return img
    return img[y1:y2, x1:x2]


def to_lab_channels(bgr: np.ndarray):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    return l, a, b


# -----------------------------------------------------------------------------
# Calibration loading
# -----------------------------------------------------------------------------
DEFAULT_CALIB = {
    "brightness": {"low": 20.0, "high": 230.0},
    "dryness": {"low": 0.0005, "high": 0.02},
    "texture-pores": {"low": 0.2, "high": 4.5},
    "lines": {"low": 0.005, "high": 0.12},
    "wrinkles": {"low": 0.003, "high": 0.10},
    "pigmentation": {"low": 0.02, "high": 0.45},
}


def load_calibration(path: str = "calibration.json") -> dict:
    calib = {k: dict(v) for k, v in DEFAULT_CALIB.items()}
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            for k, v in data.items():
                if isinstance(v, dict) and "low" in v and "high" in v:
                    calib[k] = {"low": float(v["low"]), "high": float(v["high"])}
        else:
            logger.warning("calibration.json not found at %s; using defaults", path)
    except Exception:
        logger.exception("Could not load calibration; using defaults")
    return calib


# -----------------------------------------------------------------------------
# Normalisation 0-10
# -----------------------------------------------------------------------------
def normalize_to_10(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    scaled = (value - low) / (high - low)
    return float(np.clip(scaled * 10.0, 0.0, 10.0))


# -----------------------------------------------------------------------------
# Base metrics
# -----------------------------------------------------------------------------
def calculate_metrics(image: Image.Image) -> dict:
    bgr = pil_to_cv(image)
    h, w = bgr.shape[:2]
    target = 640
    scale = target / max(h, w)
    resized = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    region = safe_center_crop(resized, pad_ratio=0.08)

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    l_channel, a_channel, b_channel = to_lab_channels(region)

    brightness_val = float(np.mean(l_channel))

    l_norm = l_channel.astype(np.float32) / 255.0
    local_mean = cv2.GaussianBlur(l_norm, (0, 0), 3)
    local_var = cv2.GaussianBlur((l_norm - local_mean) ** 2, (0, 0), 3)
    dryness_metric = float(np.mean(local_var))

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    pores_metric = float(np.mean(np.abs(lap)))

    edges_fine = cv2.Canny(gray, 30, 90)
    lines_metric = float(np.sum(edges_fine > 0)) / edges_fine.size

    blur = cv2.GaussianBlur(gray, (0, 0), 1.8)
    edges_hard = cv2.Canny(blur, 80, 160)
    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(edges_hard, cv2.MORPH_CLOSE, kernel, iterations=1)
    wrinkles_metric = float(np.sum(edges_closed > 0)) / edges_closed.size

    a = a_channel.astype(np.float32)
    b = b_channel.astype(np.float32)
    var_ab = float(np.var(a) + np.var(b))

    def skew(x):
        x = x.flatten().astype(np.float32)
        m = np.mean(x)
        s = np.std(x) + 1e-6
        return float(np.mean(((x - m) / s) ** 3))

    skew_ab = float(np.clip(abs(skew(a)) + abs(skew(b)), 0, 2))
    pigmentation_metric = var_ab / (255.0 ** 2) + 0.2 * skew_ab

    return {
        "brightness": brightness_val,
        "dryness": dryness_metric,
        "texture-pores": pores_metric,
        "lines": lines_metric,
        "wrinkles": wrinkles_metric,
        "pigmentation": pigmentation_metric,
    }


# -----------------------------------------------------------------------------
# Clinical attenuation
# -----------------------------------------------------------------------------
def clinical_dampen(param: str, score_0_10: float) -> float:
    v = float(score_0_10)
    if param == "lines":
        if v >= 9.0: v = 6.5
        elif v >= 7.0: v = 5.5
        elif v >= 5.0: v = 4.5
    elif param == "pigmentation":
        if v >= 9.0: v = 6.0
        elif v >= 7.0: v = 5.0
        elif v >= 5.0: v = 4.0
    elif param == "texture-pores":
        if v >= 9.0: v = 5.5
        elif v >= 7.0: v = 4.5
        elif v >= 5.0: v = 3.5
    elif param == "brightness":
        v = max(v, 1.0)
    elif param == "dryness":
        v = max(v, 0.8)
    elif param == "wrinkles":
        if v < 1.0:
            v = min(1.0, v)
    return round(float(np.clip(v, 0.0, 10.0)), 1)


# -----------------------------------------------------------------------------
# Calibration application
# -----------------------------------------------------------------------------
def calibrate_scores_from_metrics(metrics: dict, calibration: dict) -> dict:
    out = {}
    for k, val in metrics.items():
        if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
            logger.warning("Invalid metric for %s: %r", k, val)
            out[k] = 0.0
            continue
        low = calibration.get(k, {}).get("low", DEFAULT_CALIB[k]["low"])
        high = calibration.get(k, {}).get("high", DEFAULT_CALIB[k]["high"])
        norm = normalize_to_10(float(val), float(low), float(high))
        out[k] = clinical_dampen(k, norm)
    return out
