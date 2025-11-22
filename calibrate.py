import os
import json
import numpy as np
import cv2
from PIL import Image

# -----------------------------
# Utilidades de imagen
# -----------------------------
def pil_to_cv(image: Image.Image) -> np.ndarray:
    # Convierte PIL (RGB) a OpenCV (BGR)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def safe_center_crop(img: np.ndarray, pad_ratio: float = 0.08) -> np.ndarray:
    """Recorte centrado con padding relativo, evitando errores en caras cerca del borde."""
    h, w = img.shape[:2]
    pad = int(pad_ratio * max(h, w))
    y1, y2 = max(0, pad), max(1, h - pad)
    x1, x2 = max(0, pad), max(1, w - pad)
    if y2 - y1 < 16 or x2 - x1 < 16:  # evita recortes inválidos
        return img
    return img[y1:y2, x1:x2]

def to_lab_channels(bgr: np.ndarray):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    return l, a, b

# -----------------------------
# Carga segura de calibración
# -----------------------------
DEFAULT_CALIB = {
    "brightness": {"low": 20.0, "high": 230.0},     # L en Lab
    "dryness": {"low": 0.0005, "high": 0.02},       # var local de L
    "texture-pores": {"low": 0.2, "high": 4.5},     # media |Laplacian|
    "lines": {"low": 0.005, "high": 0.12},          # densidad de bordes finos
    "wrinkles": {"low": 0.003, "high": 0.10},       # densidad de bordes cerrados
    "pigmentation": {"low": 0.02, "high": 0.45}     # var(a)+var(b)/255^2 + 0.2*skew
}

def load_calibration(path: str = "calibration.json") -> dict:
    """Carga calibration.json; si no existe o está corrupto, usa DEFAULT_CALIB."""
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            # valida claves mínimas; mezcla con defaults si faltan
            calib = DEFAULT_CALIB.copy()
            for k, v in data.items():
                if isinstance(v, dict) and "low" in v and "high" in v:
                    calib[k] = {"low": float(v["low"]), "high": float(v["high"])}
            return calib
    except Exception as e:
        print(f"⚠️ No se pudo cargar calibración: {e}")
    return DEFAULT_CALIB

# -----------------------------
# Normalización 0–10
# -----------------------------
def normalize_to_10(value: float, low: float, high: float) -> float:
    """Mapea [low, high] a [0,10] con clipping."""
    if high == low:
        return 0.0
    scaled = (value - low) / (high - low)
    return float(np.clip(scaled * 10.0, 0.0, 10.0))

# -----------------------------
# Métricas base
# -----------------------------
def calculate_metrics(image: Image.Image) -> dict:
    """Calcula métricas físicas pre-modelo con filtros para estabilidad."""
    # Resize uniforme manteniendo proporciones
    bgr = pil_to_cv(image)
    h, w = bgr.shape[:2]
    target = 640
    scale = target / max(h, w)
    resized = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Recorte centrado seguro
    region = safe_center_crop(resized, pad_ratio=0.08)

    # Canales
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    l_channel, a_channel, b_channel = to_lab_channels(region)

    # Brightness (promedio de L)
    brightness_val = float(np.mean(l_channel))

    # Dryness (var local de L, suavizado gaussiano)
    l_norm = l_channel.astype(np.float32) / 255.0
    local_mean = cv2.GaussianBlur(l_norm, (0, 0), 3)
    local_var = cv2.GaussianBlur((l_norm - local_mean) ** 2, (0, 0), 3)
    dryness_metric = float(np.mean(local_var))

    # Texture-pores (respuesta Laplaciana media)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    pores_metric = float(np.mean(np.abs(lap)))

    # Lines (bordes finos; Canny suave)
    edges_fine = cv2.Canny(gray, 30, 90)
    lines_metric = float(np.sum(edges_fine > 0)) / edges_fine.size

    # Wrinkles (bordes tras blur y cierre morfológico)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.8)
    edges_hard = cv2.Canny(blur, 80, 160)
    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(edges_hard, cv2.MORPH_CLOSE, kernel, iterations=1)
    wrinkles_metric = float(np.sum(edges_closed > 0)) / edges_closed.size

    # Pigmentation (varianza de a/b + asimetría)
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

    return {
        "brightness": brightness_val,
        "dryness": dryness_metric,
        "texture-pores": pores_metric,
        "lines": lines_metric,
        "wrinkles": wrinkles_metric,
        "pigmentation": pigmentation_metric
    }

# -----------------------------
# Atenuación clínica y robustez
# -----------------------------
def clinical_dampen(param: str, score_0_10: float) -> float:
    """Suaviza exageraciones según parámetro en escala 0–10."""
    v = float(score_0_10)

    if param == "lines":
        # Reduce extremos; que 10 solo aparezca en pliegues evidentes
        if v >= 9.0:
            v = 6.5
        elif v >= 7.0:
            v = 5.5
        elif v >= 5.0:
            v = 4.5

    elif param == "pigmentation":
        # Evita confundir pecas con melasma
        if v >= 9.0:
            v = 6.0
        elif v >= 7.0:
            v = 5.0
        elif v >= 5.0:
            v = 4.0

    elif param == "texture-pores":
        # Penaliza menos la textura natural
        if v >= 9.0:
            v = 5.5
        elif v >= 7.0:
            v = 4.5
        elif v >= 5.0:
            v = 3.5

    elif param == "brightness":
        # Suelo de ruido para evitar 0–1 por iluminación puntual
        v = max(v, 1.0)

    elif param == "dryness":
        # Suelo de ruido, la piel rara vez es 0
        v = max(v, 0.8)

    elif param == "wrinkles":
        # Si es <1, clipea a 0–1 con suavizado
        if v < 1.0:
            v = min(1.0, v)

    return round(float(np.clip(v, 0.0, 10.0)), 1)

# -----------------------------
# Pipeline de calibración
# -----------------------------
def calibrate_scores_from_metrics(metrics: dict, calibration: dict) -> dict:
    """Normaliza métricas a 0–10 usando calibration.json y aplica atenuación clínica."""
    out = {}
    for k, val in metrics.items():
        low = calibration.get(k, {}).get("low", DEFAULT_CALIB[k]["low"])
        high = calibration.get(k, {}).get("high", DEFAULT_CALIB[k]["high"])
        norm = normalize_to_10(float(val), float(low), float(high))
        out[k] = clinical_dampen(k, norm)
    return out

# -----------------------------
# API pública del módulo
# -----------------------------
def analyze_and_calibrate(image: Image.Image, calibration_path: str = "calibration.json") -> dict:
    """
    Calcula métricas físicas y devuelve scores calibrados (0–10).
    Seguro para despliegue en Render (no depende de rutas absolutas).
    """
    calib = load_calibration(calibration_path)
    metrics = calculate_metrics(image)
    scores = calibrate_scores_from_metrics(metrics, calib)
    # Diagnóstico simple basado en máximos (puedes refinarlo en tu backend)
    max_param = max(scores, key=lambda k: scores[k])
    diagnosis = {
        "brightness": "Brillo destacado",
        "dryness": "Sequedad destacada",
        "texture-pores": "Textura con poros marcados",
        "lines": "Líneas visibles",
        "wrinkles": "Arrugas visibles",
        "pigmentation": "Pigmentación destacada"
    }.get(max_param, "Perfil cutáneo equilibrado")

    return {
        "scores": scores,
        "diagnosis": diagnosis
    }

# -----------------------------
# CLI opcional para pruebas
# -----------------------------
if __name__ == "__main__":
    # Uso: python calibrate.py (no requiere args; solo valida importaciones)
    print("✅ Módulo de calibración listo.")
