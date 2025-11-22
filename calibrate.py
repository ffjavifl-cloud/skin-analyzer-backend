import os
import numpy as np
import cv2
from PIL import Image
import json

def pil_to_cv(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def normalize(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    scaled = (value - low) / (high - low)
    return float(np.clip(scaled * 10.0, 0.0, 10.0))

def calculate_metrics(image: Image.Image) -> dict:
    img = pil_to_cv(image)
    h, w = img.shape[:2]
    target = 640
    scale = target / max(h, w)
    img_resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    rh, rw = img_resized.shape[:2]
    pad = int(0.08 * max(rh, rw))
    r = img_resized[pad:rh - pad, pad:rw - pad]

    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(r, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Brightness
    brightness_val = float(np.mean(l_channel))

    # Dryness
    l_norm = l_channel.astype(np.float32) / 255.0
    local_mean = cv2.GaussianBlur(l_norm, (0, 0), 3)
    local_var = cv2.GaussianBlur((l_norm - local_mean) ** 2, (0, 0), 3)
    dryness_metric = float(np.mean(local_var))

    # Texture-pores
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    pores_metric = float(np.mean(np.abs(lap)))

    # Lines
    edges_fine = cv2.Canny(gray, 30, 90)
    lines_metric = float(np.sum(edges_fine > 0)) / edges_fine.size

    # Wrinkles
    blur = cv2.GaussianBlur(gray, (0, 0), 1.8)
    edges_hard = cv2.Canny(blur, 80, 160)
    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(edges_hard, cv2.MORPH_CLOSE, kernel, iterations=1)
    wrinkles_metric = float(np.sum(edges_closed > 0)) / edges_closed.size

    # Pigmentation
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

def calculate_average(folder_path):
    values = {p: [] for p in ["brightness","dryness","texture-pores","lines","wrinkles","pigmentation"]}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            metrics = calculate_metrics(image)
            for k,v in metrics.items():
                values[k].append(v)
    return {k: (np.mean(v) if v else None) for k,v in values.items()}

def generate_calibration(data_dir='data'):
    calibration = {}
    for parameter in ["brightness","dryness","texture-pores","lines","wrinkles","pigmentation"]:
        mild_path = os.path.join(data_dir, parameter, 'mild')
        severe_path = os.path.join(data_dir, parameter, 'severe')

        mild_vals = calculate_average(mild_path) if os.path.exists(mild_path) else {}
        severe_vals = calculate_average(severe_path) if os.path.exists(severe_path) else {}

        low = None
        high = None
        if mild_vals.get(parameter) is not None and severe_vals.get(parameter) is not None:
            low = min(mild_vals[parameter], severe_vals[parameter])
            high = max(mild_vals[parameter], severe_vals[parameter])
        elif mild_vals.get(parameter) is not None:
            low, high = mild_vals[parameter]*0.8, mild_vals[parameter]*1.2
        elif severe_vals.get(parameter) is not None:
            low, high = severe_vals[parameter]*0.8, severe_vals[parameter]*1.2
        else:
            low, high = 0.0, 10.0  # rango seguro por defecto

        calibration[parameter] = {"low": round(float(low),4), "high": round(float(high),4)}

    with open('calibration.json', 'w') as f:
        json.dump(calibration, f, indent=4)
    print("✅ calibration.json generado con éxito.")

if __name__ == "__main__":
    generate_calibration()
