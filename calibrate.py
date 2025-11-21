import os
import json

base_path = "data"
parameters = os.listdir(base_path)
calibration = {}

for param in parameters:
    param_path = os.path.join(base_path, param)
    if not os.path.isdir(param_path):
        continue

    mild_path = os.path.join(param_path, "mild")
    severe_path = os.path.join(param_path, "severe")

    mild_count = len(os.listdir(mild_path)) if os.path.exists(mild_path) else 0
    severe_count = len(os.listdir(severe_path)) if os.path.exists(severe_path) else 0

    total = mild_count + severe_count
    if total == 0:
        continue

    calibration[param] = {
        "mild": round(mild_count / total, 2),
        "severe": round(severe_count / total, 2)
    }

with open("calibration.json", "w") as f:
    json.dump(calibration, f, indent=4)

print("✅ Archivo calibration.json generado correctamente")
