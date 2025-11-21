import os
import numpy as np
from PIL import Image
import json

def calculate_average(folder_path):
    values = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('L')  # escala de grises
            array = np.array(image) / 255.0  # normalizar
            values.append(np.mean(array))
    return round(float(np.mean(values)), 4) if values else 0.0

def generate_calibration(data_dir='data'):
    calibration = {}
    for parameter in os.listdir(data_dir):
        param_path = os.path.join(data_dir, parameter)
        if os.path.isdir(param_path):
            mild_path = os.path.join(param_path, 'mild')
            severe_path = os.path.join(param_path, 'severe')
            calibration[parameter] = {
                'mild': calculate_average(mild_path),
                'severe': calculate_average(severe_path)
            }
    with open('calibration.json', 'w') as f:
        json.dump(calibration, f, indent=4)
    print("✅ calibration.json generado con éxito.")

if __name__ == "__main__":
    generate_calibration()
