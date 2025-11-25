import os
import numpy as np
from PIL import Image
import pillow_avif  # Activar soporte para imágenes .avif si aún quedan
import joblib

# Parámetros clínicos
PARAMETERS = ["brightness", "dryness", "wrinkles", "lines", "texture-pores", "pigmentation"]

# Cargar modelos entrenados
models = joblib.load("model.pkl")

# Emojis por severidad
EMOJIS = {
    "Mild": "🟢",
    "Moderate": "🟠",
    "Severe": "🔴"
}

# Clasificación clínica universal con rango intermedio
def classify_severity(score: float) -> str:
    if score < 4.5:
        return "Mild"
    elif score < 6.5:
        return "Moderate"
    else:
        return "Severe"

def preprocess_image(image_path):
    """Convierte la imagen a escala de grises, redimensiona y normaliza"""
    img = Image.open(image_path).convert("L").resize((64, 64))
    arr = np.array(img).flatten() / 255.0
    return arr.reshape(1, -1)

def predict_scores(image_path):
    """Devuelve los scores clínicos por parámetro"""
    input_data = preprocess_image(image_path)
    results = {}
    for param in PARAMETERS:
        if param in models:
            score = models[param].predict(input_data)[0]
            results[param] = round(score, 2)
        else:
            results[param] = "Modelo no disponible"
    return results

if __name__ == "__main__":
    # Ruta de la imagen a evaluar
    image_path = "test_image.jpg"  # Cambia esto si usas otra imagen
    if not os.path.exists(image_path):
        print(f"⚠️ Imagen no encontrada: {image_path}")
    else:
        scores = predict_scores(image_path)
        print("📊 Resultados clínicos:")
        for param, score in scores.items():
            if isinstance(score, (int, float)):
                estado = classify_severity(score)
                emoji = EMOJIS[estado]
                print(f"🔹 {param}: {score} → {estado} {emoji}")
            else:
                print(f"🔹 {param}: {score}")
