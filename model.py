import os
from PIL import Image
import joblib  # opcional si quieres verificar el modelo

# Emojis por severidad (mismo criterio universal 4.5–6.5)
EMOJIS = {
    "Mild": "🟢",
    "Moderate": "🟠",
    "Severe": "🔴"
}

def classify_severity(score: float) -> str:
    if score < 4.5:
        return "Mild"
    elif score < 6.5:
        return "Moderate"
    else:
        return "Severe"

if __name__ == "__main__":
    # Ruta de la imagen a evaluar (misma que usas en Swagger)
    image_path = "test_image.jpg"  # cambia si usas otra imagen

    if not os.path.exists(image_path):
        print(f"⚠️ Imagen no encontrada: {image_path}")
    else:
        # Abrir como PIL y pasarla directa a predict_scores
        image = Image.open(image_path).convert("RGB")
        scores = predict_scores(image)

        print("📊 Resultados clínicos:")
        for param, score in scores.items():
            estado = classify_severity(score)
            emoji = EMOJIS[estado]
            print(f"🔹 {param}: {score} → {estado} {emoji}")
