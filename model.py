from PIL import Image
from calibrate import analyze_and_calibrate

def predict_scores(image: Image.Image) -> dict:
    """
    Recibe una imagen PIL y devuelve los scores clínicos calibrados.
    Utiliza calibrate.py para aplicar lógica dermatológica y evitar exageraciones.
    """
    try:
        print("🔍 Ejecutando análisis clínico...")
        result = analyze_and_calibrate(image, calibration_path="calibration.json")
        print("✅ Resultado:", result)

        scores = result.get("scores", {})
        if not isinstance(scores, dict) or not scores:
            raise ValueError("No se generaron scores válidos.")

        for k, v in scores.items():
            if not isinstance(v, (int, float)):
                raise ValueError(f"Score inválido para {k}: {v}")

        return scores

    except Exception as e:
        print(f"❌ Error en calibración: {e}")
        return {
            "brightness": 0.0,
            "dryness": 0.0,
            "lines": 0.0,
            "pigmentation": 0.0,
            "texture-pores": 0.0,
            "wrinkles": 0.0
        }
