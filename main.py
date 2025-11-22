from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import io

from model import predict_scores

app = FastAPI(title="Skin Analyzer API")

# Mantengo CORS abierto como en tu versión actual para no romper nada.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir a https://ifl-cloud.github.io cuando quieras
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
def status():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Skin Analyzer API activa"}

def clamp(x, low, high):
    return max(low, min(high, x))

def expected_scores_heuristic(sex: str | None, age: int | None):
    # Heurística inicial sin dataset: ajusta expectativas por edad/sexo en escala 0–10
    # Más edad → mayor severidad esperada en arrugas/líneas/pigmentación/sequedad.
    # brightness baja ligeramente con la edad.
    # Ajustes suaves por sexo (diferencias promedio).
    if age is None:
        age = 35
    sex = (sex or "unknown").lower()

    # Normalización de edad entre 20 y 70
    age_min, age_max = 20, 70
    t = (age - age_min) / (age_max - age_min)
    t = clamp(t, 0.0, 1.0)

    sex_adj = {
        "male": {"wrinkles": 0.4, "lines": 0.3, "texture-pores": 0.3, "pigmentation": 0.2, "dryness": 0.2, "brightness": -0.2},
        "female": {"wrinkles": -0.2, "lines": -0.1, "texture-pores": -0.1, "pigmentation": 0.1, "dryness": 0.0, "brightness": 0.1},
        "unknown": {"wrinkles": 0.0, "lines": 0.0, "texture-pores": 0.0, "pigmentation": 0.0, "dryness": 0.0, "brightness": 0.0},
    }
    adj = sex_adj.get(sex, sex_adj["unknown"])

    exp = {
        "wrinkles": clamp(2.0 + 6.0 * t + adj["wrinkles"], 0, 10),
        "lines": clamp(3.0 + 5.0 * t + adj["lines"], 0, 10),
        "texture-pores": clamp(4.0 + 3.0 * t + adj["texture-pores"], 0, 10),
        "pigmentation": clamp(3.0 + 4.0 * t + adj["pigmentation"], 0, 10),
        "dryness": clamp(2.0 + 4.0 * t + adj["dryness"], 0, 10),
        "brightness": clamp(7.0 - 2.0 * t + adj["brightness"], 0, 10),
    }
    return exp

def estimate_skin_age(scores: dict, sex: str | None, age: int | None):
    if age is None:
        age = 35
    exp = expected_scores_heuristic(sex, age)

    # Diferencia total (observado - esperado)
    observed_sum = sum(scores.values())
    expected_sum = sum(exp.values())
    delta = observed_sum - expected_sum

    # Factor: 1 punto de diferencia total ≈ 1.8 años (ajuste suave)
    factor_years = 1.8
    est = age + delta * factor_years

    # Limitar a rango clínicamente razonable
    est = clamp(est, 15, 85)
    return round(est)

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    age: int | None = Form(default=None),
    sex: str | None = Form(default=None),
):
    try:
        raw = await file.read()
        print(f"📥 Imagen recibida: {file.filename}, tamaño: {len(raw)} bytes")

        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            print(f"🖼️ Imagen convertida a RGB: {image.size}")
        except UnidentifiedImageError:
            raise ValueError("La imagen no se pudo abrir. Verifica el formato.")

        scores = predict_scores(image)
        print(f"✅ Scores generados: {scores}")

        if not isinstance(scores, dict) or not scores:
            raise ValueError("No se generaron scores válidos.")

        top_param = max(scores, key=lambda k: scores[k])
        diagnosis_map = {
            "dryness": "Signos de sequedad prominentes.",
            "pigmentation": "Pigmentación destacada.",
            "wrinkles": "Arrugas marcadas.",
            "lines": "Líneas visibles.",
            "texture-pores": "Textura/poros acentuados.",
            "brightness": "Brillo bajo (posible iluminación subóptima)."
        }
        diagnosis = diagnosis_map.get(top_param, "Evaluación clínica general.")

        # Estimación de edad de piel (heurística inicial)
        skin_age = estimate_skin_age(scores, sex, age)

        return JSONResponse(
            status_code=200,
            content={
                "diagnosis": diagnosis,
                "scores": scores,
                "skin_age": skin_age,
                "meta": {"age": age, "sex": sex},
            }
        )

    except Exception as e:
        print(f"❌ Error en análisis: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "No se pudo procesar la imagen",
                "details": str(e)
            }
        )
