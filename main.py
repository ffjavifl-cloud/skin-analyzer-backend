"""Skin Analyzer API.

Live scoring path runs through calibrate.py (calibration.json + clinical
attenuation). model.py and predict.py have been retired.
"""
import io
import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError

from calibrate import load_calibration, calculate_metrics, calibrate_scores_from_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skin-analyzer")

# --- Limits ------------------------------------------------------------------
MAX_UPLOAD_BYTES = 10 * 1024 * 1024          # reject uploads over 10 MB
Image.MAX_IMAGE_PIXELS = 40_000_000          # decompression-bomb guard (~40 MP)
MAX_DIM = 1600                               # downscale ceiling before analysis

# --- CORS: pin to known origins, no credentials ------------------------------
ALLOWED_ORIGINS = [
    "https://ffjavifl-cloud.github.io",
    "http://localhost:5173",
    "http://localhost:3000",
]

EMOJIS = {"Mild": "🟢", "Moderate": "🟠", "Severe": "🔴"}
DIAGNOSIS = {
    "brightness": "Brillo destacado",
    "dryness": "Sequedad destacada",
    "texture-pores": "Textura con poros marcados",
    "lines": "Líneas visibles",
    "wrinkles": "Arrugas visibles",
    "pigmentation": "Pigmentación destacada",
}

app = FastAPI(title="Skin Analyzer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Load calibration once at startup (avoids per-request file IO)
CALIBRATION = load_calibration("calibration.json")


def classify_severity(score: float) -> str:
    if score < 4.5:
        return "Mild"
    elif score < 6.5:
        return "Moderate"
    return "Severe"


def interpret_by_age(param: str, score: float, edad: int, sex: Optional[str]) -> str:
    if edad < 30:
        if param in ("wrinkles", "lines") and score > 6.5:
            return "Signos prematuros para edad joven."
        if param == "brightness" and score < 3.0:
            return "Brillo bajo para piel joven."
        if param == "dryness" and score > 6.5:
            return "Sequedad inusual en piel joven."
    elif edad >= 60:
        if param in ("wrinkles", "lines") and score < 4.0:
            return "Piel notablemente conservada para edad avanzada."
        if param == "brightness" and score > 7.0:
            return "Brillo elevado para edad madura."
    if sex == "male" and param == "texture-pores" and score > 6.0:
        return "Poros algo más visibles, habitual en piel masculina."
    return "Interpretación acorde a edad."


def _analyze(image: Image.Image, edad: int, sex: Optional[str]) -> dict:
    """CPU-bound work; runs in a threadpool, off the event loop."""
    scores = calibrate_scores_from_metrics(calculate_metrics(image), CALIBRATION)
    classified = {}
    for param, score in scores.items():
        sev = classify_severity(score)
        classified[param] = {
            "score": round(float(score), 2),
            "severity": sev,
            "emoji": EMOJIS[sev],
            "age_context": interpret_by_age(param, score, edad, sex),
        }
    top = max(scores, key=lambda k: scores[k])
    return {"diagnosis": DIAGNOSIS.get(top, "Perfil cutáneo equilibrado"),
            "results": classified}


@app.get("/")
@app.get("/status")
def status():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    edad: int = Form(40),
    sex: Optional[str] = Form(None),
):
    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "Archivo vacío."}, status_code=400)
    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            {"error": "La imagen supera el tamaño máximo de 10 MB."}, status_code=413
        )

    try:
        image = Image.open(io.BytesIO(raw))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return JSONResponse(
            {"error": "El archivo no es una imagen válida."}, status_code=400
        )

    if max(image.size) > MAX_DIM:
        image.thumbnail((MAX_DIM, MAX_DIM))

    try:
        result = await run_in_threadpool(_analyze, image, edad, sex)
    except Exception:
        logger.exception("Image analysis failed")
        return JSONResponse(
            {"error": "No se pudo procesar la imagen."}, status_code=500
        )

    return JSONResponse(result)
