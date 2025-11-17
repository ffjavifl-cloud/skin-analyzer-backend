import io
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageStat, ImageOps

app = FastAPI()

# ✅ CORS corregido: permite tu frontend real en GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ffjavifl-cloud.github.io"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "Accept"],
)

@app.get("/")
def root():
    return {"status": "ok"}

def safe_score(value, divisor):
    try:
        score = value / divisor
        return min(10, max(0, int(score)))
    except Exception:
        return 0

def resize_max(image: Image.Image, max_side: int = 768) -> Image.Image:
    w, h = image.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        return image.resize(new_size, Image.LANCZOS)
    return image

def analyze_skin_features(image: Image.Image):
    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    wrinkles_score = safe_score(np.mean(np.abs(laplacian)), 5)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    fine_lines_score = safe_score(np.mean(np.abs(sobelx)) + np.mean(np.abs(sobely)), 10)

    dark_pixels = np.sum(gray < 50)
    pigmentation_score = safe_score(dark_pixels, gray.size / 20)

    stat = ImageStat.Stat(image.convert("L"))
    dryness_score = safe_score((130 - stat.mean[0]) + (50 - stat.stddev[0]) / 2, 1)

    bright_pixels = np.sum(np.max(img_cv, axis=2) > 240)
    brightness_score = safe_score(bright_pixels, gray.size / 20)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    texture = cv2.subtract(gray, blurred)
    pores_score = safe_score(np.mean(np.abs(texture)), 5)

    return {
        "wrinkles_deep": wrinkles_score,
        "lines_fine": fine_lines_score,
        "pigmentation": pigmentation_score,
        "dryness": dryness_score,
        "brightness_excess": brightness_score,
        "pores_visible": pores_score
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            return JSONResponse(status_code=400, content={"error": "El archivo subido no es una imagen válida."})

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image = resize_max(image, max_side=768)

        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        brightness = stat.mean[0]
        contrast = stat.stddev[0]

        color_stat = ImageStat.Stat(image)
        r_mean, g_mean, b_mean = color_stat.mean
        redness_index = r_mean - (g_mean + b_mean) / 2

        if brightness < 60:
            diagnosis = "Tendencia a hiperpigmentación u oscuridad; revisar exposición y hidratación."
        elif brightness > 190:
            diagnosis = "Piel clara; vigilar sensibilidad y fotoprotección."
        else:
            diagnosis = "Tono equilibrado, sin anomalías evidentes."

        if redness_index > 10:
            diagnosis += " Posible eritema/rojeces visibles."
        elif redness_index < -10:
            diagnosis += " Tono verdoso/azulado; revisar balance de color y luz."

        avg_rgb = (r_mean + g_mean + b_mean) / 3
        if avg_rgb > 230:
            fitzpatrick = "Tipo I - Muy clara"
        elif avg_rgb > 200:
            fitzpatrick = "Tipo II - Clara"
        elif avg_rgb > 170:
            fitzpatrick = "Tipo III - Intermedia"
        elif avg_rgb > 130:
            fitzpatrick = "Tipo IV - Oliva"
        elif avg_rgb > 90:
            fitzpatrick = "Tipo V - Morena"
        else:
            fitzpatrick = "Tipo VI - Muy oscura"

        diagnosis += f" | Tono estimado: {fitzpatrick}"

        scores = analyze_skin_features(image)

        if scores["wrinkles_deep"] > 6:
            diagnosis += " Presencia de arrugas profundas."
        if scores["lines_fine"] > 6:
            diagnosis += " Líneas finas visibles."
        if scores["pigmentation"] > 6:
            diagnosis += " Pigmentación marcada."
        if scores["dryness"] > 6:
            diagnosis += " Sequedad visible."
        if scores["brightness_excess"] > 6:
            diagnosis += " Brillo excesivo en zonas específicas."
        if scores["pores_visible"] > 6:
            diagnosis += " Poros visibles o textura irregular."

        return {
            "result": "Análisis completado",
            "metrics": {
                "brightness": round(brightness, 2),
                "contrast": round(contrast, 2),
                "avg_color": {
                    "r": round(r_mean, 1),
                    "g": round(g_mean, 1),
                    "b": round(b_mean, 1)
                },
                "redness_index": round(redness_index, 2)
            },
            "scores": scores,
            "diagnosis": diagnosis
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error procesando la imagen: {str(e)}"})
