import io
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageStat

app = FastAPI()

# Habilitar CORS para que el frontend en GitHub Pages pueda acceder
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

def analyze_skin_features(image: Image.Image):
    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # Arrugas profundas
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    wrinkles_score = min(10, int(np.mean(np.abs(laplacian)) / 5))

    # Líneas finas
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    fine_lines_score = min(10, int((np.mean(np.abs(sobelx)) + np.mean(np.abs(sobely))) / 10))

    # Pigmentación
    dark_pixels = np.sum(gray < 50)
    pigmentation_score = min(10, int(dark_pixels / (gray.size / 20)))

    # Sequedad
    stat = ImageStat.Stat(image.convert("L"))
    dryness_score = min(10, int((130 - stat.mean[0]) + (50 - stat.stddev[0]) / 2))

    # Brillo excesivo
    bright_pixels = np.sum(np.max(img_cv, axis=2) > 240)
    brightness_score = min(10, int(bright_pixels / (gray.size / 20)))

    # Poros visibles
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    texture = cv2.subtract(gray, blurred)
    pores_score = min(10, int(np.mean(np.abs(texture)) / 5))

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
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Métricas básicas
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        brightness = stat.mean[0]
        contrast = stat.stddev[0]

        color_stat = ImageStat.Stat(image)
        r_mean, g_mean, b_mean = color_stat.mean
        redness_index = r_mean - (g_mean + b_mean) / 2

        # Diagnóstico básico
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

        # Fitzpatrick
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

        # Análisis clínico avanzado
        scores = analyze_skin_features(image)

        # Diagnóstico complementario (puedes afinarlo tú)
        if scores["wrinkles_deep"] > 6:
            diagnosis += " Presencia de arrugas profundas."
        if scores["pigmentation"] > 6:
            diagnosis += " Pigmentación marcada."
        if scores["dryness"] > 6:
            diagnosis += " Sequedad visible."
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
        return JSONResponse(status_code=500, content={"error": str(e)})
