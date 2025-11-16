import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image, ImageStat

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir esto a tu dominio más adelante
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Leer la imagen subida
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Métricas en escala de grises
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        brightness = stat.mean[0]          # 0-255 (media de luminancia)
        contrast = stat.stddev[0]          # desviación estándar ≈ contraste

        # Color promedio (pistas de tono y rojez)
        color_stat = ImageStat.Stat(image)
        r_mean, g_mean, b_mean = color_stat.mean

        # Índice de rojez: diferencia del canal R frente a G y B
        redness_index = r_mean - (g_mean + b_mean) / 2

        # Reglas clínicas simuladas (ajustables a tu criterio)
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
            "diagnosis": diagnosis
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
