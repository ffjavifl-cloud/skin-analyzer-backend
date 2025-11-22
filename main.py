from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import io

from model import predict_scores

app = FastAPI(title="Skin Analyzer Training API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir a tu dominio
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

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
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

        return {
            "diagnosis": diagnosis,
            "scores": scores
        }

    except Exception as e:
        print(f"❌ Error en análisis: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "No se pudo procesar la imagen",
                "details": str(e)
            }
        )
