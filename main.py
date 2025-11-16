from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Leer el contenido del archivo
        contents = await file.read()

        # Abrir la imagen usando Pillow
        image = Image.open(io.BytesIO(contents))

        # Simulación de análisis (puedes reemplazar esto con tu lógica real)
        width, height = image.size
        format = image.format

        # Resultado simulado
        result = {
            "status": "ok",
            "format": format,
            "dimensions": f"{width}x{height}",
            "message": "Análisis simulado exitoso"
        }

        return result

    except Exception
