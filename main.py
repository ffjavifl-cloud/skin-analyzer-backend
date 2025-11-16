from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Simulación de análisis
    return JSONResponse(content={"result": "Análisis simulado exitoso"})
