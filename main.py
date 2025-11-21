from fastapi import FastAPI, UploadFile, File
import uvicorn
import json

app = FastAPI()

# Cargar calibración
with open("calibration.json", "r") as f:
    calibration = json.load(f)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    report = {}
    for param, values in calibration.items():
        score = (values.get("severe", 0) * 10) / (values.get("mild", 1) + 1)
        report[param] = round(score, 1)

    return {
        "scores": report,
        "diagnosis": "Informe clínico generado con calibración"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
