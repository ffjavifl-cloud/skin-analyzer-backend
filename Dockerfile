FROM python:3.9

# Instala dependencias del sistema necesarias para Pillow y otras librerías
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Define el directorio de trabajo
WORKDIR /app

# Instala pip y Torch/Torchvision versión CPU, luego el resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt
COPY requirements.txt .

# Define el puerto y el comando de inicio
ENV PORT=10000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
