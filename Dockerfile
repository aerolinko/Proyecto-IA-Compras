FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo lo que hay en la raíz (incluyendo la carpeta src/)
COPY . .

EXPOSE 8000

# Le dices a uvicorn que busque dentro de la carpeta "src", en el archivo "server"
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]