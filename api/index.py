import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import pandas as pd
import joblib

# Tensorflow is imported conditionally to avoid crashing if it's too big for Vercel
import tensorflow as tf

# ==============================================================================
# 1. CARGA DE ARCHIVOS ENTRENADOS
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.keras"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
INFO_PATH = ARTIFACTS_DIR / "model_info.json"

_model = None
_preprocessor = None
_threshold = 0.5

def load_artifacts():
    global _model, _preprocessor, _threshold
    if _model is None and MODEL_PATH.exists():
        _model = tf.keras.models.load_model(str(MODEL_PATH))
    if _preprocessor is None and PREPROCESSOR_PATH.exists():
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
    if INFO_PATH.exists():
        info = json.loads(INFO_PATH.read_text())
        _threshold = info.get("threshold", 0.5)

# ==============================================================================
# 2. ESQUEMAS DE ENTRADA Y SALIDA (PYDANTIC)
# ==============================================================================
class PredictionRequest(BaseModel):
    # Numéricos
    Administrative: Optional[float] = 0
    Administrative_Duration: Optional[float] = 0
    Informational: Optional[float] = 0
    Informational_Duration: Optional[float] = 0
    ProductRelated: Optional[float] = 0
    ProductRelated_Duration: Optional[float] = 0
    BounceRates: Optional[float] = 0
    ExitRates: Optional[float] = 0
    PageValues: Optional[float] = 0
    SpecialDay: Optional[float] = 0
    # Categóricos
    Month: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: bool

class PredictionResponse(BaseModel):
    clasificacion: str
    probabilidad: float
    mensaje: str


# ==============================================================================
# 3. FASTAPI APP (VERCEL ENTRYPOINT)
# ==============================================================================
app = FastAPI(title="API Predictiva E-Commerce")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "API Activa. Usa el endpoint POST /predict para evaluar compras."}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest = Body(
    openapi_examples={
        "ejemplo_comprador": {
            "summary": "Cargar datos: Alta Intención (Compra)",
            "value": {
                "Administrative": 4,
                "Administrative_Duration": 120.5,
                "Informational": 1,
                "Informational_Duration": 45.0,
                "ProductRelated": 25,
                "ProductRelated_Duration": 1500.0,
                "BounceRates": 0.0,
                "ExitRates": 0.01,
                "PageValues": 55.5,
                "SpecialDay": 0.0,
                "Month": "Nov",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 2,
                "VisitorType": "Returning_Visitor",
                "Weekend": True
            }
        },
        "ejemplo_rebote": {
            "summary": "Cargar datos: Baja Intención (Rebote)",
            "value": {
                "Administrative": 0,
                "Administrative_Duration": 0.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 1,
                "ProductRelated_Duration": 0.0,
                "BounceRates": 0.2,
                "ExitRates": 0.2,
                "PageValues": 0.0,
                "SpecialDay": 0.0,
                "Month": "Feb",
                "OperatingSystems": 1,
                "Browser": 1,
                "Region": 1,
                "TrafficType": 1,
                "VisitorType": "Returning_Visitor",
                "Weekend": False
            }
        }
    }
)):
    load_artifacts()
    
    if _model is None or _preprocessor is None:
        raise HTTPException(status_code=500, detail="El modelo aún no está entrenado. Ejecuta train.py localmente primero.")

    # Convertimos los datos que envió el usuario a un DataFrame
    df = pd.DataFrame([request.model_dump()])

    # Importar y aplicar Feature Engineering para que coincida con el formato de entrenamiento
    from train import feature_engineering
    try:
        df = feature_engineering(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en feature engineering: {str(e)}")

    # Preprocesamos usando los imputers/scalers entrenados
    try:
        X = _preprocessor.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando datos: {str(e)}")

    # Predecimos la probabilidad
    prob = float(_model.predict(X, verbose=0)[0][0])
    
    # Decidimos basándonos en el threshold óptimo encontrado en el entrenamiento
    will_buy = prob >= _threshold
    
    # Generamos los campos requeridos por la asignación
    clasificacion = "compra" if will_buy else "no compra"
    
    if will_buy:
        mensaje = f"El usuario presenta un {prob*100:.1f}% de probabilidades de hacer la compra, altamente probable."
    else:
        mensaje = f"El usuario presenta solo un {prob*100:.1f}% de probabilidades de hacer la compra, poco probable."

    return {
        "clasificacion": clasificacion,
        "probabilidad": round(prob, 4),
        "mensaje": mensaje
    }
