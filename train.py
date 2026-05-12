import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# ---- LIBRERÍAS DE MACHINE LEARNING ----
# Scikit-Learn para el Pipeline de Preprocesamiento
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Keras y Hyperband para la Red Neuronal
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import joblib

# ==============================================================================
# 1. CONFIGURACIÓN DEL PROYECTO
# ==============================================================================
# Aquí puedes editar el nombre de tu archivo CSV cuando lo tengas listo.
CSV_PATH = "data/dataset_shop.csv"

# Definimos exactamente qué columnas vamos a usar DESPUÉS del Feature Engineering
NUMERIC_FEATURES = [
    "Administrative", "Avg_Admin_Duration", "Informational", "Avg_Info_Duration",
    "ProductRelated", "Avg_Product_Duration", "ExitRates", 
    "PageValues", "SpecialDay"
]
CATEGORICAL_FEATURES = [
    "Month", "OperatingSystems", "Browser", "Region", "TrafficType", 
    "VisitorType", "Weekend"
]
TARGET = "Revenue"

# Carpetas donde se guardarán los resultados
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.keras"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
MODEL_INFO_PATH = ARTIFACTS_DIR / "model_info.json"


# ==============================================================================
# 2. FEATURE ENGINEERING (Ingeniería de Características)
# ==============================================================================
def feature_engineering(df):
    """
    Mejora la calidad de los datos eliminando colinealidad perfecta y calculando 
    métricas derivadas más estables (promedios en lugar de totales).
    No agrupa variables categóricas.
    """
    df = df.copy()
    
    # 1. Eliminar BounceRates (Altamente colineal con ExitRates, casi redundancia perfecta)
    if 'BounceRates' in df.columns:
        df = df.drop(columns=['BounceRates'])
        
    # 2. Convertir Duraciones Totales a Duraciones Promedio por Página
    # Usamos + 1e-5 para evitar errores matemáticos de división por cero
    if 'Administrative' in df.columns and 'Administrative_Duration' in df.columns:
        df['Avg_Admin_Duration'] = df['Administrative_Duration'] / (df['Administrative'] + 1e-5)
        df = df.drop(columns=['Administrative_Duration'])
        
    if 'Informational' in df.columns and 'Informational_Duration' in df.columns:
        df['Avg_Info_Duration'] = df['Informational_Duration'] / (df['Informational'] + 1e-5)
        df = df.drop(columns=['Informational_Duration'])
        
    if 'ProductRelated' in df.columns and 'ProductRelated_Duration' in df.columns:
        df['Avg_Product_Duration'] = df['ProductRelated_Duration'] / (df['ProductRelated'] + 1e-5)
        df = df.drop(columns=['ProductRelated_Duration'])
        
    return df

# ==============================================================================
# 3. PREPROCESADOR DE DATOS (SCIKIT-LEARN)
# ==============================================================================
def build_preprocessor():
    """
    Crea el Pipeline que se encarga de rellenar valores nulos (imputers) y 
    escalar/codificar los datos numéricos y categóricos.
    """
    # Para los números: Si falta algo ponemos la mediana, y luego escalamos.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Para texto/categorías: Si falta algo ponemos lo más común, y luego hacemos One-Hot Encoding.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Unimos ambos en un solo preprocesador
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder="drop")
    
    return preprocessor


# ==============================================================================
# 4. ARQUITECTURA DEL MODELO (KERAS)
# ==============================================================================
def build_model(hp, input_dim):
    """
    Esta función es utilizada por Hyperband para probar diferentes combinaciones
    de capas, neuronas, funciones de activación y tasas de aprendizaje.
    """
    # Hyperband decide cuántas capas intermedias adicionales poner
    n_layers = hp.Int("n_layers", min_value=1, max_value=4, step=1)
    activation = hp.Choice("activation", ["relu"])
    lr = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    model = keras.Sequential()
    
    # 1. Capa de entrada: explícitamente fijada a la cantidad de features (input_dim)
    model.add(keras.layers.Input(shape=(input_dim,)))
    model.add(keras.layers.Dense(input_dim, activation=activation))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    # 2. Capas intermedias: Hyperband itera en la cantidad de neuronas para CADA capa
    for i in range(n_layers):
        # Creamos una variable hp diferente para cada capa (ej. units_0, units_1)
        units = hp.Int(f"units_{i+1}", min_value=32, max_value=256, step=32)
        model.add(keras.layers.Dense(units, activation=activation))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.2))

    # 3. Capa de salida: Sigmoid porque es un problema binario (Compra / No Compra)
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ==============================================================================
# 5. FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ==============================================================================
def main():
    print("🤖 Iniciando el entrenamiento del Modelo de Intención de Compra...\n")
    
    # --- PASO A: Cargar Datos ---
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"✅ Dataset cargado correctamente: {len(df)} filas.")
        
        # Eliminar filas repetidas
        filas_antes = len(df)
        df = df.drop_duplicates()
        filas_despues = len(df)
        if filas_antes != filas_despues:
            print(f"🧹 Se eliminaron {filas_antes - filas_despues} filas duplicadas. Total final: {filas_despues} filas.")
            
    except Exception as e:
        print(f"❌ Error: No se encontró el dataset en '{CSV_PATH}'.")
        return

    # Convertir 'Weekend' y 'Revenue' a números por si vienen como TRUE/FALSE
    for col in ["Weekend", TARGET]:
        if df[col].dtype == object:
            df[col] = df[col].map({"TRUE": True, "FALSE": False, True: True, False: False})
        df[col] = df[col].astype(int)

    # --- PASO A.2: Feature Engineering ---
    df = feature_engineering(df)
    print("🧠 Feature Engineering aplicado (Duraciones promedio calculadas, redundancias eliminadas).")

    # Separar en características (X) y objetivo (y)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # --- PASO B: Dividir Datos ---
    # 80% para entrenar, 20% para probar
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # --- PASO C: Preprocesar Datos ---
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    input_dim = X_train_processed.shape[1]
    print(f"⚙️ Datos preprocesados. Entradas finales al modelo: {input_dim} columnas.")

    # Ajustar Pesos para datos desbalanceados (muchos "No Compra", pocos "Compra")
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # --- PASO D: Buscar el mejor modelo con Hyperband ---
    print("\n🔍 Buscando los mejores hiperparámetros con Hyperband...")
    # Englobamos build_model para pasarle el input_dim
    from functools import partial
    model_builder = partial(build_model, input_dim=input_dim)

    tuner = kt.Hyperband(
        hypermodel=model_builder,
        # Cambiamos el objetivo a minimizar la pérdida (val_loss) en lugar del accuracy.
        # En datos desbalanceados, el val_loss continuo es mucho mejor guía matemática.
        objective=kt.Objective("val_loss", direction="min"),
        max_epochs=50,          # 300 es excesivo para datos tabulares (causará overfitting masivo)
        factor=3,
        directory="tuner_results",
        project_name="purchase_intent",
        overwrite=True
    )

    # Buscar el mejor
    tuner.search(
        X_train_processed, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=64, # Lotes más pequeños (64 en vez de 256) ayudan a generalizar mejor
        class_weight=class_weights,
        # Aumentamos la paciencia a 10 para que no aborte prematuramente por los picos
        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )

    # Extraer el mejor
    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]
    
    # --- PASO E: Evaluación ---
    print("\n📈 Evaluando el mejor modelo...")
    y_prob = best_model.predict(X_test_processed, verbose=0).ravel()
    
    # Encontrar el threshold óptimo por F1 (por problemas desbalanceados)
    best_t, best_f1 = 0.5, 0.0
    for t in [i/100 for i in range(20, 80)]:
        y_tmp = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, y_tmp, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
            
    y_pred = (y_prob >= best_t).astype(int)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print(f"Accuracy Final: {acc*100:.2f}%")
    print(f"Threshold Óptimo Utilizado: {best_t:.2f}")
    print("="*40)
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=["No compra", "Compra"]))

    # --- PASO F: Guardar los archivos generados para la API ---
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    
    # 1. Guardar el Modelo
    best_model.save(MODEL_PATH)
    
    # 2. Guardar el Preprocesador
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    
    # 3. Guardar Metadatos e Hiperparámetros
    info = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "input_dim": input_dim,
        "test_accuracy": round(acc, 4),
        "threshold": best_t,
        "best_hyperparameters": best_hp.values,
    }
    MODEL_INFO_PATH.write_text(json.dumps(info, indent=2))
    
    print(f"📁 ¡Entrenamiento Finalizado! Archivos guardados en '{ARTIFACTS_DIR}' listos para la API.")


if __name__ == "__main__":
    main()
