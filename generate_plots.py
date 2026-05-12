import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Configuraciones
CSV_PATH = "data/dataset_shop.csv"
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.keras"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
INFO_PATH = ARTIFACTS_DIR / "model_info.json"

try:
    from train import (
        TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES, 
        feature_engineering, build_model, build_preprocessor
    )
except ImportError:
    print("⚠️ Error importando dependencias de train.py. Asegúrate de estar en la raíz del proyecto.")

def plot_correlation_matrix():
    print("📊 Generando Matriz de Correlación...")
    df = pd.read_csv(CSV_PATH)
    
    # Feature Engineering para ver las correlaciones reales del modelo final
    df = feature_engineering(df)
    
    # Solo usamos características numéricas para la correlación lineal
    numeric_df = df[NUMERIC_FEATURES].copy()
    
    # Calculamos correlación
    corr = numeric_df.corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matriz de Correlación de Características Numéricas", fontsize=14, pad=20)
    plt.tight_layout()
    
    out_path = ARTIFACTS_DIR / "correlacion.png"
    plt.savefig(out_path, dpi=300)
    print(f"✅ Guardado: {out_path}")
    plt.close()

def plot_confusion_matrix():
    print("🧮 Generando Cuadro de Confusión...")
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        print("⚠️ No se encontró el modelo o preprocesador. Asegúrate de ejecutar train.py primero.")
        return

    # Cargar artefactos
    model = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    info = json.loads(INFO_PATH.read_text())
    threshold = info.get("threshold", 0.5)

    # Cargar y preparar datos exactamente igual que en train.py
    df = pd.read_csv(CSV_PATH)
    df = df.drop_duplicates()
    for col in ["Weekend", TARGET]:
        if df[col].dtype == object:
            df[col] = df[col].map({"TRUE": True, "FALSE": False, True: True, False: False})
        df[col] = df[col].astype(int)

    df = feature_engineering(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # Mismo split que usamos en el entrenamiento para evaluar sobre Test Set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    X_test_processed = preprocessor.transform(X_test)
    y_prob = model.predict(X_test_processed, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    # Graficar
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Compra", "Compra"])
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Matriz de Confusión (Test Set)\nThreshold: {threshold:.2f}", fontsize=14, pad=20)
    
    out_path = ARTIFACTS_DIR / "matriz_confusion.png"
    plt.savefig(out_path, dpi=300)
    print(f"✅ Guardado: {out_path}")
    plt.close()

def plot_learning_curve():
    print("📈 Re-entrenando el modelo ganador desde cero para capturar su curva de aprendizaje...")
    try:
        from train import build_model, build_preprocessor
        import keras_tuner as kt
        from functools import partial
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError:
        print("⚠️ Error importando train.py. Asegúrate de estar en la raíz del proyecto.")
        return

    # 1. Cargar y preparar datos (mismo pipeline que train.py)
    df = pd.read_csv(CSV_PATH)
    df = df.drop_duplicates()
    for col in ["Weekend", TARGET]:
        if df[col].dtype == object:
            df[col] = df[col].map({"TRUE": True, "FALSE": False, True: True, False: False})
        df[col] = df[col].astype(int)

    df = feature_engineering(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    input_dim = X_train_processed.shape[1]
    
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    
    # 2. Cargar el tuner para recuperar los mejores hiperparámetros de la carpeta "tuner_results"
    model_builder = partial(build_model, input_dim=input_dim)
    tuner = kt.Hyperband(
        hypermodel=model_builder,
        objective="val_accuracy",
        max_epochs=20,
        factor=3,
        directory="tuner_results",
        project_name="purchase_intent"
    )
    
    try:
        best_hp = tuner.get_best_hyperparameters(1)[0]
    except Exception as e:
        print("⚠️ No se encontraron resultados del tuner. Termina de ejecutar train.py primero.")
        return

    # 3. Construir un modelo fresco con esos hiperparámetros y entrenarlo
    model = tuner.hypermodel.build(best_hp)
    
    history = model.fit(
        X_train_processed, y_train,
        validation_split=0.2,
        epochs=30, # Suficientes épocas para ver la evolución
        batch_size=128,
        class_weight=class_weights,
        verbose=0 # Silencioso para no ensuciar la terminal
    )
    
    # 4. Graficar
    plt.figure(figsize=(14, 6))
    
    # Gráfico de Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='#6c63ff', linewidth=2.5)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#00d4aa', linewidth=2.5)
    plt.title('Curva de Aprendizaje - Precisión (Accuracy)', fontsize=14, pad=15)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Gráfico de Pérdida (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='#ff5f7e', linewidth=2.5)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='#ff9a5c', linewidth=2.5)
    plt.title('Curva de Aprendizaje - Pérdida (Loss)', fontsize=14, pad=15)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = ARTIFACTS_DIR / "curva_aprendizaje.png"
    plt.savefig(out_path, dpi=300)
    print(f"✅ Guardado: {out_path}")
    plt.close()

if __name__ == "__main__":
    print("--- GENERADOR DE GRÁFICAS PARA README ---")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    plot_correlation_matrix()
    plot_confusion_matrix()
    plot_learning_curve()
    
    print("\n💡 Tip: Para incluir estas imágenes en tu README.md usa:")
    print("![Matriz de Correlación](artifacts/correlacion.png)")
    print("![Matriz de Confusión](artifacts/matriz_confusion.png)")
    print("![Curva de Aprendizaje](artifacts/curva_aprendizaje.png)")
