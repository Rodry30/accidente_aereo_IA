# -*- coding: utf-8 -*-
"""
Script 03: Preprocesamiento de Características (Random Forest)
Autor: Antigravity AI
Descripción: Carga 'features.csv', limpia NaNs/infinitos, analiza el balance de clases,
             divide en train/test estratificado, normaliza y guarda las matrices procesadas.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
BASE_DIR = Path(r"C:\Users\rodri\Documents\8vo ciclo\Inteligencia Artificial\accidente_aereo_IA\Entrenamiento_papas\Clasificacion_papas_rf")
INPUT_CSV = BASE_DIR / "features.csv"
OUTPUT_NPZ = BASE_DIR / "features_procesado.npz"
SCALER_PKL = BASE_DIR / "scaler.pkl"
LABEL_MAP_PKL = BASE_DIR / "label_map.pkl"
PLOT_PATH = BASE_DIR / "distribucion_clases.png"

def main():
    print("=" * 80)
    print(" INICIANDO PREPROCESAMIENTO DE DATOS (RF) ".center(80, "="))
    print("=" * 80)
    
    # 1. Verificar existencia del CSV de entrada
    if not INPUT_CSV.exists():
        print(f"[ERROR] No se encontró el archivo '{INPUT_CSV.resolve()}'.")
        print("Por favor, ejecuta primero '02_extraer_features.py'.")
        sys.exit(1)
        
    print(f"[INFO] Cargando características desde: {INPUT_CSV.name}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"[INFO] Datos cargados. Dimensiones: {df.shape}")
    
    # 2. Limpieza de Datos
    if "filepath" in df.columns:
        df = df.drop(columns=["filepath"])
        print("[INFO] Columna 'filepath' eliminada por no ser variable predictora.")
        
    # Detectar NaNs e infinitos
    nan_counts = df.isna().sum().sum()
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    
    print(f"[INFO] Valores NaN detectados: {nan_counts}")
    print(f"[INFO] Valores Infinitos detectados: {inf_counts}")
    
    if inf_counts > 0:
        df = df.replace([np.inf, -np.inf], np.nan)
        
    total_inicial = len(df)
    df = df.dropna()
    filas_eliminadas = total_inicial - len(df)
    
    if filas_eliminadas > 0:
        print(f"[INFO] Se eliminaron {filas_eliminadas} filas por nulos o infinitos.")
    else:
        print("[INFO] No se requirió limpieza de nulos. Datos limpios.")
        
    # 3. Análisis de Desbalance
    conteo_clases = df.groupby("label").size().reset_index(name="Cantidad")
    conteo_clases = conteo_clases.sort_values(by="Cantidad", ascending=False)
    
    print("\nVariedades con más imágenes:")
    for idx, row in conteo_clases.head(5).iterrows():
        print(f"  - {row['label']}: {row['Cantidad']} imágenes")
        
    print("\nVariedades con menos imágenes:")
    for idx, row in conteo_clases.tail(5).iterrows():
        print(f"  - {row['label']}: {row['Cantidad']} imágenes")
        
    # Graficar distribución
    plt.figure(figsize=(12, 18))
    sns.barplot(
        x="Cantidad",
        y="label",
        data=conteo_clases,
        palette="viridis",
        hue="label",
        legend=False
    )
    plt.title("Distribución de Imágenes por Variedad (Proyecto Random Forest)", fontsize=14, fontweight='bold')
    plt.xlabel("Cantidad de Imágenes", fontsize=12)
    plt.ylabel("Variedad (Clase)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"\n[INFO] Gráfico de distribución guardado en: {PLOT_PATH.resolve()}")
    
    # 4. Separación X, y
    X = df.drop(columns=["label", "label_id"])
    y = df["label_id"]
    
    # Guardar label map
    label_map = df.groupby("label_id")["label"].first().to_dict()
    joblib.dump(label_map, LABEL_MAP_PKL)
    print(f"[INFO] Mapa de etiquetas guardado en: {LABEL_MAP_PKL.name}")
    
    # 5. Train/Test Split (80/20 estratificado)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print("\n" + "-" * 50)
    print(" DIVISIÓN DEL DATASET (80% Train, 20% Test) ".center(50, "-"))
    print("-" * 50)
    print(f"Muestras de Entrenamiento (X_train): {X_train.shape[0]}")
    print(f"Muestras de Prueba (X_test):        {X_test.shape[0]}")
    print(f"Número de características:           {X_train.shape[1]}")
    
    # 6. Normalización (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, SCALER_PKL)
    print(f"\n[INFO] StandardScaler guardado en: {SCALER_PKL.name}")
    
    # 7. Guardar en NPZ
    np.savez(
        OUTPUT_NPZ,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy()
    )
    print(f"[INFO] Matrices procesadas guardadas en: {OUTPUT_NPZ.name}")
    print("\n" + "=" * 80)
    print(" PREPROCESAMIENTO FINALIZADO CON ÉXITO ".center(80, "-"))
    print("=" * 80)

if __name__ == "__main__":
    main()
