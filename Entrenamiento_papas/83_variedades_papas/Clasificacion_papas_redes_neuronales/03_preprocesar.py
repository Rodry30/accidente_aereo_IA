# -*- coding: utf-8 -*-
"""
Script 03: Preprocesamiento de Características
Descripción: Carga 'features.csv', limpia NaNs/infinitos, analiza el balance de clases,
             divide en train/test de manera estratificada, normaliza las características
             numéricas y guarda los archivos resultantes.
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
INPUT_CSV = Path("features.csv")
OUTPUT_NPZ = Path("features_procesado.npz")
SCALER_PKL = Path("scaler.pkl")
LABEL_MAP_PKL = Path("label_map.pkl")
PLOT_PATH = Path("distribucion_clases.png")

def main():
    print("=" * 80)
    print(" INICIANDO PREPROCESAMIENTO DE DATOS ".center(80, "="))
    print("=" * 80)
    
    # 1. Verificar existencia del archivo de entrada
    if not INPUT_CSV.exists():
        print(f"[ERROR] No se encontró el archivo de entrada '{INPUT_CSV.resolve()}'.")
        print("Por favor, ejecuta primero '02_extraer_features.py' para generar la matriz de características.")
        sys.exit(1)
        
    print(f"[INFO] Cargando características desde: {INPUT_CSV.name}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"[INFO] Datos cargados con éxito. Dimensiones iniciales: {df.shape}")
    
    # 2. Limpieza de Datos
    # Separar filepath temporalmente o descartarlo
    if "filepath" in df.columns:
        df = df.drop(columns=["filepath"])
        print("[INFO] Columna 'filepath' eliminada por no ser una característica de entrenamiento.")
        
    # Detectar NaN o Infinitos
    nan_counts = df.isna().sum().sum()
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    
    print(f"[INFO] Valores NaN detectados: {nan_counts}")
    print(f"[INFO] Valores Infinitos detectados: {inf_counts}")
    
    # Reemplazar infinitos por NaN y eliminar filas con NaN
    if inf_counts > 0:
        df = df.replace([np.inf, -np.inf], np.nan)
        
    total_filas_inicial = len(df)
    df = df.dropna()
    filas_eliminadas = total_filas_inicial - len(df)
    
    if filas_eliminadas > 0:
        print(f"[INFO] Se eliminaron {filas_eliminadas} filas debido a valores nulos o infinitos.")
    else:
        print("[INFO] No se requirió eliminar ninguna fila. Datos limpios.")
        
    # 3. Análisis de Desbalance
    conteo_clases = df.groupby("label").size().reset_index(name="Cantidad")
    conteo_clases = conteo_clases.sort_values(by="Cantidad", ascending=False)
    
    # Reportar top 5 y bottom 5 clases
    print("\n" + "-" * 50)
    print(" ANÁLISIS DE BALANCE DE CLASES ".center(50, "-"))
    print("-" * 50)
    print("Las 5 variedades con MÁS imágenes:")
    for idx, row in conteo_clases.head(5).iterrows():
        print(f"  - {row['label']}: {row['Cantidad']} imágenes")
        
    print("\nLas 5 variedades con MENOS imágenes:")
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
    plt.title("Distribución de Cantidad de Imágenes por Variedad de Papa Nativa", fontsize=14, fontweight='bold')
    plt.xlabel("Cantidad de Imágenes", fontsize=12)
    plt.ylabel("Variedad (Clase)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"\n[INFO] Gráfico de distribución guardado como: {PLOT_PATH.resolve()}")
    
    # 4. Separación X, y
    X = df.drop(columns=["label", "label_id"])
    y = df["label_id"]
    
    # Crear y guardar el mapa de etiquetas id -> nombre
    label_map = df.groupby("label_id")["label"].first().to_dict()
    joblib.dump(label_map, LABEL_MAP_PKL)
    print(f"[INFO] Mapeo de etiquetas guardado en: {LABEL_MAP_PKL.name}")
    
    # 5. Train/Test Split
    # Split de 80% entrenamiento y 20% prueba estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print("\n" + "-" * 50)
    print(" DIVISIÓN DEL DATASET (80% Train, 20% Test) ".center(50, "-"))
    print("-" * 50)
    print(f"Total Entrenamiento (X_train): {X_train.shape[0]} muestras")
    print(f"Total Prueba (X_test):        {X_test.shape[0]} muestras")
    print(f"Número de características:     {X_train.shape[1]}")
    
    # 6. Normalización de características
    scaler = StandardScaler()
    
    # Fit y transform sobre Train, y transform sobre Test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Guardar scaler
    joblib.dump(scaler, SCALER_PKL)
    print(f"\n[INFO] StandardScaler guardado con éxito en: {SCALER_PKL.name}")
    
    # 7. Guardar conjuntos de datos en NPZ
    np.savez(
        OUTPUT_NPZ,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy()
    )
    print(f"[INFO] Matrices procesadas guardadas en: {OUTPUT_NPZ.name}")
    print("\n" + "=" * 80)
    print(" PREPROCESAMIENTO COMPLETADO EXITOSAMENTE ".center(80, "-"))
    print("=" * 80)

if __name__ == "__main__":
    main()
