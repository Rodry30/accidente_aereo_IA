#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de Entrenamiento: Clasificador Random Forest de Variedades de Papa por Color
----------------------------------------------------------------------------------
Objetivo:
    Entrenar un modelo Random Forest para clasificar variedades de papa en una de 
    las 5 clases de color identificadas en el reporte de color, basándose en 
    características visuales extraídas de las imágenes (RGB, HSV, Histograma, Textura).

Entradas:
    - mapeo_variedades_colores.csv: Archivo de mapeo de variedad a grupo/cluster de color.
    - dataset128x128: Carpeta de imágenes por variedad (V1, V2, ..., V84).

Salidas (guardadas en rf_papas_bycolor/):
    - modelo_rf_colores.pkl (Modelo RF entrenado)
    - scaler_colores.pkl (StandardScaler para normalizar características)
    - label_encoder_colores.pkl (LabelEncoder para clases de color)
    - metrics_rf_colores.json (Métricas en formato estructurado JSON)
    - reporte_rf_colores.txt (Reporte legible para publicaciones científicas)
    - confusion_matrix_rf_colores.png (Gráfico de la matriz de confusión)
    - importancia_features_rf_colores.png (Gráfico de importancia de características)
    - resultados_rf_colores.csv (Detalles de predicciones para cada imagen)
"""

import os
import json
import pickle
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Ignorar advertencias menores de graficación y procesado
warnings.filterwarnings("ignore")

# Configuración de Rutas del Proyecto
BASE_DIR = Path(__file__).resolve().parents[3]
DATASET_PATH = BASE_DIR / "Entrenamiento_papas" / "dataset" / "dataset128x128"
MAPEO_CSV_PATH = BASE_DIR / "Entrenamiento_papas" / "Clasificacion_papas" / "mapeo_variedades_colores.csv"
OUTPUT_DIR = BASE_DIR / "Entrenamiento_papas" / "colores_papas" / "rf_papas_bycolor"

# Mapeo de clusters numéricos a etiquetas descriptivas de las 5 clases del reporte
CLUSTER_TO_LABEL = {
    0: "Naranja brillante - tipo1",
    1: "Azul/Morado",
    2: "Naranja brillante - tipo2",
    3: "Naranja brillante - tipo3",
    4: "Amarillo suave"
}

# Nombres de las características para visualización de importancia
FEATURE_NAMES = [
    "RGB_R_Mean", "RGB_G_Mean", "RGB_B_Mean",
    "RGB_R_Std", "RGB_G_Std", "RGB_B_Std",
    "HSV_H_Mean", "HSV_S_Mean", "HSV_V_Mean",
    "Intensity_Variance", "Edge_Mean_Sobel",
    "Hist_Bin_0", "Hist_Bin_1", "Hist_Bin_2", "Hist_Bin_3", "Hist_Bin_4",
    "Hist_Bin_5", "Hist_Bin_6", "Hist_Bin_7", "Hist_Bin_8", "Hist_Bin_9"
]


def extraer_caracteristicas_imagen(img_path: Path) -> np.ndarray | None:
    """
    Carga una imagen y extrae un vector de características visuales:
      - RGB promedio (ignorando fondo)
      - HSV promedio (ignorando fondo)
      - Desviación estándar de RGB
      - Histograma de intensidad en escala de grises (10 bins)
      - Textura simple (varianza de intensidad y promedio de magnitud de Sobel)
    """
    try:
        # Leer imagen usando OpenCV en formato BGR y convertir a RGB
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Máscara para ignorar el fondo blanco/muy claro (RGB > 240)
        # Esto asegura que solo extraemos características de la papa
        mask = ~((img_rgb[:, :, 0] > 240) & 
                 (img_rgb[:, :, 1] > 240) & 
                 (img_rgb[:, :, 2] > 240))
        
        # Si la máscara está vacía (por ejemplo, una imagen completamente blanca), usar toda la imagen
        if not np.any(mask):
            mask = np.ones(img_rgb.shape[:2], dtype=bool)
            
        # Píxeles de la papa en RGB
        pixels_rgb = img_rgb[mask]
        if len(pixels_rgb) == 0:
            return None
            
        # 1. RGB Promedio
        rgb_mean = pixels_rgb.mean(axis=0)
        
        # 2. Desviación estándar de RGB
        rgb_std = pixels_rgb.std(axis=0)
        
        # Convertir a HSV para extraer promedio HSV
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        pixels_hsv = img_hsv[mask]
        hsv_mean = pixels_hsv.mean(axis=0)
        
        # Convertir a escala de grises para textura e histograma de intensidad
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        pixels_gray = img_gray[mask]
        
        # 3. Histograma de intensidad (10 bins normalizados)
        hist, _ = np.histogram(pixels_gray, bins=10, range=(0, 256))
        hist = hist.astype(float)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum
            
        # 4. Textura simple: Varianza de la intensidad
        intensity_variance = pixels_gray.var()
        
        # 5. Textura simple: Promedio de magnitud de bordes de Sobel
        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_mean = sobel_magnitude[mask].mean()
        
        # Concatenar todas las características en un solo vector de tamaño 21
        vector_features = np.hstack([
            rgb_mean,         # 3 features
            rgb_std,          # 3 features
            hsv_mean,         # 3 features
            intensity_variance, # 1 feature
            edge_mean,        # 1 feature
            hist              # 10 features
        ])
        
        return vector_features
        
    except Exception as e:
        print(f"  Advertencia: Error al procesar la imagen {img_path.name}: {e}")
        return None


def main():
    print("=" * 80)
    print("INICIANDO ENTRENAMIENTO DEL MODELO RANDOM FOREST (COLORES)")
    print("=" * 80)
    
    # Crear directorio de salida si no existe
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Leer el CSV de mapeo_variedades_colores.csv
    if not MAPEO_CSV_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de mapeo en: {MAPEO_CSV_PATH}")
        
    print(f"Cargando mapeo de variedades desde: {MAPEO_CSV_PATH}")
    df_mapeo = pd.read_csv(MAPEO_CSV_PATH)
    
    # Crear diccionario de mapeo variedad -> clase de color
    mapeo_clases = {}
    for _, row in df_mapeo.iterrows():
        var_name = str(row['variedad']).strip()
        cluster_id = int(row['cluster'])
        class_label = CLUSTER_TO_LABEL.get(cluster_id)
        if class_label:
            mapeo_clases[var_name] = class_label
            
    print(f"Mapeo cargado con éxito. Total variedades en mapeo: {len(mapeo_clases)}")
    
    # 2. Recorrer las carpetas de imágenes y extraer características
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No se encontró la carpeta del dataset en: {DATASET_PATH}")
        
    print(f"Extrayendo características de imágenes en: {DATASET_PATH}")
    
    X_list = []
    y_list = []
    image_paths_list = []
    variedad_list = []
    
    # Obtener subcarpetas ordenadas
    carpetas_variedad = sorted([d for d in DATASET_PATH.iterdir() if d.is_dir()])
    
    for carpeta in carpetas_variedad:
        var_name = carpeta.name
        
        # Validar si esta variedad está en el CSV de mapeo
        if var_name not in mapeo_clases:
            print(f"  Advertencia: Variedad {var_name} no se encuentra en el archivo de mapeo. Omitiendo.")
            continue
            
        color_class = mapeo_clases[var_name]
        
        # Buscar imágenes válidas (.jpg, .png)
        imagenes = list(carpeta.glob("*.jpg")) + list(carpeta.glob("*.png"))
        
        if not imagenes:
            print(f"  Advertencia: La variedad {var_name} no contiene imágenes en su carpeta. Omitiendo variedad.")
            continue
            
        # Procesar imágenes de esta variedad
        caracteristicas_variedad = []
        for img_path in imagenes:
            feat = extraer_caracteristicas_imagen(img_path)
            if feat is not None:
                X_list.append(feat)
                y_list.append(color_class)
                # Guardar ruta relativa para el archivo de resultados CSV
                rel_path = img_path.relative_to(BASE_DIR).as_posix()
                image_paths_list.append(rel_path)
                variedad_list.append(var_name)
                caracteristicas_variedad.append(feat)
                
        # Si no se pudo extraer ninguna imagen válida para la variedad
        if len(caracteristicas_variedad) == 0:
            print(f"  Advertencia: La variedad {var_name} no tiene ninguna imagen válida procesable. Omitiendo variedad.")
            
    # Convertir a arrays de numpy
    X = np.array(X_list)
    y = np.array(y_list)
    
    if len(X) == 0:
        raise ValueError("No se extrajeron imágenes o características válidas. Deteniendo el script.")
        
    print(f"[OK] Procesamiento completado.")
    print(f"  Total de imágenes procesadas: {len(X)}")
    print(f"  Dimensiones del dataset de características: {X.shape}")
    
    # 3. Codificación de etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    clases_unicas = le.classes_
    print(f"  Clases encontradas ({len(clases_unicas)}): {list(clases_unicas)}")
    
    # Mostrar distribución de clases original
    conteo_clases = pd.Series(y).value_counts()
    print("\nDistribución de clases en el dataset completo:")
    for c, val in conteo_clases.items():
        print(f"  - {c}: {val} imágenes")
        
    # 4. División de datos en entrenamiento (80%) y prueba (20%) estratificado
    # Mantenemos los índices para hacer el seguimiento de las imágenes
    indices = np.arange(len(X))
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        indices, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_train = X[X_train_idx]
    X_test = X[X_test_idx]
    
    print(f"\nDivisión de datos:")
    print(f"  - Conjunto de entrenamiento: {len(X_train)} muestras")
    print(f"  - Conjunto de prueba: {len(X_test)} muestras")
    
    # 5. Normalización de características (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Entrenamiento del modelo Random Forest
    print("\nEntrenando modelo Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
    rf_model.fit(X_train_scaled, y_train)
    print("[OK] Modelo entrenado con éxito.")
    
    # 7. Evaluación en el conjunto de prueba
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    clf_rep_dict = classification_report(y_test, y_pred, target_names=clases_unicas, output_dict=True)
    clf_rep_text = classification_report(y_test, y_pred, target_names=clases_unicas)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    print("\n================ METRICAS PRINCIPALES (TEST SET) ================")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1-Score Macro: {f1_macro:.4f}")
    print("\nReporte de Clasificación:")
    print(clf_rep_text)
    print("==================================================================")
    
    # 8. Guardar Archivos de Resultados
    print(f"\nGuardando resultados en: {OUTPUT_DIR}")
    
    # A. Guardar los serializables en formato pickle (.pkl)
    with open(OUTPUT_DIR / "modelo_rf_colores.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open(OUTPUT_DIR / "scaler_colores.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(OUTPUT_DIR / "label_encoder_colores.pkl", "wb") as f:
        pickle.dump(le, f)
    print("  [OK] modelo_rf_colores.pkl, scaler_colores.pkl, label_encoder_colores.pkl guardados.")
    
    # B. Guardar métricas JSON
    metrics_json = {
        "total_muestras": int(len(X)),
        "samples_train": int(len(X_train)),
        "samples_test": int(len(X_test)),
        "accuracy": float(acc),
        "f1_score_macro": float(f1_macro),
        "clases": list(clases_unicas),
        "classification_report": clf_rep_dict
    }
    with open(OUTPUT_DIR / "metrics_rf_colores.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=4, ensure_ascii=False)
    print("  [OK] metrics_rf_colores.json guardado.")
    
    # C. Guardar reporte descriptivo TXT
    reporte_txt = []
    reporte_txt.append("=" * 80)
    reporte_txt.append("REPORTE DETALLADO DE CLASIFICACIÓN DE PAPAS POR GRUPO DE COLOR")
    reporte_txt.append("MODELO: RANDOM FOREST CLASSIFIER")
    reporte_txt.append("=" * 80)
    reporte_txt.append(f"Fecha: 2026-06-12 (Local Time)")
    reporte_txt.append(f"Dataset de origen: dataset128x128")
    reporte_txt.append(f"Mapeo de referencia: mapeo_variedades_colores.csv")
    reporte_txt.append("-" * 80)
    reporte_txt.append("CONFIGURACIÓN DEL MODELO:")
    reporte_txt.append("  - Clasificador: RandomForestClassifier")
    reporte_txt.append("  - Estimadores (n_estimators): 300")
    reporte_txt.append("  - Semilla aleatoria (random_state): 42")
    reporte_txt.append("  - Pesos de clase (class_weight): balanced")
    reporte_txt.append("  - División del conjunto (train/test): 80% / 20% (Estratificado)")
    reporte_txt.append("-" * 80)
    reporte_txt.append("RESUMEN DE DATOS:")
    reporte_txt.append(f"  - Total muestras procesadas: {len(X)}")
    reporte_txt.append(f"  - Muestras de entrenamiento: {len(X_train)}")
    reporte_txt.append(f"  - Muestras de prueba: {len(X_test)}")
    reporte_txt.append(f"  - Número de clases: {len(clases_unicas)}")
    for c, val in conteo_clases.items():
        reporte_txt.append(f"    * {c}: {val} imágenes")
    reporte_txt.append("-" * 80)
    reporte_txt.append("MÉTRICAS DEL CONJUNTO DE PRUEBA:")
    reporte_txt.append(f"  - Exactitud (Accuracy): {acc:.6f} ({acc*100:.2f}%)")
    reporte_txt.append(f"  - F1-Score Macro: {f1_macro:.6f}")
    reporte_txt.append("\nReporte de Clasificación:")
    reporte_txt.append(clf_rep_text)
    reporte_txt.append("-" * 80)
    reporte_txt.append("MATRIZ DE CONFUSIÓN:")
    reporte_txt.append(str(conf_mat))
    reporte_txt.append("\nEtiquetas correspondientes:")
    for idx, c in enumerate(clases_unicas):
        reporte_txt.append(f"  [{idx}]: {c}")
    reporte_txt.append("=" * 80)
    
    reporte_txt_content = "\n".join(reporte_txt)
    with open(OUTPUT_DIR / "reporte_rf_colores.txt", "w", encoding="utf-8") as f:
        f.write(reporte_txt_content)
    print("  [OK] reporte_rf_colores.txt guardado.")
    
    # D. Generar y guardar gráfico de matriz de confusión (PNG)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_mat, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=clases_unicas, 
        yticklabels=clases_unicas,
        cbar=True,
        square=True
    )
    plt.title("Matriz de Confusión — Random Forest (Colores Papas)\nDataset de Prueba", fontsize=14, pad=20)
    plt.xlabel("Clase Predicha", fontsize=12, labelpad=10)
    plt.ylabel("Clase Real", fontsize=12, labelpad=10)
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_rf_colores.png", dpi=300)
    plt.close()
    print("  [OK] confusion_matrix_rf_colores.png guardado.")
    
    # E. Generar y guardar gráfico de importancia de características (PNG)
    importancia = rf_model.feature_importances_
    df_importancia = pd.DataFrame({
        "Caracteristica": FEATURE_NAMES,
        "Importancia": importancia
    }).sort_values("Importancia", ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=df_importancia, 
        x="Importancia", 
        y="Caracteristica", 
        palette="viridis",
        edgecolor="black"
    )
    plt.title("Importancia de Características Visuales — Random Forest (Colores)", fontsize=14, pad=15)
    plt.xlabel("Importancia Relativa (Gini Importance)", fontsize=12)
    plt.ylabel("Característica", fontsize=12)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "importancia_features_rf_colores.png", dpi=300)
    plt.close()
    print("  [OK] importancia_features_rf_colores.png guardado.")
    
    # F. Guardar predicciones de todo el dataset en resultados_rf_colores.csv
    # Para ello, necesitamos predecir sobre todo el dataset (X normalizado por completo)
    X_scaled_all = scaler.transform(X)
    y_pred_all = rf_model.predict(X_scaled_all)
    y_pred_all_labels = le.inverse_transform(y_pred_all)
    y_pred_all_proba = rf_model.predict_proba(X_scaled_all)
    
    # Crear un identificador de a qué split pertenece cada imagen (Train/Test)
    split_col = np.array(["train"] * len(X))
    split_col[X_test_idx] = "test"
    
    # Crear columnas de probabilidad por clase
    proba_cols = {}
    for idx, c in enumerate(clases_unicas):
        # Limpiar nombre de clase para el header del CSV
        c_clean = c.replace(" - ", "_").replace("/", "_").replace(" ", "_")
        proba_cols[f"prob_{c_clean}"] = y_pred_all_proba[:, idx]
        
    resultados_dict = {
        "imagen_path": image_paths_list,
        "variedad": variedad_list,
        "color_grupo_true": y_list,
        "color_grupo_pred": y_pred_all_labels,
        "split": split_col
    }
    # Añadir probabilidades
    resultados_dict.update(proba_cols)
    
    df_resultados = pd.DataFrame(resultados_dict)
    df_resultados.to_csv(OUTPUT_DIR / "resultados_rf_colores.csv", index=False, encoding="utf-8")
    print("  [OK] resultados_rf_colores.csv guardado.")
    
    print("\nPROCESO DE ENTRENAMIENTO FINALIZADO EXITOSAMENTE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
