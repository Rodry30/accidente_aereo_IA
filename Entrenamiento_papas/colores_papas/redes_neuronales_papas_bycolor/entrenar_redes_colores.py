#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de Entrenamiento: Red Neuronal MLP para Clasificación de Papas por Color
----------------------------------------------------------------------------------
Objetivo:
    Entrenar una Red Neuronal MLP (Multi-Layer Perceptron) para clasificar las 
    variedades de papa en una de las 5 clases de color del reporte, basándose en 
    características tabulares (RGB, HSV, Histograma, Varianza, Bordes y Entropía).

Entradas:
    - mapeo_variedades_colores.csv: Archivo de mapeo de variedad a grupo de color.
    - dataset128x128: Carpeta de imágenes por variedad (V1, V2, ..., V84).

Salidas (guardadas en redes_neuronales_papas_bycolor/):
    - modelo_mlp_colores.keras (Modelo de Red Neuronal entrenado en formato nativo Keras)
    - scaler_colores.pkl (StandardScaler para normalizar características)
    - label_encoder_colores.pkl (LabelEncoder para clases de color)
    - metrics_mlp_colores.json (Métricas detalladas en formato JSON)
    - reporte_mlp_colores.txt (Reporte legible en formato de texto científico)
    - confusion_matrix_mlp_colores.png (Matriz de confusión en formato gráfico)
    - training_history_mlp_colores.png (Curvas de entrenamiento de Loss y Accuracy)
    - resultados_mlp_colores.csv (Detalles de predicciones para cada imagen)

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

# Configurar logs de TensorFlow silenciosos para evitar inundar la consola
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Ignorar advertencias de graficación
warnings.filterwarnings("ignore")

# Configurar semilla de aleatoriedad para reproducibilidad científica
np.random.seed(42)
tf.random.set_seed(42)

# Configuración de Rutas
BASE_DIR = Path(__file__).resolve().parents[3]
DATASET_PATH = BASE_DIR / "Entrenamiento_papas" / "dataset" / "dataset128x128"
MAPEO_CSV_PATH = BASE_DIR / "Entrenamiento_papas" / "Clasificacion_papas" / "mapeo_variedades_colores.csv"
OUTPUT_DIR = BASE_DIR / "Entrenamiento_papas" / "colores_papas" / "redes_neuronales_papas_bycolor"

# Mapeo de clusters a las 5 clases descriptivas del reporte
CLUSTER_TO_LABEL = {
    0: "Naranja brillante - tipo1",
    1: "Azul/Morado",
    2: "Naranja brillante - tipo2",
    3: "Naranja brillante - tipo3",
    4: "Amarillo suave"
}

# Nombres de las 22 características
FEATURE_NAMES = [
    "RGB_R_Mean", "RGB_G_Mean", "RGB_B_Mean",
    "RGB_R_Std", "RGB_G_Std", "RGB_B_Std",
    "HSV_H_Mean", "HSV_S_Mean", "HSV_V_Mean",
    "Intensity_Variance", "Edge_Mean_Sobel", "Shannon_Entropy",
    "Hist_Bin_0", "Hist_Bin_1", "Hist_Bin_2", "Hist_Bin_3", "Hist_Bin_4",
    "Hist_Bin_5", "Hist_Bin_6", "Hist_Bin_7", "Hist_Bin_8", "Hist_Bin_9"
]


def extraer_caracteristicas_imagen(img_path: Path) -> np.ndarray | None:
    """
    Carga una imagen y extrae un vector de 22 características tabulares:
      - RGB promedio (3)
      - RGB desviación estándar (3)
      - HSV promedio (3)
      - Varianza de la intensidad de gris (1)
      - Magnitud promedio de bordes de Sobel (1)
      - Entropía de Shannon (1)
      - Histograma de intensidad en escala de grises (10 bins)
    """
    try:
        # Cargar la imagen y convertir a RGB
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Máscara para ignorar el fondo blanco de estudio (RGB > 240)
        mask = ~((img_rgb[:, :, 0] > 240) & 
                 (img_rgb[:, :, 1] > 240) & 
                 (img_rgb[:, :, 2] > 240))
        
        if not np.any(mask):
            mask = np.ones(img_rgb.shape[:2], dtype=bool)
            
        pixels_rgb = img_rgb[mask]
        if len(pixels_rgb) == 0:
            return None
            
        # 1. Promedio RGB
        rgb_mean = pixels_rgb.mean(axis=0)
        
        # 2. Desviación Estándar RGB
        rgb_std = pixels_rgb.std(axis=0)
        
        # 3. Promedio HSV
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        pixels_hsv = img_hsv[mask]
        hsv_mean = pixels_hsv.mean(axis=0)
        
        # 4. Escala de grises para histograma y textura
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        pixels_gray = img_gray[mask]
        
        # Histograma de intensidad normalizado (10 bins)
        hist, _ = np.histogram(pixels_gray, bins=10, range=(0, 256))
        hist = hist.astype(float)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum
            
        # Varianza de intensidad
        intensity_variance = pixels_gray.var()
        
        # Promedio de bordes (Magnitud de Sobel)
        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_mean = sobel_magnitude[mask].mean()
        
        # Entropía de Shannon del área enmascarada
        _, counts = np.unique(pixels_gray, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Consolidar en vector único de 22 dimensiones
        vector_features = np.hstack([
            rgb_mean,            # 3
            rgb_std,             # 3
            hsv_mean,            # 3
            intensity_variance,   # 1
            edge_mean,           # 1
            entropy,             # 1
            hist                 # 10
        ])
        
        return vector_features
        
    except Exception as e:
        print(f"  Advertencia: Error al procesar imagen {img_path.name}: {e}")
        return None


def main():
    print("=" * 80)
    print("INICIANDO ENTRENAMIENTO DE LA RED NEURONAL MLP (COLORES)")
    print("=" * 80)
    
    # Crear carpeta de salida
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar Mapeo
    if not MAPEO_CSV_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de mapeo en: {MAPEO_CSV_PATH}")
        
    print(f"Cargando mapeo de variedades desde: {MAPEO_CSV_PATH}")
    df_mapeo = pd.read_csv(MAPEO_CSV_PATH)
    
    mapeo_clases = {}
    for _, row in df_mapeo.iterrows():
        var_name = str(row['variedad']).strip()
        cluster_id = int(row['cluster'])
        class_label = CLUSTER_TO_LABEL.get(cluster_id)
        if class_label:
            mapeo_clases[var_name] = class_label
            
    print(f"Mapeo cargado. Total de variedades en el mapeo: {len(mapeo_clases)}")
    
    # 2. Procesar imágenes del dataset
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No se encontró la carpeta del dataset en: {DATASET_PATH}")
        
    print(f"Procesando imágenes en: {DATASET_PATH}")
    
    X_list = []
    y_list = []
    image_paths_list = []
    variedad_list = []
    
    carpetas_variedad = sorted([d for d in DATASET_PATH.iterdir() if d.is_dir()])
    
    for carpeta in carpetas_variedad:
        var_name = carpeta.name
        
        # Saltar variedades que no están en el mapeo
        if var_name not in mapeo_clases:
            print(f"  Advertencia: Variedad {var_name} no catalogada en el mapeo. Omitiendo.")
            continue
            
        color_class = mapeo_clases[var_name]
        imagenes = list(carpeta.glob("*.jpg")) + list(carpeta.glob("*.png"))
        
        if not imagenes:
            print(f"  Advertencia: Variedad {var_name} no tiene imágenes válidas. Omitiendo.")
            continue
            
        caracteristicas_variedad = []
        for img_path in imagenes:
            feat = extraer_caracteristicas_imagen(img_path)
            if feat is not None:
                X_list.append(feat)
                y_list.append(color_class)
                rel_path = img_path.relative_to(BASE_DIR).as_posix()
                image_paths_list.append(rel_path)
                variedad_list.append(var_name)
                caracteristicas_variedad.append(feat)
                
        if len(caracteristicas_variedad) == 0:
            print(f"  Advertencia: La variedad {var_name} no contiene imágenes procesables. Omitiendo variedad.")
            
    X = np.array(X_list)
    y = np.array(y_list)
    
    if len(X) == 0:
        raise ValueError("No se obtuvieron muestras de características válidas. Deteniendo ejecución.")
        
    print(f"[OK] Procesamiento completado.")
    print(f"  Total de muestras procesadas: {len(X)}")
    print(f"  Dimensiones del dataset de características: {X.shape}")
    
    # 3. Codificar etiquetas a enteros
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    clases_unicas = le.classes_
    print(f"  Clases encontradas ({len(clases_unicas)}): {list(clases_unicas)}")
    
    # Mostrar distribución
    conteo_clases = pd.Series(y).value_counts()
    print("\nDistribución de clases en el dataset completo:")
    for c, val in conteo_clases.items():
        print(f"  - {c}: {val} imágenes")
        
    # 4. Dividir conjunto en entrenamiento (80%) y prueba (20%) con stratify
    indices = np.arange(len(X))
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        indices, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_train = X[X_train_idx]
    X_test = X[X_test_idx]
    
    print(f"\nDivisión de datos:")
    print(f"  - Conjunto de entrenamiento: {len(X_train)} muestras")
    print(f"  - Conjunto de prueba: {len(X_test)} muestras")
    
    # 5. Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Calcular pesos de clase para manejar el desbalance de datos
    classes_train = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes_train,
        y=y_train
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(classes_train, weights)}
    print(f"Pesos de clases calculados (balanceados): {class_weight_dict}")
    
    # 7. Crear el modelo MLP en Keras
    print("\nDefiniendo arquitectura de la Red Neuronal MLP...")
    model = Sequential([
        Input(shape=(22,)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(5, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callback de parada temprana (Early Stopping)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    # 8. Entrenar el modelo
    print("\nIniciando entrenamiento del modelo MLP...")
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1
    )
    print("[OK] Modelo entrenado con éxito.")
    
    # 9. Evaluación en el conjunto de prueba
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    clf_rep_dict = classification_report(y_test, y_pred, target_names=clases_unicas, output_dict=True)
    clf_rep_text = classification_report(y_test, y_pred, target_names=clases_unicas)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    print("\n================ METRICAS PRINCIPALES (TEST SET - MLP) ================")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1-Score Macro: {f1_macro:.4f}")
    print("\nReporte de Clasificación:")
    print(clf_rep_text)
    print("=======================================================================")
    
    # 10. Guardar Archivos de Resultados
    print(f"\nGuardando resultados en: {OUTPUT_DIR}")
    
    # A. Guardar modelo en formato Keras nativo
    model.save(OUTPUT_DIR / "modelo_mlp_colores.keras")
    
    # B. Guardar scaler y encoder con pickle
    with open(OUTPUT_DIR / "scaler_colores.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(OUTPUT_DIR / "label_encoder_colores.pkl", "wb") as f:
        pickle.dump(le, f)
    print("  [OK] modelo_mlp_colores.keras, scaler_colores.pkl, label_encoder_colores.pkl guardados.")
    
    # C. Guardar métricas en JSON
    metrics_json = {
        "total_muestras": int(len(X)),
        "samples_train": int(len(X_train)),
        "samples_test": int(len(X_test)),
        "accuracy": float(acc),
        "f1_score_macro": float(f1_macro),
        "clases": list(clases_unicas),
        "classification_report": clf_rep_dict
    }
    with open(OUTPUT_DIR / "metrics_mlp_colores.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=4, ensure_ascii=False)
    print("  [OK] metrics_mlp_colores.json guardado.")
    
    # D. Guardar reporte descriptivo TXT
    reporte_txt = []
    reporte_txt.append("=" * 80)
    reporte_txt.append("REPORTE DETALLADO DE CLASIFICACIÓN DE PAPAS POR GRUPO DE COLOR")
    reporte_txt.append("MODELO: MULTI-LAYER PERCEPTRON (MLP)")
    reporte_txt.append("=" * 80)
    reporte_txt.append(f"Fecha: 2026-06-12 (Local Time)")
    reporte_txt.append(f"Dataset de origen: dataset128x128")
    reporte_txt.append(f"Mapeo de referencia: mapeo_variedades_colores.csv")
    reporte_txt.append("-" * 80)
    reporte_txt.append("CONFIGURACIÓN DE LA RED NEURONAL:")
    reporte_txt.append("  - Capas Ocultas: 128 (ReLU), 64 (ReLU), 32 (ReLU)")
    reporte_txt.append("  - Capa de Salida: 5 (Softmax)")
    reporte_txt.append("  - Optimizador: Adam (learning_rate=0.001)")
    reporte_txt.append("  - Dropout: 0.2 (Capa 1), 0.1 (Capa 2)")
    reporte_txt.append("  - Pérdida (Loss): sparse_categorical_crossentropy")
    reporte_txt.append("  - Early Stopping: Activado (val_loss, patience=15)")
    reporte_txt.append("  - Pesos de clase: balanced (aplicado durante el fit)")
    reporte_txt.append("  - División del conjunto: 80% / 20% (Estratificado)")
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
    with open(OUTPUT_DIR / "reporte_mlp_colores.txt", "w", encoding="utf-8") as f:
        f.write(reporte_txt_content)
    print("  [OK] reporte_mlp_colores.txt guardado.")
    
    # E. Guardar matriz de confusión (PNG)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_mat, 
        annot=True, 
        fmt="d", 
        cmap="Oranges", 
        xticklabels=clases_unicas, 
        yticklabels=clases_unicas,
        cbar=True,
        square=True
    )
    plt.title("Matriz de Confusión — Red Neuronal MLP (Colores Papas)\nDataset de Prueba", fontsize=14, pad=20)
    plt.xlabel("Clase Predicha", fontsize=12, labelpad=10)
    plt.ylabel("Clase Real", fontsize=12, labelpad=10)
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_mlp_colores.png", dpi=300)
    plt.close()
    print("  [OK] confusion_matrix_mlp_colores.png guardado.")
    
    # F. Guardar historial de entrenamiento (PNG)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico de Pérdida
    ax1.plot(history.history['loss'], label='Entrenamiento', color='darkorange', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validación', color='steelblue', linewidth=2)
    ax1.set_title('Pérdida (Loss) por época', fontsize=12, pad=10)
    ax1.set_xlabel('Época', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de Exactitud
    ax2.plot(history.history['accuracy'], label='Entrenamiento', color='darkorange', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validación', color='steelblue', linewidth=2)
    ax2.set_title('Exactitud (Accuracy) por época', fontsize=12, pad=10)
    ax2.set_xlabel('Época', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Historial de Entrenamiento — Red Neuronal MLP", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_history_mlp_colores.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] training_history_mlp_colores.png guardado.")
    
    # G. Guardar predicciones de todo el dataset en resultados_mlp_colores.csv
    X_scaled_all = scaler.transform(X)
    y_pred_all_proba = model.predict(X_scaled_all)
    y_pred_all = np.argmax(y_pred_all_proba, axis=1)
    y_pred_all_labels = le.inverse_transform(y_pred_all)
    
    split_col = np.array(["train"] * len(X))
    split_col[X_test_idx] = "test"
    
    proba_cols = {}
    for idx, c in enumerate(clases_unicas):
        c_clean = c.replace(" - ", "_").replace("/", "_").replace(" ", "_")
        proba_cols[f"prob_{c_clean}"] = y_pred_all_proba[:, idx]
        
    resultados_dict = {
        "imagen_path": image_paths_list,
        "variedad": variedad_list,
        "color_grupo_true": y_list,
        "color_grupo_pred": y_pred_all_labels,
        "split": split_col
    }
    resultados_dict.update(proba_cols)
    
    df_resultados = pd.DataFrame(resultados_dict)
    df_resultados.to_csv(OUTPUT_DIR / "resultados_mlp_colores.csv", index=False, encoding="utf-8")
    print("  [OK] resultados_mlp_colores.csv guardado.")
    
    print("\nPROCESO DE ENTRENAMIENTO MLP FINALIZADO EXITOSAMENTE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
