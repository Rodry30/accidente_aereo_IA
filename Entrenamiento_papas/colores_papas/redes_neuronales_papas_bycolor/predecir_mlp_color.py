#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de Prueba: Predicción de Color de Papa usando Red Neuronal MLP
------------------------------------------------------------------
Objetivo:
    Cargar el modelo MLP (.keras) entrenado y clasificar el color de una 
    papa ingresando la ruta de la imagen.

Uso:
    python predecir_mlp_color.py [ruta_de_la_imagen]
    O simplemente ejecutar y seguir las instrucciones en pantalla.
"""

import os
import sys
import pickle
import warnings
from pathlib import Path

import cv2
import numpy as np

# Configurar logs de TensorFlow silenciosos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Ignorar advertencias
warnings.filterwarnings("ignore")

# Configuración de Rutas Relativas
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "modelo_mlp_colores.keras"
SCALER_PATH = SCRIPT_DIR / "scaler_colores.pkl"
LABEL_ENCODER_PATH = SCRIPT_DIR / "label_encoder_colores.pkl"


def extraer_caracteristicas_imagen(img_path: Path) -> np.ndarray:
    """
    Extrae las mismas 22 características visuales tabulares del script de entrenamiento:
      - RGB promedio (3)
      - RGB desviación estándar (3)
      - HSV promedio (3)
      - Varianza de la intensidad de gris (1)
      - Magnitud promedio de bordes de Sobel (1)
      - Entropía de Shannon (1)
      - Histograma de intensidad en escala de grises (10 bins)
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en: {img_path}")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Máscara para ignorar fondo blanco (RGB > 240)
    mask = ~((img_rgb[:, :, 0] > 240) & 
             (img_rgb[:, :, 1] > 240) & 
             (img_rgb[:, :, 2] > 240))
    
    if not np.any(mask):
        mask = np.ones(img_rgb.shape[:2], dtype=bool)
        
    pixels_rgb = img_rgb[mask]
    if len(pixels_rgb) == 0:
        raise ValueError("La imagen está vacía o solo contiene fondo blanco.")
        
    # 1. Promedio RGB
    rgb_mean = pixels_rgb.mean(axis=0)
    
    # 2. Desviación Estándar RGB
    rgb_std = pixels_rgb.std(axis=0)
    
    # 3. Promedio HSV
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    pixels_hsv = img_hsv[mask]
    hsv_mean = pixels_hsv.mean(axis=0)
    
    # 4. Escala de grises para histograma, textura y entropía
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
    
    # Vector de 22 dimensiones
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


def main():
    print("=" * 70)
    print("  PREDICTOR DE COLOR DE PAPA (RED NEURONAL MLP)  ")
    print("=" * 70)
    
    # Verificar que los archivos del modelo existen
    if not (MODEL_PATH.exists() and SCALER_PATH.exists() and LABEL_ENCODER_PATH.exists()):
        print("[ERROR] No se encontraron los archivos del modelo entrenado en este directorio.")
        print(f"Asegúrese de haber ejecutado 'entrenar_redes_colores.py' primero.")
        sys.exit(1)
        
    # Obtener la ruta de la imagen
    if len(sys.argv) > 1:
        img_path_str = sys.argv[1]
    else:
        img_path_str = input("Ingrese la ruta de la imagen (.jpg, .png): ").strip()
        
    img_path = Path(img_path_str)
    if not img_path.exists():
        print(f"[ERROR] La ruta especificada no existe: '{img_path}'")
        sys.exit(1)
        
    print(f"\n[OK] Imagen encontrada: {img_path.name}")
    print("Procesando imagen y extrayendo características...")
    
    try:
        # Extraer características (22 dimensiones)
        features = extraer_caracteristicas_imagen(img_path)
        
        # Cargar los componentes del modelo
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(LABEL_ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
            
        # Normalizar características (espera 2D array)
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Predicción
        probabilities = model.predict(features_scaled, verbose=0)[0]
        pred_idx = np.argmax(probabilities)
        pred_label = le.inverse_transform([pred_idx])[0]
        
        # Imprimir resultados
        print("\n" + "=" * 50)
        print("                 RESULTADO DE LA PREDICCIÓN                ")
        print("=" * 50)
        print(f" Clase Predicha:  {pred_label.upper()}")
        print("-" * 50)
        print(" Distribución de Probabilidades:")
        for idx, label in enumerate(le.classes_):
            prob = probabilities[idx]
            marker = "  ==> " if label == pred_label else "      "
            print(f"{marker}{label:<30}: {prob:.2%} ({prob:.4f})")
        print("=" * 50)
        
    except Exception as e:
        print(f"[ERROR] Ocurrió un fallo durante la predicción: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
