# -*- coding: utf-8 -*-
"""
Script 06: Clasificador de Papa Nativa (Inferencia)
Descripción: Carga el modelo, escalador y mapa de etiquetas entrenados.
             Recibe la ruta de una imagen de papa, extrae sus características,
             la clasifica con el modelo MLP y muestra los Top-3 resultados más probables.
"""

import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# Desactivar advertencias de skimage
warnings.filterwarnings("ignore")

try:
    import joblib
    from skimage.color import rgb2hsv
    from skimage.filters import sobel, threshold_otsu
    from skimage.feature import local_binary_pattern
    from skimage.measure import label, regionprops, shannon_entropy
except ImportError:
    print("[ERROR] Faltan dependencias indispensables. Asegúrate de ejecutar esto en el entorno virtual '.pt'.")
    sys.exit(1)

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
MODEL_PKL = Path("modelo_mlp.pkl")
SCALER_PKL = Path("scaler.pkl")
LABEL_MAP_PKL = Path("label_map.pkl")
IMG_SIZE = (128, 128)

def extraer_caracteristicas_imagen(img_path):
    """
    Extrae exactamente el mismo conjunto de 45 características tabulares
    que se utilizaron para entrenar el modelo.
    """
    # 1. Cargar la imagen y redimensionar/convertir
    with Image.open(img_path) as img:
        img_resized = img.resize(IMG_SIZE)
        img_rgb = img_resized.convert("RGB")
        img_gray = img_resized.convert("L")
        
    # Convertir a arrays de numpy en formato [0, 1]
    rgb_arr = np.array(img_rgb) / 255.0
    gray_arr = np.array(img_gray) / 255.0
    
    features = {}
    
    # R, G, B
    r_channel = rgb_arr[:, :, 0]
    g_channel = rgb_arr[:, :, 1]
    b_channel = rgb_arr[:, :, 2]
    
    features["r_mean"] = np.mean(r_channel)
    features["r_std"] = np.std(r_channel)
    features["r_median"] = np.median(r_channel)
    
    features["g_mean"] = np.mean(g_channel)
    features["g_std"] = np.std(g_channel)
    features["g_median"] = np.median(g_channel)
    
    features["b_mean"] = np.mean(b_channel)
    features["b_std"] = np.std(b_channel)
    features["b_median"] = np.median(b_channel)
    
    # HSV
    hsv_arr = rgb2hsv(rgb_arr)
    h_channel = hsv_arr[:, :, 0]
    s_channel = hsv_arr[:, :, 1]
    v_channel = hsv_arr[:, :, 2]
    
    features["h_mean"] = np.mean(h_channel)
    features["h_std"] = np.std(h_channel)
    features["s_mean"] = np.mean(s_channel)
    features["s_std"] = np.std(s_channel)
    features["v_mean"] = np.mean(v_channel)
    features["v_std"] = np.std(v_channel)
    features["v_median"] = np.median(v_channel)
    
    # Grises
    features["gray_mean"] = np.mean(gray_arr)
    features["gray_std"] = np.std(gray_arr)
    features["gray_median"] = np.median(gray_arr)
    
    # Histograma de gris (8 bins)
    hist_gray, _ = np.histogram(gray_arr, bins=8, range=(0.0, 1.0))
    hist_gray_norm = hist_gray / np.sum(hist_gray) if np.sum(hist_gray) > 0 else hist_gray
    for idx, val in enumerate(hist_gray_norm):
        features[f"hist_gray_{idx}"] = val
        
    # Sobel
    sobel_img = sobel(gray_arr)
    features["sobel_mean"] = np.mean(sobel_img)
    features["sobel_std"] = np.std(sobel_img)
    
    # Energía y Entropía
    features["energy"] = np.sum(gray_arr ** 2)
    features["entropy"] = shannon_entropy(gray_arr)
    
    # LBP (10 bins)
    lbp = local_binary_pattern(gray_arr, P=24, R=3, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 25))
    lbp_hist_norm = lbp_hist / np.sum(lbp_hist) if np.sum(lbp_hist) > 0 else lbp_hist
    for idx, val in enumerate(lbp_hist_norm):
        features[f"lbp_{idx}"] = val
        
    # Forma
    features["aspect_ratio"] = float(IMG_SIZE[0] / IMG_SIZE[1])
    
    try:
        thresh = threshold_otsu(gray_arr)
        binary = gray_arr > thresh
        # Invertir si el fondo es claro
        esquinas = [binary[0, 0], binary[0, -1], binary[-1, 0], binary[-1, -1]]
        if sum(esquinas) >= 2:
            binary = ~binary
            
        features["object_ratio"] = np.sum(binary) / binary.size
        
        labeled_img = label(binary)
        regions = regionprops(labeled_img)
        features["n_regions"] = len(regions)
        
        if len(regions) > 0:
            principal = max(regions, key=lambda r: r.area)
            area = principal.area
            perimeter = principal.perimeter
            if area > 0:
                features["compactness"] = (perimeter ** 2) / (4 * np.pi * area)
            else:
                features["compactness"] = 0.0
        else:
            features["compactness"] = 0.0
    except Exception:
        features["object_ratio"] = 0.0
        features["n_regions"] = 0
        features["compactness"] = 0.0
        
    return features

def clasificar_papa(img_path):
    # 1. Cargar modelo, escalador y mapa
    if not MODEL_PKL.exists() or not SCALER_PKL.exists() or not LABEL_MAP_PKL.exists():
        print("[ERROR] No se encontraron los archivos del modelo o preprocesamiento.")
        print("Asegúrate de haber ejecutado los scripts 03_preprocesar.py y 04_entrenar_mlp.py.")
        sys.exit(1)
        
    modelo = joblib.load(MODEL_PKL)
    scaler = joblib.load(SCALER_PKL)
    label_map = joblib.load(LABEL_MAP_PKL)
    
    img_path = Path(img_path)
    if not img_path.exists():
        print(f"[ERROR] La imagen especificada no existe: '{img_path.resolve()}'")
        return
        
    print(f"\n[INFO] Procesando imagen: {img_path.name}...")
    
    # 2. Extraer características
    try:
        features = extraer_caracteristicas_imagen(img_path)
    except Exception as e:
        print(f"[ERROR] No se pudo leer o procesar la imagen. Error: {e}")
        return
        
    # Convertir a DataFrame
    df = pd.DataFrame([features])
    
    # Asegurar el orden exacto de las características requerido por el escalador
    if hasattr(scaler, "feature_names_in_"):
        columnas_ordenadas = list(scaler.feature_names_in_)
        df = df[columnas_ordenadas]
    else:
        # Fallback si no tiene guardados los nombres (aunque scikit-learn moderno sí los guarda)
        print("[ADVERTENCIA] No se pudo recuperar el orden del scaler, usando orden por defecto.")
        
    # 3. Escalar características
    X_scaled = scaler.transform(df)
    
    # 4. Predecir
    pred_class_id = modelo.predict(X_scaled)[0]
    pred_probabilities = modelo.predict_proba(X_scaled)[0]
    
    clase_predicha = label_map[pred_class_id]
    confianza = pred_probabilities[pred_class_id]
    
    print("\n" + "=" * 50)
    print(" RESULTADO DE LA CLASIFICACIÓN ".center(50, "="))
    print("=" * 50)
    print(f"  Variedad Predicha: \033[1;32m{clase_predicha}\033[0m")
    print(f"  Confianza (Probabilidad): {confianza * 100:.2f}%")
    print("-" * 50)
    
    # Obtener los top 3 resultados
    top3_indices = np.argsort(pred_probabilities)[-3:][::-1]
    
    print("Top 3 alternativas más probables:")
    for idx, class_idx in enumerate(top3_indices, 1):
        clase_nom = label_map[class_idx]
        prob = pred_probabilities[class_idx]
        destacar = "*" if class_idx == pred_class_id else " "
        print(f"  {idx}. [{destacar}] {clase_nom}: {prob * 100:.2f}%")
    print("=" * 50 + "\n")
    
    # Intentar mostrar la imagen
    try:
        import matplotlib.pyplot as plt
        img = Image.open(img_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Predicción: {clase_predicha} ({confianza * 100:.1f}%)", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        print("[INFO] Mostrando ventana con la imagen predicha. Ciérrala para terminar.")
        plt.show()
    except Exception:
        # Si se ejecuta sin entorno gráfico, omitir visualización sin tirar error
        pass

def main():
    # Si se pasa como argumento de consola
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Modo interactivo
        print("=" * 60)
        print(" CLASIFICADOR INTERACTIVO DE PAPAS NATIVAS ".center(60, "="))
        print("=" * 60)
        img_path = input("Ingresa la ruta de la imagen de papa a clasificar: ").strip()
        # Quitar comillas si el usuario arrastró el archivo a la consola
        if img_path.startswith('"') and img_path.endswith('"'):
            img_path = img_path[1:-1]
        elif img_path.startswith("'") and img_path.endswith("'"):
            img_path = img_path[1:-1]
            
    if img_path:
        clasificar_papa(img_path)
    else:
        print("[INFO] No se proporcionó ninguna ruta. Saliendo...")

if __name__ == "__main__":
    main()
