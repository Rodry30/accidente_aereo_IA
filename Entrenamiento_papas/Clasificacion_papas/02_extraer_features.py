# -*- coding: utf-8 -*-
"""
Script 02: Extracción de Características Tabulares
Descripción: Lee todas las imágenes del dataset, calcula 27 características de color,
             14 de textura (incluyendo Sobel y LBP) y 4 de forma para cada una,
             y guarda la matriz resultante en 'features.csv'.
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Desactivar advertencias innecesarias de skimage
warnings.filterwarnings("ignore")

try:
    from skimage.color import rgb2hsv
    from skimage.filters import sobel, threshold_otsu
    from skimage.feature import local_binary_pattern
    from skimage.measure import label, regionprops, shannon_entropy
except ImportError:
    print("[ERROR] No se pudieron importar las librerías de scikit-image.")
    print("Asegúrate de que la instalación de dependencias en el entorno virtual haya finalizado.")
    sys.exit(1)

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
DATASET_PATH = Path("../dataset/dataset128x128")
OUTPUT_CSV = Path("features.csv")
IMG_SIZE = (128, 128)

def extraer_caracteristicas_imagen(img_path):
    """
    Extrae características de color, textura y forma de una sola imagen.
    """
    # 1. Cargar la imagen y redimensionar/convertir
    with Image.open(img_path) as img:
        img_resized = img.resize(IMG_SIZE)
        img_rgb = img_resized.convert("RGB")
        img_gray = img_resized.convert("L")
        
    # Convertir a arrays de numpy en formato de punto flotante [0, 1]
    rgb_arr = np.array(img_rgb) / 255.0
    gray_arr = np.array(img_gray) / 255.0
    
    # --------------------------------------------------------------------------
    # A. FEATURES DE COLOR (27 features)
    # --------------------------------------------------------------------------
    features = {}
    
    # Canales R, G, B
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
    
    # Canales HSV (convertido usando skimage)
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
    features["v_median"] = np.median(v_channel) # Para completar 27 features de color
    
    # Canal de Grises
    features["gray_mean"] = np.mean(gray_arr)
    features["gray_std"] = np.std(gray_arr)
    features["gray_median"] = np.median(gray_arr) # Para completar 27 features de color
    
    # Histograma de gris (8 bins normalizados)
    hist_gray, _ = np.histogram(gray_arr, bins=8, range=(0.0, 1.0))
    hist_gray_norm = hist_gray / np.sum(hist_gray) if np.sum(hist_gray) > 0 else hist_gray
    for idx, val in enumerate(hist_gray_norm):
        features[f"hist_gray_{idx}"] = val
        
    # --------------------------------------------------------------------------
    # B. FEATURES DE TEXTURA (14 features)
    # --------------------------------------------------------------------------
    # Contraste local con filtro de Sobel
    sobel_img = sobel(gray_arr)
    features["sobel_mean"] = np.mean(sobel_img)
    features["sobel_std"] = np.std(sobel_img)
    
    # Energía de la imagen (suma de cuadrados de píxeles normalizados)
    features["energy"] = np.sum(gray_arr ** 2)
    
    # Entropía de Shannon
    features["entropy"] = shannon_entropy(gray_arr)
    
    # Local Binary Pattern (LBP) - radio=3, puntos=24, uniform
    lbp = local_binary_pattern(gray_arr, P=24, R=3, method="uniform")
    # Histograma de 10 bins sobre el resultado del LBP
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 25))
    lbp_hist_norm = lbp_hist / np.sum(lbp_hist) if np.sum(lbp_hist) > 0 else lbp_hist
    for idx, val in enumerate(lbp_hist_norm):
        features[f"lbp_{idx}"] = val
        
    # --------------------------------------------------------------------------
    # C. FEATURES DE FORMA (4 features)
    # --------------------------------------------------------------------------
    # Aspect ratio
    features["aspect_ratio"] = float(IMG_SIZE[0] / IMG_SIZE[1])
    
    # Umbralización con Otsu y relación de área objeto/fondo
    try:
        thresh = threshold_otsu(gray_arr)
        binary = gray_arr > thresh
        # Generalmente la papa es el objeto en primer plano, pero si el fondo es más claro, invertimos
        # Si la mayoría de las esquinas son claras, asumimos que el fondo es claro y la papa es oscura
        esquinas = [binary[0, 0], binary[0, -1], binary[-1, 0], binary[-1, -1]]
        if sum(esquinas) >= 2:
            binary = ~binary  # Invertir para que la papa sea True
            
        features["object_ratio"] = np.sum(binary) / binary.size
        
        # Regiones conectadas y compacidad
        labeled_img = label(binary)
        regions = regionprops(labeled_img)
        
        features["n_regions"] = len(regions)
        
        if len(regions) > 0:
            # Seleccionar la región de mayor área como la papa principal
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
        # Fallback en caso de fallar Otsu o regionprops (por ejemplo, imágenes uniformes)
        features["object_ratio"] = 0.0
        features["n_regions"] = 0
        features["compactness"] = 0.0
        
    return features

def main():
    print("=" * 80)
    print(" INICIANDO EXTRACCIÓN DE FEATURES TABULARES DE PAPAS ".center(80, "="))
    print("=" * 80)
    
    if not DATASET_PATH.exists():
        print(f"[ERROR] El directorio del dataset no existe en: {DATASET_PATH.resolve()}")
        sys.exit(1)
        
    # Obtener todas las variedades (subcarpetas)
    clases = sorted([d.name for d in DATASET_PATH.iterdir() if d.is_dir() and d.name != ".cache"])
    
    # Recolectar todas las rutas de imágenes y sus correspondientes etiquetas
    lista_imagenes = []
    for label_id, clase in enumerate(clases):
        clase_path = DATASET_PATH / clase
        imagenes_clase = list(clase_path.glob("*.jpg")) + list(clase_path.glob("*.jpeg")) + list(clase_path.glob("*.png"))
        for path in imagenes_clase:
            lista_imagenes.append((path, clase, label_id))
            
    print(f"[INFO] Total de imágenes a procesar: {len(lista_imagenes)}")
    
    # Lista para almacenar las filas del dataset
    registros = []
    
    # Procesar con tqdm
    for filepath, label_name, label_id in tqdm(lista_imagenes, desc="Extrayendo características"):
        try:
            features = extraer_caracteristicas_imagen(filepath)
            
            # Agregar metadatos obligatorios
            features["label"] = label_name
            features["label_id"] = label_id
            features["filepath"] = str(filepath.resolve())
            
            registros.append(features)
        except Exception as e:
            print(f"\n[ADVERTENCIA] Error procesando {filepath.name}: {e}. Continuando...")
            
    if not registros:
        print("[ERROR] No se pudo extraer ninguna característica del dataset.")
        sys.exit(1)
        
    # Crear DataFrame y guardar a CSV
    df = pd.DataFrame(registros)
    
    # Reordenar columnas para que los metadatos queden al final
    columnas_metadatos = ["label", "label_id", "filepath"]
    columnas_features = [col for col in df.columns if col not in columnas_metadatos]
    df = df[columnas_features + columnas_metadatos]
    
    # Guardar en CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "=" * 80)
    print(" EXTRACCIÓN FINALIZADA ".center(80, "-"))
    print("=" * 80)
    print(f"Archivo guardado en: {OUTPUT_CSV.resolve()}")
    print(f"Total de imágenes procesadas con éxito: {len(df)}")
    print(f"Total de características numéricas extraídas: {len(columnas_features)}")
    print(f"Dimensiones de la matriz de características (DataFrame shape): {df.shape}")
    print("=" * 80)

if __name__ == "__main__":
    main()
