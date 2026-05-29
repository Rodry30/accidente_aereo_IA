# -*- coding: utf-8 -*-
"""
Script 02: Extracción y Reutilización de Características (Random Forest)
Autor: Antigravity AI
Descripción: Copia el archivo 'features.csv' desde el proyecto de red neuronal
             si existe (en carpeta original o renombrada), de lo contrario extrae
             las 45 características tabulares desde cero para todas las imágenes.
"""

import sys
import shutil
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Desactivar advertencias de skimage
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
BASE_DIR = Path(r"C:\Users\rodri\Documents\8vo ciclo\Inteligencia Artificial\accidente_aereo_IA\Entrenamiento_papas\Clasificacion_papas_rf")
DATASET_PATH = Path(r"C:\Users\rodri\Documents\8vo ciclo\Inteligencia Artificial\accidente_aereo_IA\Entrenamiento_papas\dataset\dataset128x128")
IMG_SIZE = (128, 128)

# Posibles rutas del CSV del proyecto MLP (redes neuronales)
FEATURES_CSV_MLP_ORIGINAL = Path(r"C:\Users\rodri\Documents\8vo ciclo\Inteligencia Artificial\accidente_aereo_IA\Entrenamiento_papas\Clasificacion_papas\features.csv")
FEATURES_CSV_MLP_REDES = Path(r"C:\Users\rodri\Documents\8vo ciclo\Inteligencia Artificial\accidente_aereo_IA\Entrenamiento_papas\Clasificacion_papas_redes_neuronales\features.csv")

FEATURES_CSV = BASE_DIR / "features.csv"

# ==============================================================================
# INTENTAR REUTILIZAR EL ARCHIVO FEATURES.CSV EXISTENTE
# ==============================================================================
def intentar_copiar_csv():
    # Comprobar primero la ruta renombrada (redes neuronales)
    if FEATURES_CSV_MLP_REDES.exists():
        print(f"[INFO] Se detectó 'features.csv' en la carpeta renombrada de Redes Neuronales.")
        print(f"       Origen: {FEATURES_CSV_MLP_REDES}")
        shutil.copy(FEATURES_CSV_MLP_REDES, FEATURES_CSV)
        print(f"[OK] Copiado exitosamente a: {FEATURES_CSV.resolve()}")
        return True
    
    # Comprobar la ruta original
    elif FEATURES_CSV_MLP_ORIGINAL.exists():
        print(f"[INFO] Se detectó 'features.csv' en la carpeta original de Clasificacion_papas.")
        print(f"       Origen: {FEATURES_CSV_MLP_ORIGINAL}")
        shutil.copy(FEATURES_CSV_MLP_ORIGINAL, FEATURES_CSV)
        print(f"[OK] Copiado exitosamente a: {FEATURES_CSV.resolve()}")
        return True
        
    return False

# ==============================================================================
# EXTRACTOR DE CARACTERÍSTICAS COMPLETO (FALLBACK)
# ==============================================================================
def extraer_caracteristicas_imagen(img_path):
    """
    Extrae exactamente el mismo conjunto de 45 características tabulares.
    """
    from skimage.color import rgb2hsv
    from skimage.filters import sobel, threshold_otsu
    from skimage.feature import local_binary_pattern
    from skimage.measure import label, regionprops, shannon_entropy
    
    with Image.open(img_path) as img:
        img_resized = img.resize(IMG_SIZE)
        img_rgb = img_resized.convert("RGB")
        img_gray = img_resized.convert("L")
        
    rgb_arr = np.array(img_rgb) / 255.0
    gray_arr = np.array(img_gray) / 255.0
    
    features = {}
    
    # Color - RGB (9 features)
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
    
    # Color - HSV (7 features)
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
    
    # Color - Grises (3 features)
    features["gray_mean"] = np.mean(gray_arr)
    features["gray_std"] = np.std(gray_arr)
    features["gray_median"] = np.median(gray_arr)
    
    # Color - Histograma Gris (8 features)
    hist_gray, _ = np.histogram(gray_arr, bins=8, range=(0.0, 1.0))
    hist_gray_norm = hist_gray / np.sum(hist_gray) if np.sum(hist_gray) > 0 else hist_gray
    for idx, val in enumerate(hist_gray_norm):
        features[f"hist_gray_{idx}"] = val
        
    # Textura - Sobel (2 features)
    sobel_img = sobel(gray_arr)
    features["sobel_mean"] = np.mean(sobel_img)
    features["sobel_std"] = np.std(sobel_img)
    
    # Textura - Energía (1 feature)
    features["energy"] = np.sum(gray_arr ** 2)
    
    # Textura - Entropía (1 feature)
    features["entropy"] = shannon_entropy(gray_arr)
    
    # Textura - LBP (10 features)
    lbp = local_binary_pattern(gray_arr, P=24, R=3, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 25))
    lbp_hist_norm = lbp_hist / np.sum(lbp_hist) if np.sum(lbp_hist) > 0 else lbp_hist
    for idx, val in enumerate(lbp_hist_norm):
        features[f"lbp_{idx}"] = val
        
    # Forma - Aspect Ratio (1 feature)
    features["aspect_ratio"] = float(IMG_SIZE[0] / IMG_SIZE[1])
    
    # Forma - Segmentación Otsu (3 features)
    try:
        thresh = threshold_otsu(gray_arr)
        binary = gray_arr > thresh
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

def realizar_extraccion_completa():
    print("[INFO] Iniciando extracción de características desde las imágenes...")
    
    try:
        from skimage.color import rgb2hsv
    except ImportError:
        print("[ERROR] No se pudo importar scikit-image. Asegúrate de instalar requirements.txt.")
        sys.exit(1)
        
    clases = sorted([d.name for d in DATASET_PATH.iterdir() if d.is_dir() and d.name != ".cache"])
    
    lista_imagenes = []
    for label_id, clase in enumerate(clases):
        clase_path = DATASET_PATH / clase
        imagenes = list(clase_path.glob("*.jpg")) + list(clase_path.glob("*.jpeg")) + list(clase_path.glob("*.png"))
        for path in imagenes:
            lista_imagenes.append((path, clase, label_id))
            
    print(f"[INFO] Total de imágenes a procesar: {len(lista_imagenes)}")
    
    registros = []
    for filepath, label_name, label_id in tqdm(lista_imagenes, desc="Extrayendo features"):
        try:
            features = extraer_caracteristicas_imagen(filepath)
            features["label"] = label_name
            features["label_id"] = label_id
            features["filepath"] = str(filepath.resolve())
            registros.append(features)
        except Exception as e:
            print(f"\n[ADVERTENCIA] Error procesando {filepath.name}: {e}. Continuando...")
            
    if not registros:
        print("[ERROR] No se pudo extraer ninguna característica.")
        sys.exit(1)
        
    df = pd.DataFrame(registros)
    
    # Reordenar columnas
    columnas_meta = ["label", "label_id", "filepath"]
    columnas_features = [col for col in df.columns if col not in columnas_meta]
    df = df[columnas_features + columnas_meta]
    
    # Guardar en CSV
    df.to_csv(FEATURES_CSV, index=False)
    print(f"[OK] Archivo guardado con éxito en: {FEATURES_CSV.resolve()}")

def main():
    print("=" * 80)
    print(" EXTRACCIÓN Y COPIADO DE CARACTERÍSTICAS ".center(80, "="))
    print("=" * 80)
    
    # Crear la carpeta de trabajo si no existe
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Intentar copiar primero
    if intentar_copiar_csv():
        print("\n" + "=" * 80)
        print(" PROCESO COMPLETADO POR REUTILIZACIÓN ".center(80, "-"))
        print("=" * 80)
        return
        
    print("[INFO] No se pudo reutilizar 'features.csv' del proyecto MLP.")
    
    # Fallback: extracción completa
    if not DATASET_PATH.exists():
        print(f"[ERROR] El dataset no existe en: {DATASET_PATH.resolve()}")
        sys.exit(1)
        
    realizar_extraccion_completa()
    print("\n" + "=" * 80)
    print(" PROCESO COMPLETADO POR EXTRACCIÓN ".center(80, "-"))
    print("=" * 80)

if __name__ == "__main__":
    main()
