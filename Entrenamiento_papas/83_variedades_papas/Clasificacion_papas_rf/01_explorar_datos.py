# -*- coding: utf-8 -*-
"""
Script 01: Exploración del Dataset (Random Forest)
Descripción: Verifica la existencia del dataset de imágenes de papas nativas,
             analiza su legibilidad e imprime estadísticas de distribución de clases.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
BASE_DIR = Path(r"C:\Users\rodri\Documents\8vo ciclo\Inteligencia Artificial\accidente_aereo_IA\Entrenamiento_papas\Clasificacion_papas_rf")
DATASET_PATH = Path(r"C:\Users\rodri\Documents\8vo ciclo\Inteligencia Artificial\accidente_aereo_IA\Entrenamiento_papas\dataset\dataset128x128")
IMG_SIZE = (128, 128)

def main():
    print("=" * 80)
    print(" INICIANDO EXPLORACIÓN DEL DATASET DE PAPAS NATIVAS (RF) ".center(80, "="))
    print("=" * 80)
    print(f"Ruta del dataset: {DATASET_PATH.resolve()}")
    
    # 1. Verificar existencia del dataset
    if not DATASET_PATH.exists():
        print(f"[ERROR] No se encontró el directorio del dataset en: {DATASET_PATH.resolve()}")
        print("Por favor, asegúrate de que el dataset se haya descargado correctamente.")
        sys.exit(1)
        
    # 2. Encontrar subcarpetas de clases (V1 a V83/V84)
    clases = sorted([d.name for d in DATASET_PATH.iterdir() if d.is_dir() and d.name != ".cache"])
    num_clases = len(clases)
    
    print(f"\n[INFO] Número de clases/variedades detectadas: {num_clases}")
    if num_clases == 0:
        print("[ERROR] No se encontraron subcarpetas de clases en el dataset.")
        sys.exit(1)
        
    # 3. Escanear legibilidad de imágenes
    conteo_por_clase = {}
    imagenes_corruptas = []
    total_imagenes = 0
    
    print("\n[INFO] Escaneando y verificando legibilidad de las imágenes...")
    
    for clase in clases:
        clase_path = DATASET_PATH / clase
        # Buscar formatos comunes de imágenes
        archivos = list(clase_path.glob("*.jpg")) + list(clase_path.glob("*.jpeg")) + list(clase_path.glob("*.png"))
        
        conteo_valido = 0
        for img_path in archivos:
            try:
                # Cargar imagen y verificarla
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    _ = img.size
                conteo_valido += 1
                total_imagenes += 1
            except Exception as e:
                imagenes_corruptas.append((img_path, str(e)))
                
        conteo_por_clase[clase] = conteo_valido
        
    # 4. Estadísticas del dataset
    df_estadisticas = pd.DataFrame(list(conteo_por_clase.items()), columns=["Clase", "Cantidad"])
    
    print("\n" + "=" * 80)
    print(" ESTADÍSTICAS DEL DATASET ".center(80, "-"))
    print("=" * 80)
    print(f"Total de imágenes legibles encontradas: {total_imagenes}")
    print(f"Imágenes corruptas o no legibles:       {len(imagenes_corruptas)}")
    
    if len(imagenes_corruptas) > 0:
        print("\nDetalle de archivos corruptos (primeros 5):")
        for i, (path, err) in enumerate(imagenes_corruptas[:5], 1):
            print(f"  {i}. {path.name} en {path.parent.name} -> {err}")
            
    min_img = df_estadisticas["Cantidad"].min()
    max_img = df_estadisticas["Cantidad"].max()
    mean_img = df_estadisticas["Cantidad"].mean()
    std_img = df_estadisticas["Cantidad"].std()
    
    print(f"\nDistribución de imágenes por variedad:")
    print(f"  - Mínimo:  {min_img}")
    print(f"  - Máximo:  {max_img}")
    print(f"  - Promedio: {mean_img:.2f}")
    print(f"  - Desviación Estándar: {std_img:.2f}")
    
    # Detección de clases desbalanceadas (menos de 20 imágenes)
    clases_desbalanceadas = df_estadisticas[df_estadisticas["Cantidad"] < 20]
    num_desbalanceadas = len(clases_desbalanceadas)
    
    print(f"\nAlerta de clases desbalanceadas (< 20 imágenes): {num_desbalanceadas} clases")
    if num_desbalanceadas > 0:
        for idx, row in clases_desbalanceadas.iterrows():
            print(f"  * Clase '{row['Clase']}': {row['Cantidad']} imágenes")
    else:
        print("  * ¡Perfecto! Todas las clases tienen un número de imágenes saludable (>= 20).")
        
    print("\n" + "=" * 80)
    print(" EXPLORACIÓN COMPLETADA ".center(80, "="))
    print("=" * 80)

if __name__ == "__main__":
    main()
