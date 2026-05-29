# -*- coding: utf-8 -*-
"""
Script 01: Exploración del Dataset de Papas Nativas
Descripción: Analiza la estructura del dataset de imágenes de papas nativas,
             verifica la legibilidad de las imágenes con Pillow y presenta estadísticas
             detalladas de la distribución de las 83 clases.
"""

import sys
import random
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
DATASET_PATH = Path("../dataset/dataset128x128")
IMG_SIZE = (128, 128)

def explorar_dataset():
    print("=" * 80)
    print(" INICIANDO EXPLORACIÓN DEL DATASET DE PAPAS NATIVAS ".center(80, "="))
    print("=" * 80)
    print(f"Ruta del dataset: {DATASET_PATH.resolve()}")
    
    # Verificar si el dataset existe
    if not DATASET_PATH.exists():
        print(f"[ERROR] No se encontró el directorio del dataset en: {DATASET_PATH.resolve()}")
        print("Por favor, asegúrate de haber ejecutado 'etl_dataset_papa.py' primero.")
        sys.exit(1)
        
    # Obtener todas las subcarpetas (variedades)
    clases = sorted([d.name for d in DATASET_PATH.iterdir() if d.is_dir() and d.name != ".cache"])
    num_clases = len(clases)
    
    print(f"\n[INFO] Número de clases/variedades detectadas: {num_clases}")
    if num_clases == 0:
        print("[ERROR] No se encontraron subcarpetas en el directorio del dataset.")
        sys.exit(1)
        
    print(f"[INFO] Variedades detectadas (primeras 5 y últimas 5):")
    if num_clases <= 10:
        print(f"       {clases}")
    else:
        print(f"       {clases[:5]} ... {clases[-5:]}")
        
    # Diccionario para almacenar la cantidad de imágenes legibles e ilegibles por clase
    conteo_por_clase = {}
    imagenes_corruptas = []
    total_imagenes = 0
    
    print("\n[INFO] Verificando legibilidad de las imágenes en cada directorio...")
    
    # Recorrer cada subcarpeta
    for clase in clases:
        clase_path = DATASET_PATH / clase
        archivos_imagen = list(clase_path.glob("*.jpg")) + list(clase_path.glob("*.jpeg")) + list(clase_path.glob("*.png"))
        
        conteo_clase_valido = 0
        for img_path in archivos_imagen:
            try:
                # Intentar abrir la imagen con Pillow y validar sus dimensiones
                with Image.open(img_path) as img:
                    img.verify()  # Verifica que la imagen no esté rota
                
                # Intentar cargarla realmente para asegurar legibilidad
                with Image.open(img_path) as img:
                    _ = img.size
                conteo_clase_valido += 1
                total_imagenes += 1
            except Exception as e:
                imagenes_corruptas.append((img_path, str(e)))
                print(f"[ADVERTENCIA] Imagen corrupta detectada: {img_path.name} en clase {clase}. Error: {e}")
                
        conteo_por_clase[clase] = conteo_clase_valido
        
    # Crear un DataFrame para análisis estadístico
    df_estadisticas = pd.DataFrame(list(conteo_por_clase.items()), columns=["Clase", "Cantidad"])
    
    print("\n" + "=" * 80)
    print(" ESTADÍSTICAS DEL DATASET ".center(80, "-"))
    print("=" * 80)
    print(f"Total de imágenes legibles encontradas: {total_imagenes}")
    print(f"Imágenes corruptas/ilegibles detectadas: {len(imagenes_corruptas)}")
    
    if len(imagenes_corruptas) > 0:
        print("\nDetalle de archivos corruptos:")
        for idx, (ruta, err) in enumerate(imagenes_corruptas[:10], 1):
            print(f"  {idx}. {ruta} -> {err}")
        if len(imagenes_corruptas) > 10:
            print(f"  ... y {len(imagenes_corruptas) - 10} más.")
            
    # Estadísticas descriptivas de distribución
    min_img = df_estadisticas["Cantidad"].min()
    max_img = df_estadisticas["Cantidad"].max()
    mean_img = df_estadisticas["Cantidad"].mean()
    std_img = df_estadisticas["Cantidad"].std()
    
    print(f"\nImágenes por clase:")
    print(f"  - Mínimo:  {min_img}")
    print(f"  - Máximo:  {max_img}")
    print(f"  - Promedio: {mean_img:.2f}")
    print(f"  - Desviación Estándar (std): {std_img:.2f}")
    
    # Clases desbalanceadas (menos de 20 imágenes)
    clases_desbalanceadas = df_estadisticas[df_estadisticas["Cantidad"] < 20]
    num_desbalanceadas = len(clases_desbalanceadas)
    
    print(f"\nDetección de clases desbalanceadas (< 20 imágenes): {num_desbalanceadas} clases")
    if num_desbalanceadas > 0:
        for index, row in clases_desbalanceadas.iterrows():
            print(f"  - Clase '{row['Clase']}': {row['Cantidad']} imágenes")
    else:
        print("  - ¡Excelente! Todas las clases tienen 20 o más imágenes.")
        
    print("\n" + "=" * 80)
    
    # Preguntar al usuario de manera interactiva si desea ver visualizaciones
    try:
        respuesta = input("¿Deseas mostrar 5 imágenes de ejemplo de 3 variedades distintas usando matplotlib? (s/n): ").strip().lower()
        if respuesta in ['s', 'si', 'y', 'yes', 'sí']:
            mostrar_ejemplos(clases)
    except Exception as e:
        print(f"\n[INFO] No se pudo leer la entrada del usuario ({e}). Saltando visualización interactiva.")

def mostrar_ejemplos(clases):
    # Seleccionar 3 clases aleatorias
    clases_seleccionadas = random.sample(clases, min(3, len(clases)))
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle("Muestras de Variedades de Papas Nativas", fontsize=16, fontweight='bold', color='#4a3b32')
    
    for i, clase in enumerate(clases_seleccionadas):
        clase_path = DATASET_PATH / clase
        imagenes = list(clase_path.glob("*.jpg")) + list(clase_path.glob("*.png"))
        
        # Seleccionar 5 imágenes aleatorias de la clase
        muestras = random.sample(imagenes, min(5, len(imagenes)))
        
        for j in range(5):
            ax = axes[i, j]
            if j < len(muestras):
                img_path = muestras[j]
                try:
                    img = Image.open(img_path)
                    ax.imshow(img)
                    if j == 2:
                        ax.set_title(f"Variedad: {clase}", fontsize=12, fontweight='bold', color='#2e2520')
                except Exception:
                    ax.text(0.5, 0.5, "Error al cargar", ha='center', va='center')
            ax.axis('off')
            
    plt.tight_layout()
    print("\n[INFO] Mostrando ventana con ejemplos de imágenes. Ciérrala para terminar.")
    plt.show()

if __name__ == "__main__":
    explorar_dataset()
