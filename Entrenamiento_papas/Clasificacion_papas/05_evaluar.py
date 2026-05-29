# -*- coding: utf-8 -*-
"""
Script 05: Evaluación Exhaustiva del Modelo
Descripción: Carga el modelo MLP entrenado, los conjuntos de prueba y el mapa
             de etiquetas. Calcula exactitud Top-1, Top-3, Top-5, genera la matriz
             de confusión, identifica las mejores y peores variedades, y escribe
             un reporte completo en 'reporte_evaluacion.txt'.
"""

import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        top_k_accuracy_score
    )
except ImportError:
    print("[ERROR] No se pudo importar scikit-learn. Ejecuta la instalación de dependencias primero.")
    sys.exit(1)

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
MODEL_PKL = Path("modelo_mlp.pkl")
INPUT_NPZ = Path("features_procesado.npz")
LABEL_MAP_PKL = Path("label_map.pkl")
REPORT_TXT = Path("reporte_evaluacion.txt")
CONF_MATRIX_PNG = Path("matriz_confusion.png")

def main():
    print("=" * 80)
    print(" INICIANDO EVALUACIÓN COMPLETA DEL MODELO MLP ".center(80, "="))
    print("=" * 80)
    
    # 1. Verificar existencia de archivos requeridos
    archivos_requeridos = [MODEL_PKL, INPUT_NPZ, LABEL_MAP_PKL]
    for archivo in archivos_requeridos:
        if not archivo.exists():
            print(f"[ERROR] No se encontró el archivo requerido: '{archivo.resolve()}'.")
            print("Por favor, ejecuta en orden '03_preprocesar.py' y '04_entrenar_mlp.py' antes de evaluar.")
            sys.exit(1)
            
    # 2. Cargar archivos
    print("[INFO] Cargando modelo, datos y mapa de etiquetas...")
    modelo = joblib.load(MODEL_PKL)
    label_map = joblib.load(LABEL_MAP_PKL)
    
    with np.load(INPUT_NPZ) as data:
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        
    print(f"[INFO] Carga completada. Muestras de prueba: {X_test.shape[0]}")
    
    # Reconstruir los nombres de clases ordenados por su ID
    sorted_ids = sorted(label_map.keys())
    label_names = [label_map[i] for i in sorted_ids]
    
    # 3. Realizar predicciones
    print("[INFO] Calculando predicciones del modelo en el conjunto de prueba...")
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)
    
    # 4. Calcular métricas globales
    top1 = accuracy_score(y_test, y_pred)
    top3 = top_k_accuracy_score(y_test, y_prob, k=3)
    top5 = top_k_accuracy_score(y_test, y_prob, k=5)
    
    print("\n" + "-" * 50)
    print(" MÉTRICAS DE EXACTITUD GLOBALES ".center(50, "-"))
    print("-" * 50)
    print(f"  - Accuracy Top-1: {top1 * 100:.2f}%")
    print(f"  - Accuracy Top-3: {top3 * 100:.2f}%")
    print(f"  - Accuracy Top-5: {top5 * 100:.2f}%")
    
    # 5. Reporte por clase
    report_dict = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=label_names)
    
    # 6. Matriz de Confusión
    print(f"\n[INFO] Generando matriz de confusión gigante ({CONF_MATRIX_PNG.name})...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=False,  # anotaciones en 83 clases son ilegibles, mejor visualización cromática
        cmap="YlOrRd",
        xticklabels=label_names,
        yticklabels=label_names
    )
    plt.title("Matriz de Confusión - Clasificación de Papas Nativas Peruanas (83 clases)", fontsize=18, fontweight='bold')
    plt.xlabel("Clase Predicha", fontsize=14)
    plt.ylabel("Clase Real", fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_PNG, dpi=150)
    plt.close()
    
    # 7. Identificar Top 10 mejores y peores variedades
    # Extraer F1-scores de cada clase
    clases_metricas = []
    for clase in label_names:
        if clase in report_dict:
            clases_metricas.append({
                "clase": clase,
                "precision": report_dict[clase]["precision"],
                "recall": report_dict[clase]["recall"],
                "f1-score": report_dict[clase]["f1-score"],
                "support": report_dict[clase]["support"]
            })
            
    df_metricas = pd.DataFrame(clases_metricas)
    
    # Ordenar por F1-Score
    df_ordenado = df_metricas.sort_values(by="f1-score", ascending=False)
    
    mejores_10 = df_ordenado.head(10)
    peores_10 = df_ordenado.tail(10).iloc[::-1]  # Invertir para que la peor quede al inicio
    
    # 8. Analizar la confusión de las peores 10
    confusion_peores = []
    for idx, row in peores_10.iterrows():
        clase_nombre = row["clase"]
        # Encontrar el ID de esta clase
        clase_id = [k for k, v in label_map.items() if v == clase_nombre][0]
        
        # Fila de la matriz de confusión correspondiente a esta clase
        fila_confusion = cm[clase_id]
        
        # Buscar el valor máximo que no sea la diagonal (predicción correcta)
        max_conf_val = -1
        max_conf_id = -1
        
        for c_id, count in enumerate(fila_confusion):
            if c_id != clase_id:
                if count > max_conf_val:
                    max_conf_val = count
                    max_conf_id = c_id
                    
        confundida_con = "Ninguna"
        if max_conf_val > 0 and max_conf_id != -1:
            confundida_con = label_map[max_conf_id]
            
        confusion_peores.append({
            "clase": clase_nombre,
            "f1_score": row["f1-score"],
            "confundida_con": confundida_con,
            "veces_confundida": max_conf_val
        })
        
    # 9. Guardar Reporte en reporte_evaluacion.txt
    print(f"[INFO] Escribiendo reporte final en: {REPORT_TXT.name}...")
    
    linea_decorativa = "=" * 80 + "\n"
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(linea_decorativa)
        f.write("=== REPORTE DE EVALUACIÓN — MLP PAPAS NATIVAS PERUANAS ===\n")
        f.write(linea_decorativa)
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Dataset: dataset128x128\n")
        f.write(f"Total imágenes de Entrenamiento: {X_train.shape[0]}\n")
        f.write(f"Total imágenes de Prueba:      {X_test.shape[0]}\n")
        f.write("Clases: 83 variedades de papas nativas\n\n")
        
        f.write(f"Exactitud Global (Top-1 Accuracy): {top1 * 100:.2f}%\n")
        f.write(f"Exactitud Global (Top-3 Accuracy): {top3 * 100:.2f}%\n")
        f.write(f"Exactitud Global (Top-5 Accuracy): {top5 * 100:.2f}%\n\n")
        
        f.write("=== REPORTE DE CLASIFICACIÓN COMPLETO ===\n")
        f.write(report_text)
        f.write("\n")
        
        f.write("=== TOP 10 VARIEDADES MEJOR CLASIFICADAS ===\n")
        for idx, row in mejores_10.iterrows():
            f.write(f"  * {row['clase']} -> F1-Score: {row['f1-score']:.4f} (Soporte: {int(row['support'])})\n")
        f.write("\n")
        
        f.write("=== TOP 10 VARIEDADES PEOR CLASIFICADAS ===\n")
        for item in confusion_peores:
            f.write(f"  * {item['clase']} -> F1-Score: {item['f1_score']:.4f}\n")
            if item['confundida_con'] != "Ninguna":
                f.write(f"    - Se confunde más con: '{item['confundida_con']}' (detectada {item['veces_confundida']} veces)\n")
            else:
                f.write("    - No presenta confusiones significativas con otras clases en el test set\n")
        f.write(linea_decorativa)
        
    print("\n" + "=" * 80)
    print(" EVALUACIÓN FINALIZADA Y REPORTE GUARDADO ".center(80, "-"))
    print("=" * 80)
    print(f"Reporte escrito en: {REPORT_TXT.resolve()}")
    print(f"Gráfico de matriz de confusión guardado en: {CONF_MATRIX_PNG.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
