# -*- coding: utf-8 -*-
"""
Script 05: Evaluación Exhaustiva del Modelo (Random Forest)
Descripción: Carga el modelo, conjunto de test y mapas de etiquetas.
             Calcula exactitud Top-1, Top-3, Top-5 y OOB. Genera la matriz
             de confusión, identifica las mejores y peores variedades con sus
             confusiones, extrae la importancia de características y consolida
             el reporte final en 'reporte_evaluacion_rf.txt'.
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
BASE_DIR = Path(r"C:\Users\rodri\Documents\8vo ciclo\Inteligencia Artificial\accidente_aereo_IA\Entrenamiento_papas\Clasificacion_papas_rf")
MODEL_PKL = BASE_DIR / "modelo_rf.pkl"
INPUT_NPZ = BASE_DIR / "features_procesado.npz"
LABEL_MAP_PKL = BASE_DIR / "label_map.pkl"
SCALER_PKL = BASE_DIR / "scaler.pkl"
REPORT_TXT = BASE_DIR / "reporte_evaluacion_rf.txt"
CONF_MATRIX_PNG = BASE_DIR / "matriz_confusion_rf.png"

def main():
    print("=" * 80)
    print(" INICIANDO EVALUACIÓN COMPLETA - RANDOM FOREST ".center(80, "="))
    print("=" * 80)
    
    # 1. Verificar existencia de archivos requeridos
    archivos = [MODEL_PKL, INPUT_NPZ, LABEL_MAP_PKL, SCALER_PKL]
    for archivo in archivos:
        if not archivo.exists():
            print(f"[ERROR] No se encontró el archivo requerido: '{archivo.resolve()}'.")
            print("Por favor, ejecuta en orden los scripts 03_preprocesar.py y 04_entrenar_rf.py.")
            sys.exit(1)
            
    # 2. Cargar archivos
    print("[INFO] Cargando modelo, matrices escaladas, scaler y mapas de etiquetas...")
    modelo = joblib.load(MODEL_PKL)
    label_map = joblib.load(LABEL_MAP_PKL)
    scaler = joblib.load(SCALER_PKL)
    
    with np.load(INPUT_NPZ) as data:
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        
    print(f"[INFO] Conjunto de prueba cargado con éxito. Muestras de prueba: {X_test.shape[0]}")
    
    # Reconstruir nombres de variedades ordenadas por ID
    sorted_ids = sorted(label_map.keys())
    label_names = [label_map[i] for i in sorted_ids]
    
    # 3. Realizar predicciones
    print("[INFO] Realizando inferencia en el conjunto de prueba...")
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)
    
    # 4. Calcular métricas de exactitud globales
    top1 = accuracy_score(y_test, y_pred)
    top3 = top_k_accuracy_score(y_test, y_prob, k=3)
    top5 = top_k_accuracy_score(y_test, y_prob, k=5)
    oob = modelo.oob_score_ if hasattr(modelo, "oob_score_") else 0.0
    
    print("\n" + "-" * 50)
    print(" METRICAS DE EXACTITUD - RANDOM FOREST ".center(50, "-"))
    print("-" * 50)
    print(f"  - OOB Score:      {oob * 100:.2f}%")
    print(f"  - Accuracy Top-1: {top1 * 100:.2f}%")
    print(f"  - Accuracy Top-3: {top3 * 100:.2f}%")
    print(f"  - Accuracy Top-5: {top5 * 100:.2f}%")
    
    # 5. Generar reporte detallado de clasificación
    report_dict = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=label_names)
    
    # 6. Matriz de confusión
    print(f"\n[INFO] Generando y guardando matriz de confusión gigante ({CONF_MATRIX_PNG.name})...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names
    )
    plt.title("Matriz de Confusión - Random Forest (83 clases)", fontsize=18, fontweight='bold')
    plt.xlabel("Variedad Predicha", fontsize=14)
    plt.ylabel("Variedad Real", fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_PNG, dpi=150)
    plt.close()
    
    # 7. Identificar las mejores 10 y peores 10 variedades
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
    df_ordenado = df_metricas.sort_values(by="f1-score", ascending=False)
    
    mejores_10 = df_ordenado.head(10)
    peores_10 = df_ordenado.tail(10).iloc[::-1]
    
    # Analizar confusión de las peores
    confusion_peores = []
    for idx, row in peores_10.iterrows():
        clase_nombre = row["clase"]
        clase_id = [k for k, v in label_map.items() if v == clase_nombre][0]
        
        fila_confusion = cm[clase_id]
        
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
        
    # 8. Extraer Importancia de Características (Top 20)
    importancias = modelo.feature_importances_
    feature_names = list(scaler.feature_names_in_)
    df_importancia = pd.DataFrame({
        "Feature": feature_names,
        "Importancia": importancias
    }).sort_values(by="Importancia", ascending=False)
    top20 = df_importancia.head(20)
    
    # 9. Escribir reporte en reporte_evaluacion_rf.txt
    print(f"[INFO] Escribiendo reporte consolidado en: {REPORT_TXT.name}...")
    
    linea_decorativa = "=" * 80 + "\n"
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(linea_decorativa)
        f.write("=== REPORTE DE EVALUACIÓN — RANDOM FOREST PAPAS NATIVAS PERUANAS ===\n")
        f.write(linea_decorativa)
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Dataset: dataset128x128\n")
        f.write(f"Total imágenes de Entrenamiento: {X_train.shape[0]}\n")
        f.write(f"Total imágenes de Prueba:      {X_test.shape[0]}\n")
        f.write("Clases: 83 variedades de papas nativas\n")
        f.write(f"Hiperparámetros del Modelo: {modelo.get_params(deep=False)}\n\n")
        
        f.write(f"OOB Score (Out-of-Bag Score):      {oob * 100:.2f}%\n")
        f.write(f"Exactitud Global (Top-1 Accuracy): {top1 * 100:.2f}%\n")
        f.write(f"Exactitud Global (Top-3 Accuracy): {top3 * 100:.2f}%\n")
        f.write(f"Exactitud Global (Top-5 Accuracy): {top5 * 100:.2f}%\n\n")
        
        f.write("=== REPORTE DE CLASIFICACIÓN COMPLETO ===\n")
        f.write(report_text)
        f.write("\n")
        
        f.write("=== TOP 20 CARACTERÍSTICAS MÁS IMPORTANTES (RANDOM FOREST) ===\n")
        for pos, (idx, row) in enumerate(top20.iterrows(), 1):
            f.write(f"  {pos:<3}. {row['Feature']:<30} -> Importancia: {row['Importancia'] * 100:.4f}%\n")
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
