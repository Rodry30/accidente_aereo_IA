# -*- coding: utf-8 -*-
"""
Script 06: Efecto del Número de Árboles (Estimadores)
Descripción: Entrena múltiples clasificadores Random Forest con diferente número de árboles
             (de 10 a 500), evalúa el desempeño en Train, Test y OOB, y grafica las
             3 curvas resultantes para analizar la estabilidad del modelo.
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Desactivar advertencias de OOB score para muy pocos estimadores
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
BASE_DIR = Path(r"C:\Users\rodri\Documents\8vo ciclo\Inteligencia Artificial\accidente_aereo_IA\Entrenamiento_papas\Clasificacion_papas_rf")
INPUT_NPZ = BASE_DIR / "features_procesado.npz"
PLOT_PATH = BASE_DIR / "curva_n_estimators.png"

def main():
    print("=" * 80)
    print(" EFECTO DEL NÚMERO DE ÁRBOLES EN EL RENDIMIENTO (RF) ".center(80, "="))
    print("=" * 80)
    
    # 1. Verificar existencia de datos
    if not INPUT_NPZ.exists():
        print(f"[ERROR] No se encontró el archivo '{INPUT_NPZ.resolve()}'.")
        print("Por favor, ejecuta primero '03_preprocesar.py'.")
        sys.exit(1)
        
    print("[INFO] Cargando matrices preprocesadas...")
    with np.load(INPUT_NPZ) as data:
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        
    print(f"[INFO] Iniciando simulación con n_estimators...")
    
    n_estimators_list = [10, 25, 50, 100, 150, 200, 300, 400, 500]
    train_scores = []
    test_scores  = []
    oob_scores   = []
    
    print("-" * 70)
    print(f"{'n_estimators':<15}{'Train Accuracy':<18}{'Test Accuracy':<18}{'OOB Score':<15}")
    print("-" * 70)
    
    for n in n_estimators_list:
        rf = RandomForestClassifier(
            n_estimators=n,
            max_features='sqrt',
            class_weight='balanced',
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        train_acc = rf.score(X_train, y_train)
        test_acc = rf.score(X_test, y_test)
        oob_acc = rf.oob_score_
        
        train_scores.append(train_acc)
        test_scores.append(test_acc)
        oob_scores.append(oob_acc)
        
        print(f"{n:<15}{train_acc:<18.4f}{test_acc:<18.4f}{oob_acc:<15.4f}")
        
    print("-" * 70)
    
    # Graficar las 3 curvas
    print(f"\n[INFO] Generando y guardando gráfico en: {PLOT_PATH.name}...")
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, train_scores, label='Train Accuracy', marker='o', color='tab:blue', linewidth=2)
    plt.plot(n_estimators_list, test_scores,  label='Test Accuracy',  marker='s', color='tab:orange', linewidth=2)
    plt.plot(n_estimators_list, oob_scores,   label='OOB Score (Out-of-Bag)', marker='^', color='tab:green', linestyle='--', linewidth=2)
    
    plt.xlabel('Número de árboles (n_estimators)', fontsize=12)
    plt.ylabel('Exactitud (Accuracy)', fontsize=12)
    plt.title('Random Forest — Efecto del número de árboles en el rendimiento', fontsize=14, fontweight='bold', pad=15)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    
    print(f"[OK] Gráfico guardado: {PLOT_PATH.resolve()}")
    print("\n" + "=" * 80)
    print(" ANÁLISIS DE ÁRBOLES FINALIZADO ".center(80, "-"))
    print("=" * 80)

if __name__ == "__main__":
    main()
