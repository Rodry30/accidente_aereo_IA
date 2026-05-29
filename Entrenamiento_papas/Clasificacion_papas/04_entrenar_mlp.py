# -*- coding: utf-8 -*-
"""
Script 04: Entrenamiento de la Red Neuronal MLP
Descripción: Carga las características preprocesadas, inicializa un modelo MLP
             con parada temprana, entrena la red neuronal mostrando su progreso,
             guarda el modelo final y grafica la curva de pérdida de entrenamiento.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.neural_network import MLPClassifier

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
INPUT_NPZ = Path("features_procesado.npz")
MODEL_PKL = Path("modelo_mlp.pkl")
PLOT_PATH = Path("curva_perdida.png")

def main():
    print("=" * 80)
    print(" INICIANDO ENTRENAMIENTO DE RED NEURONAL MLP ".center(80, "="))
    print("=" * 80)
    
    # 1. Verificar existencia del archivo preprocesado
    if not INPUT_NPZ.exists():
        print(f"[ERROR] No se encontró el archivo de datos procesados '{INPUT_NPZ.resolve()}'.")
        print("Por favor, ejecuta primero '03_preprocesar.py' para procesar los datos.")
        sys.exit(1)
        
    print(f"[INFO] Cargando matrices de datos desde: {INPUT_NPZ.name}...")
    with np.load(INPUT_NPZ) as data:
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        
    print(f"[INFO] Matrices cargadas con éxito:")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_test shape:  {X_test.shape}")
    print(f"  - y_train shape: {y_train.shape}")
    print(f"  - y_test shape:  {y_test.shape}")
    
    # 2. Inicializar el clasificador MLP
    print("\n[INFO] Inicializando MLPClassifier con la estructura (256, 128, 64)...")
    modelo = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),  # 3 capas ocultas
        activation='relu',
        solver='adam',
        alpha=0.0001,               # regularización L2
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,        # detener si no mejora la validación
        validation_fraction=0.1,    # 10% de train para validación interna
        n_iter_no_change=15,        # paciencia
        random_state=42,
        verbose=True                # mostrar pérdida por época en consola
    )
    
    # 3. Entrenar el modelo
    print("\n[INFO] Iniciando ajuste del modelo (fit)...")
    print("-" * 50)
    modelo.fit(X_train, y_train)
    print("-" * 50)
    
    # 4. Guardar modelo entrenado
    print(f"\n[INFO] Guardando modelo en: {MODEL_PKL.name}...")
    joblib.dump(modelo, MODEL_PKL)
    print(f"[INFO] Modelo guardado. Épocas/Iteraciones entrenadas: {modelo.n_iter_}")
    
    # 5. Calcular accuracies finales
    print("\n[INFO] Evaluando exactitudes (Accuracy) rápidas...")
    acc_train = modelo.score(X_train, y_train)
    acc_test = modelo.score(X_test, y_test)
    print(f"  - Accuracy en Train: {acc_train * 100:.2f}%")
    print(f"  - Accuracy en Test:  {acc_test * 100:.2f}%")
    
    # 6. Graficar curvas de aprendizaje (Loss y Validación)
    print(f"\n[INFO] Generando gráfico de curvas de aprendizaje en: {PLOT_PATH.name}...")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Graficar pérdida de entrenamiento (eje Y izquierdo)
    color = 'tab:red'
    ax1.set_xlabel('Épocas / Iteraciones', fontsize=12)
    ax1.set_ylabel('Pérdida de Entrenamiento (Loss)', color=color, fontsize=12)
    line1 = ax1.plot(modelo.loss_curve_, color=color, label='Pérdida (Train Loss)', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Graficar puntaje de validación en el eje Y derecho si early_stopping está activo
    lines = line1
    if hasattr(modelo, "validation_scores_") and modelo.validation_scores_ is not None:
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy de Validación (10% Split)', color=color, fontsize=12)
        line2 = ax2.plot(modelo.validation_scores_, color=color, label='Acc. Validación', linewidth=2, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        lines = line1 + line2
        
    # Leyendas combinadas
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='upper right', fontsize=10)
    
    plt.title('Curvas de Aprendizaje de la Red Neuronal MLP - Papas Nativas', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    
    print(f"[INFO] Gráfico guardado exitosamente.")
    print("=" * 80)
    print(" ENTRENAMIENTO FINALIZADO ".center(80, "-"))
    print("=" * 80)

if __name__ == "__main__":
    main()
