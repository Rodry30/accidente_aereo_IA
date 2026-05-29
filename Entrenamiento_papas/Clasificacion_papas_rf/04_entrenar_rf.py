# -*- coding: utf-8 -*-
"""
Script 04: Entrenamiento y Optimización de Random Forest
Autor: Antigravity AI
Descripción: Entrena un modelo base de Random Forest, realiza una búsqueda de
             hiperparámetros óptimos mediante RandomizedSearchCV, calcula la importancia
             de características, guarda el mejor modelo y grafica el Top 20 de importancia.
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Desactivar advertencias de scikit-learn
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
BASE_DIR = Path(r"C:\Users\rodri\Documents\8vo ciclo\Inteligencia Artificial\accidente_aereo_IA\Entrenamiento_papas\Clasificacion_papas_rf")
INPUT_NPZ = BASE_DIR / "features_procesado.npz"
SCALER_PKL = BASE_DIR / "scaler.pkl"
MODEL_PKL = BASE_DIR / "modelo_rf.pkl"
PLOT_PATH = BASE_DIR / "importancia_features.png"

def main():
    print("=" * 80)
    print(" INICIANDO ENTRENAMIENTO DE RANDOM FOREST ".center(80, "="))
    print("=" * 80)
    
    # 1. Verificar existencia de datos de entrada
    if not INPUT_NPZ.exists() or not SCALER_PKL.exists():
        print(f"[ERROR] No se encontraron los archivos procesados en '{BASE_DIR.resolve()}'.")
        print("Por favor, ejecuta primero '03_preprocesar.py'.")
        sys.exit(1)
        
    print("[INFO] Cargando matrices de entrenamiento...")
    with np.load(INPUT_NPZ) as data:
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        
    scaler = joblib.load(SCALER_PKL)
    feature_names = list(scaler.feature_names_in_)
    
    print(f"[INFO] Datos cargados:")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_test shape:  {X_test.shape}")
    print(f"  - Número de características: {len(feature_names)}")
    
    # 2. Modelo Base para comparación
    print("\n" + "-" * 50)
    print(" 2.1 ENTRENANDO MODELO RANDOM FOREST BASE ".center(50, "-"))
    print("-" * 50)
    print("[INFO] Ajustando modelo base de 300 árboles (esto puede tardar unos segundos)...")
    
    modelo_base = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    modelo_base.fit(X_train, y_train)
    
    oob_base = modelo_base.oob_score_
    acc_train_base = modelo_base.score(X_train, y_train)
    acc_test_base = modelo_base.score(X_test, y_test)
    
    print(f"[BASE] OOB Score (Bosque Base):        {oob_base * 100:.2f}%")
    print(f"[BASE] Accuracy en Train (Bosque Base): {acc_train_base * 100:.2f}%")
    print(f"[BASE] Accuracy en Test (Bosque Base):  {acc_test_base * 100:.2f}%")
    
    # 3. Búsqueda de Hiperparámetros con RandomizedSearchCV
    print("\n" + "-" * 50)
    print(" 2.2 BÚSQUEDA DE HIPERPARÁMETROS OPTIMIZADOS ".center(50, "-"))
    print("-" * 50)
    print("[INFO] Iniciando RandomizedSearchCV con 20 iteraciones y CV de 3 folds...")
    
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Se agrega oob_score=True por defecto para estimar la generalización interna en el mejor modelo
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced', oob_score=True),
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='accuracy',
        random_state=42,
        verbose=2,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    
    print(f"\n[INFO] Mejores hiperparámetros encontrados:")
    for param, val in search.best_params_.items():
        print(f"  - {param}: {val}")
    print(f"[INFO] Mejor Accuracy en Validación Cruzada (CV): {search.best_score_ * 100:.2f}%")
    
    modelo_final = search.best_estimator_
    
    # 4. Guardar Modelo Final Optimizado
    print(f"\n[INFO] Guardando el modelo final optimizado en: {MODEL_PKL.name}...")
    joblib.dump(modelo_final, MODEL_PKL)
    print(f"[INFO] Modelo final guardado con éxito.")
    
    # Métricas del modelo optimizado
    oob_final = modelo_final.oob_score_
    acc_train_final = modelo_final.score(X_train, y_train)
    acc_test_final = modelo_final.score(X_test, y_test)
    
    print("\n" + "-" * 50)
    print(" RENDIMIENTO FINAL DEL MODELO OPTIMIZADO ".center(50, "-"))
    print("-" * 50)
    print(f"  - OOB Score:         {oob_final * 100:.2f}%")
    print(f"  - Accuracy en Train: {acc_train_final * 100:.2f}%")
    print(f"  - Accuracy en Test:  {acc_test_final * 100:.2f}%")
    print("-" * 50)
    
    # 5. Análisis de Importancia de Características (Feature Importance)
    print("\n[INFO] Calculando importancia de características...")
    importancias = modelo_final.feature_importances_
    
    df_importancia = pd.DataFrame({
        "Feature": feature_names,
        "Importancia": importancias
    }).sort_values(by="Importancia", ascending=False)
    
    # Imprimir top 20
    top20 = df_importancia.head(20)
    print("\n" + " TOP 20 CARACTERÍSTICAS MÁS IMPORTANTES ".center(60, "-"))
    print(f"{'Posición':<10}{'Característica':<35}{'Importancia (%)':<15}")
    print("-" * 60)
    for pos, (idx, row) in enumerate(top20.iterrows(), 1):
        print(f"{pos:<10}{row['Feature']:<35}{row['Importancia'] * 100:<15.4f}")
    print("-" * 60)
    
    # Graficar la importancia
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x="Importancia",
        y="Feature",
        data=top20,
        palette="crest",
        hue="Feature",
        legend=False
    )
    plt.title("Top 20 Características Más Importantes - Random Forest", fontsize=14, fontweight='bold')
    plt.xlabel("Importancia Relativa", fontsize=12)
    plt.ylabel("Característica (Feature)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"[INFO] Gráfico de importancia guardado en: {PLOT_PATH.resolve()}")
    
    print("\n" + "=" * 80)
    print(" ENTRENAMIENTO FINALIZADO CON ÉXITO ".center(80, "-"))
    print("=" * 80)

if __name__ == "__main__":
    main()
