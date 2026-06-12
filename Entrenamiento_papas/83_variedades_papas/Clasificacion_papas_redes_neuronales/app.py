# -*- coding: utf-8 -*-
"""
Gradio Web App - Clasificación de Papas Nativas Peruanas
Descripción: Carga el modelo MLP, escalador y mapa de etiquetas entrenados localmente.
             Proporciona una interfaz web premium con Gradio Blocks para subir una foto
             de papa nativa andina y clasificarla instantáneamente en tiempo real.
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
    import gradio as gr
    from skimage.color import rgb2hsv
    from skimage.filters import sobel, threshold_otsu
    from skimage.feature import local_binary_pattern
    from skimage.measure import label, regionprops, shannon_entropy
except ImportError:
    print("[ERROR] Faltan dependencias indispensables. Asegúrate de ejecutar esto en el entorno virtual '.pt'.")
    sys.exit(1)

# ==============================================================================
# CONFIGURACIÓN DE RUTA Y MODELO
# ==============================================================================
BASE_DIR = Path(r"C:\Users\rodri\Documents\8vo ciclo\Inteligencia Artificial\accidente_aereo_IA\Entrenamiento_papas\Clasificacion_papas_redes_neuronales")
MODEL_PKL = BASE_DIR / "modelo_mlp.pkl"
SCALER_PKL = BASE_DIR / "scaler.pkl"
LABEL_MAP_PKL = BASE_DIR / "label_map.pkl"
IMG_SIZE = (128, 128)

modelo = None
scaler = None
label_map = None
model_loaded = False

def cargar_pipeline():
    global modelo, scaler, label_map, model_loaded
    if MODEL_PKL.exists() and SCALER_PKL.exists() and LABEL_MAP_PKL.exists():
        try:
            modelo = joblib.load(MODEL_PKL)
            scaler = joblib.load(SCALER_PKL)
            label_map = joblib.load(LABEL_MAP_PKL)
            model_loaded = True
            print("[INFO] Pipeline MLP cargado exitosamente para la Web App.")
            return True
        except Exception as e:
            print(f"[ERROR] Error al cargar los archivos pkl: {e}")
    else:
        print("[ADVERTENCIA] No se encontraron modelo_mlp.pkl, scaler.pkl o label_map.pkl en el directorio.")
    return False

# Cargar el pipeline al iniciar
cargar_pipeline()

# ==============================================================================
# PIPELINE DE EXTRACCIÓN DE CARACTERÍSTICAS DESDE PIL IMAGE
# ==============================================================================
def extraer_caracteristicas_de_pil(img_pil):
    """
    Recibe una imagen PIL, la redimensiona a 128x128 y extrae las 45 características.
    """
    img_resized = img_pil.resize(IMG_SIZE)
    img_rgb = img_resized.convert("RGB")
    img_gray = img_resized.convert("L")
    
    # Formato normalizado a [0, 1]
    rgb_arr = np.array(img_rgb) / 255.0
    gray_arr = np.array(img_gray) / 255.0
    
    features = {}
    
    # 1. Color - RGB
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
    
    # 2. Color - HSV
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
    
    # 3. Color - Gris y Mediana
    features["gray_mean"] = np.mean(gray_arr)
    features["gray_std"] = np.std(gray_arr)
    features["gray_median"] = np.median(gray_arr)
    
    # 4. Color - Histograma Gris
    hist_gray, _ = np.histogram(gray_arr, bins=8, range=(0.0, 1.0))
    hist_gray_norm = hist_gray / np.sum(hist_gray) if np.sum(hist_gray) > 0 else hist_gray
    for idx, val in enumerate(hist_gray_norm):
        features[f"hist_gray_{idx}"] = val
        
    # 5. Textura - Sobel
    sobel_img = sobel(gray_arr)
    features["sobel_mean"] = np.mean(sobel_img)
    features["sobel_std"] = np.std(sobel_img)
    
    # 6. Textura - Energía y Entropía
    features["energy"] = np.sum(gray_arr ** 2)
    features["entropy"] = shannon_entropy(gray_arr)
    
    # 7. Textura - LBP (10 bins)
    lbp = local_binary_pattern(gray_arr, P=24, R=3, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 25))
    lbp_hist_norm = lbp_hist / np.sum(lbp_hist) if np.sum(lbp_hist) > 0 else lbp_hist
    for idx, val in enumerate(lbp_hist_norm):
        features[f"lbp_{idx}"] = val
        
    # 8. Forma
    features["aspect_ratio"] = float(IMG_SIZE[0] / IMG_SIZE[1])
    
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

# ==============================================================================
# FUNCIÓN DE PREDICCIÓN PARA GRADIO
# ==============================================================================
def predecir_imagen(imagen_pil):
    if not model_loaded:
        return {"⚠️ Modelo no cargado o no encontrado": 1.0}
        
    if imagen_pil is None:
        return {"⚠️ Sube una imagen válida": 1.0}
        
    try:
        # Extraer características
        features = extraer_caracteristicas_de_pil(imagen_pil)
        df = pd.DataFrame([features])
        
        # Alinear columnas con las del escalador
        if hasattr(scaler, "feature_names_in_"):
            df = df[list(scaler.feature_names_in_)]
            
        # Escalar
        X_scaled = scaler.transform(df)
        
        # Calcular probabilidades
        probabilidades = modelo.predict_proba(X_scaled)[0]
        
        # Seleccionar las Top 5 predicciones más probables
        top5_indices = np.argsort(probabilidades)[-5:][::-1]
        
        resultado = {}
        for class_idx in top5_indices:
            clase_nom = label_map[class_idx]
            resultado[clase_nom] = float(probabilidades[class_idx])
            
        return resultado
    except Exception as e:
        return {f"⚠️ Error al procesar: {str(e)}": 1.0}

# ==============================================================================
# DISEÑO E INTERFAZ WEB PREMIUM (GRADIO BLOCKS)
# ==============================================================================
theme = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="stone",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
)

custom_css = """
#app-container {
    max-width: 900px;
    margin: 0 auto;
}
.header-gradient {
    background: linear-gradient(135deg, #dfa233, #a26914);
    padding: 30px;
    border-radius: 16px;
    text-align: center;
    color: white;
    box-shadow: 0 8px 32px 0 rgba(162, 105, 20, 0.2);
    margin-bottom: 25px;
}
.header-gradient h1 {
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0 0 10px 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.15);
}
.header-gradient p {
    font-size: 1.1rem;
    font-weight: 400;
    opacity: 0.95;
    margin: 0;
}
.glass-panel {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 12px;
}
"""

with gr.Blocks(theme=theme, css=custom_css) as demo:
    with gr.Column(elem_id="app-container"):
        
        # Encabezado Premium con degradado andino
        gr.HTML(
            """
            <div class="header-gradient">
                <h1>🥔 Clasificador de Papas Nativas Peruanas</h1>
                <p>Sube una foto de una papa nativa andina. Nuestro modelo extraerá características tabulares y estimará su variedad biológica instantáneamente en tiempo real.</p>
            </div>
            """
        )
        
        status_message = "✅ Pipeline cargado: Red Neuronal MLP local lista" if model_loaded else "⚠️ No se cargó modelo_mlp.pkl. Entrena el modelo antes de ejecutar."
        gr.Markdown(f"<div style='text-align:center; font-weight:bold; color: #a26914; margin-bottom: 20px;'>{status_message}</div>")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil", 
                    label="📷 Sube la foto de la papa", 
                    sources=["upload", "webcam"],
                    elem_classes="glass-panel"
                )
                submit_btn = gr.Button("🔍 Clasificar variedad", variant="primary")
                
            with gr.Column(scale=1):
                label_output = gr.Label(
                    num_top_classes=5, 
                    label="🏆 Resultados de la Clasificación (Top 5)",
                    elem_classes="glass-panel"
                )
                
        # Acciones
        submit_btn.click(
            fn=predecir_imagen,
            inputs=image_input,
            outputs=label_output
        )
        
        # Pie de página
        gr.Markdown(
            """
            <hr style="border: 0; border-top: 1px solid #ccc; margin: 30px 0 15px 0;" />
            <div style="text-align:center; font-size: 0.9rem; color: #666;">
                Model: Red Neuronal MLP (256, 128, 64) · Features: 45 tabulares (Color, Textura y Forma) · Universidad Nacional de San Antonio Abad del Cusco (UNSAAC 2024)
            </div>
            """
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
