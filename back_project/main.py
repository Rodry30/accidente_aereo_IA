from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
import subprocess
import os
import io
import joblib
from PIL import Image
from typing import Optional, List

app = FastAPI()

# CORS para que el frontend pueda consumirlo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar la carpeta del dataset original como archivos estáticos
from fastapi.staticfiles import StaticFiles
DATASET_DIR = r"C:\Users\rodri\Documents\8vo ciclo\Inteligencia Artificial\accidente_aereo_IA\Entrenamiento_papas\dataset\dataset128x128"
if os.path.exists(DATASET_DIR):
    app.mount("/dataset", StaticFiles(directory=DATASET_DIR), name="dataset")
    print(f"[INFO] Carpeta del dataset montada en /dataset desde: {DATASET_DIR}")
else:
    print(f"[WARNING] No se encontró el dataset en la ruta física: {DATASET_DIR}. El visor de imágenes de ejemplo no estará disponible.")


# ==============================================================================
# CARGA DE MODELOS Y PIPELINES (ACCIDENTE AÉREO Y PAPAS NATIVAS)
# ==============================================================================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# 1. Cargar modelo de Accidente Aéreo (Keras/TensorFlow)
accidente_model_path = os.path.join(MODEL_DIR, "test.keras")
model = tf.keras.models.load_model(accidente_model_path)
print("[INFO] Modelo de Accidente Aéreo cargado exitosamente.")

# 2. Cargar modelos de Papas Nativas (scikit-learn MLP y RF)
try:
    modelo_mlp = joblib.load(os.path.join(MODEL_DIR, "modelo_mlp.pkl"))
    scaler_mlp = joblib.load(os.path.join(MODEL_DIR, "scaler_mlp.pkl"))
    label_map_mlp = joblib.load(os.path.join(MODEL_DIR, "label_map_mlp.pkl"))
    papas_mlp_loaded = True
    print("[INFO] Modelo de Papas Nativas MLP cargado exitosamente.")
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo MLP de papas: {e}")
    papas_mlp_loaded = False

try:
    modelo_rf = joblib.load(os.path.join(MODEL_DIR, "modelo_rf.pkl"))
    scaler_rf = joblib.load(os.path.join(MODEL_DIR, "scaler_rf.pkl"))
    label_map_rf = joblib.load(os.path.join(MODEL_DIR, "label_map_rf.pkl"))
    papas_rf_loaded = True
    print("[INFO] Modelo de Papas Nativas Random Forest cargado exitosamente.")
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo RF de papas: {e}")
    papas_rf_loaded = False

IMG_SIZE_PAPA = (128, 128)

# ==============================================================================
# DIAGNÓSTICO EN PROLOG (DIABETES)
# ==============================================================================
class DiagnosticoRequest(BaseModel):
    sintomas: List[str]

def run_prolog_diagnostico(sintomas: List[str]) -> List[str]:
    prolog_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabetes.pl")
    sintomas_str = f"[{','.join(sintomas)}]"
    
    cmd = "swipl"
    try:
        subprocess.run([cmd, "--version"], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        default_path = r"C:\Program Files\swipl\bin\swipl.exe"
        if os.path.exists(default_path):
            cmd = default_path

    try:
        result = subprocess.run(
            [cmd, "-s", prolog_file, "-g", f"evaluar({sintomas_str}).", "-t", "halt."],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        if not output:
            return []
        return [d.strip() for d in output.split(",") if d.strip()]
    except Exception as e:
        print(f"Error al ejecutar SWI-Prolog: {e}")
        return []

@app.post("/diagnosticar")
def diagnosticar(req: DiagnosticoRequest):
    resultados = run_prolog_diagnostico(req.sintomas)
    return {
        "sintomas_recibidos": req.sintomas,
        "diagnosticos": resultados
    }

# ==============================================================================
# PREDICCIÓN ACCIDENTE AÉREO (TITANIC)
# ==============================================================================
class Pasajero(BaseModel):
    sex: str
    age: float
    n_siblings_spouses: int
    parch: int
    fare: float
    clase: str
    deck: str
    embark_town: str
    alone: str

@app.post("/predecir")
def predecir(pasajero: Pasajero):
    ejemplo = {
        'sex':                np.array([pasajero.sex]),
        'age':                np.array([pasajero.age]),
        'n_siblings_spouses': np.array([pasajero.n_siblings_spouses]),
        'parch':              np.array([pasajero.parch]),
        'fare':               np.array([pasajero.fare]),
        'class':              np.array([pasajero.clase]),
        'deck':               np.array([pasajero.deck]),
        'embark_town':        np.array([pasajero.embark_town]),
        'alone':              np.array([pasajero.alone])
    }

    prediccion = model(ejemplo)
    probabilidad = float(tf.sigmoid(prediccion).numpy()[0][0])

    return {
        "probabilidad": round(probabilidad, 4),
        "sobrevive": probabilidad > 0.5,
        "mensaje": "SOBREVIVE ✅" if probabilidad > 0.5 else "NO SOBREVIVE ❌"
    }

# ==============================================================================
# EXTRACCIÓN DE CARACTERÍSTICAS PARA PAPAS NATIVAS
# ==============================================================================
def extraer_caracteristicas_de_pil(img_pil):
    """
    Recibe una imagen PIL en memoria, la redimensiona y extrae exactamente las 45 características.
    """
    from skimage.color import rgb2hsv
    from skimage.filters import sobel, threshold_otsu
    from skimage.feature import local_binary_pattern
    from skimage.measure import label, regionprops, shannon_entropy
    
    img_resized = img_pil.resize(IMG_SIZE_PAPA)
    img_rgb = img_resized.convert("RGB")
    img_gray = img_resized.convert("L")
    
    rgb_arr = np.array(img_rgb) / 255.0
    gray_arr = np.array(img_gray) / 255.0
    
    features = {}
    
    # RGB Mean, Std, Median (9 features)
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
    
    # HSV Mean, Std, Median de V (7 features)
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
    
    # Gris Mean, Std, Median (3 features)
    features["gray_mean"] = np.mean(gray_arr)
    features["gray_std"] = np.std(gray_arr)
    features["gray_median"] = np.median(gray_arr)
    
    # Histograma Gris (8 features)
    hist_gray, _ = np.histogram(gray_arr, bins=8, range=(0.0, 1.0))
    hist_gray_norm = hist_gray / np.sum(hist_gray) if np.sum(hist_gray) > 0 else hist_gray
    for idx, val in enumerate(hist_gray_norm):
        features[f"hist_gray_{idx}"] = val
        
    # Textura - Sobel (2 features)
    sobel_img = sobel(gray_arr)
    features["sobel_mean"] = np.mean(sobel_img)
    features["sobel_std"] = np.std(sobel_img)
    
    # Textura - Energía y Entropía (2 features)
    features["energy"] = np.sum(gray_arr ** 2)
    features["entropy"] = shannon_entropy(gray_arr)
    
    # Textura - LBP (10 features)
    lbp = local_binary_pattern(gray_arr, P=24, R=3, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 25))
    lbp_hist_norm = lbp_hist / np.sum(lbp_hist) if np.sum(lbp_hist) > 0 else lbp_hist
    for idx, val in enumerate(lbp_hist_norm):
        features[f"lbp_{idx}"] = val
        
    # Forma - Aspect Ratio (1 feature)
    features["aspect_ratio"] = float(IMG_SIZE_PAPA[0] / IMG_SIZE_PAPA[1])
    
    # Forma - Segmentación Otsu y Regiones (3 features)
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
# ENDPOINT DE CLASIFICACIÓN DE PAPAS NATIVAS (MLP / RF)
# ==============================================================================
def aplicar_temperature_scaling(probabilidades: np.ndarray, T: float = 2.0) -> np.ndarray:
    """
    Aplica Temperature Scaling sobre un vector de probabilidades de scikit-learn.
    """
    eps = 1e-15
    logits = np.log(probabilidades + eps)
    logits_scaled = logits / T
    # Softmax estable
    exp_logits = np.exp(logits_scaled - np.max(logits_scaled))
    probabilidades_calibradas = exp_logits / np.sum(exp_logits)
    return probabilidades_calibradas

@app.post("/predecir_papa")
async def predecir_papa(
    file: UploadFile = File(...),
    modelo_tipo: str = Query("mlp", pattern="^(mlp|rf)$"),
    temperatura: float = Query(2.0, ge=0.1, le=10.0)
):
    # Validar que los modelos estén cargados
    if modelo_tipo == "mlp" and not papas_mlp_loaded:
        raise HTTPException(status_code=503, detail="El modelo MLP de papas nativas no está disponible.")
    if modelo_tipo == "rf" and not papas_rf_loaded:
        raise HTTPException(status_code=503, detail="El modelo Random Forest de papas nativas no está disponible.")
        
    try:
        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"El archivo subido no pudo leerse como imagen: {e}")
        
    try:
        # Extraer características
        features = extraer_caracteristicas_de_pil(img_pil)
        df = pd.DataFrame([features])
        
        # Seleccionar modelo
        if modelo_tipo == "mlp":
            clf = modelo_mlp
            scl = scaler_mlp
            lbl_map = label_map_mlp
        else:
            clf = modelo_rf
            scl = scaler_rf
            lbl_map = label_map_rf
            
        # Alinear columnas con scaler
        if hasattr(scl, "feature_names_in_"):
            df = df[list(scl.feature_names_in_)]
            
        # Escalar
        X_scaled = scl.transform(df)
        
        # Predicción probabilística
        probabilidades = clf.predict_proba(X_scaled)[0]
        
        # Aplicar calibración mediante Temperature Scaling si es MLP
        if modelo_tipo == "mlp":
            probabilidades = aplicar_temperature_scaling(probabilidades, temperatura)
            
        pred_class_id = np.argmax(probabilidades)
        
        clase_predicha = lbl_map[pred_class_id]
        confianza = probabilidades[pred_class_id]
        
        # Obtener Top 5 más probables
        top5_indices = np.argsort(probabilidades)[-5:][::-1]
        top5_probabilidades = []
        for idx in top5_indices:
            top5_probabilidades.append({
                "variedad": lbl_map[idx],
                "probabilidad": round(float(probabilidades[idx]), 4)
            })
            
        # Buscar una imagen de ejemplo física para la variedad predicha
        imagen_ejemplo_url = None
        if os.path.exists(DATASET_DIR):
            folder_path = os.path.join(DATASET_DIR, clase_predicha)
            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    # Retornar la ruta relativa servida por FastAPI
                    imagen_ejemplo_url = f"http://127.0.0.1:8000/dataset/{clase_predicha}/{files[0]}"
            
        return {
            "variedad_predicha": clase_predicha,
            "confianza": round(float(confianza), 4),
            "modelo_utilizado": modelo_tipo.upper(),
            "imagen_ejemplo_url": imagen_ejemplo_url,
            "top_5": top5_probabilidades
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno durante la clasificación: {str(e)}")