from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import subprocess
import os

from typing import Optional, List

app = FastAPI()

# CORS para que el frontend pueda consumirlo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schema del request para el diagnóstico de Diabetes
class DiagnosticoRequest(BaseModel):
    sintomas: List[str]

# Función para ejecutar el diagnóstico en Prolog usando swipl
def run_prolog_diagnostico(sintomas: List[str]) -> List[str]:
    prolog_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabetes.pl")
    sintomas_str = f"[{','.join(sintomas)}]"
    
    # Determinar el comando swipl
    cmd = "swipl"
    try:
        subprocess.run([cmd, "--version"], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback a la ruta por defecto de Windows
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

# Carga el modelo una sola vez al iniciar
model = tf.keras.models.load_model("model/test.keras")

# Schema del request — mismas columnas de tu dataset
class Pasajero(BaseModel):
    sex: str
    age: float
    n_siblings_spouses: int
    parch: int
    fare: float
    clase: str        # 'class' es palabra reservada en Python
    deck: str
    embark_town: str
    alone: str

@app.post("/predecir")
def predecir(pasajero: Pasajero):
    # Armamos el dict igual que cuando probaste el modelo
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