from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

from typing import Optional

app = FastAPI()

# CORS para que Angular pueda consumirlo
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://localhost:4201",
    ],
    allow_methods=["POST"],
    allow_headers=["*"],
)

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