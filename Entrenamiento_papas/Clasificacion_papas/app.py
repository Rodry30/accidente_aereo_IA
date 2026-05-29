import huggingface_hub
if not hasattr(huggingface_hub, 'HfFolder'):
    class _HfFolder:
        @staticmethod
        def get_token(): return None
        @staticmethod
        def save_token(token): pass
    huggingface_hub.HfFolder = _HfFolder

import gradio_client.utils as _gcu
_orig = _gcu.json_schema_to_python_type
def _safe_json_schema(schema):
    try:
        return _orig(schema)
    except TypeError:
        return "Any"
_gcu.json_schema_to_python_type = _safe_json_schema

# starlette 0.40+ removed the old TemplateResponse(name, context) API.
# gradio 4.44.1 still uses the old API, so we restore backwards compatibility.
from starlette import templating as _st
_orig_tr = _st.Jinja2Templates.TemplateResponse
def _compat_tr(self, *args, **kwargs):
    if args and isinstance(args[0], str):
        name = args[0]
        context = dict(args[1]) if len(args) > 1 else {}
        request = context.get('request')
        return _orig_tr(self, request, name, context, *args[2:], **kwargs)
    return _orig_tr(self, *args, **kwargs)
_st.Jinja2Templates.TemplateResponse = _compat_tr

import gradio as gr
import numpy as np
from PIL import Image
import os

CLASSES = sorted([
    'V10','V11','V12','V13','V14','V15','V16','V17','V18','V19',
    'V20','V21','V22','V23','V24','V25','V26','V27','V28','V29',
    'V30','V31','V32','V33','V34','V35','V36','V37','V38','V39',
    'V40','V41','V42','V43','V44','V45','V46','V47','V48','V49',
    'V50','V51','V52','V53','V54','V55','V56','V57','V58','V59',
    'V60','V61','V62','V63','V64','V65','V66','V67','V68','V69',
    'V70','V71','V72','V73','V75','V76','V77','V78','V79',
    'V80','V81','V82','V83','V84',
    'V4','V5','V6','V7','V8','V9'
])  # 80 variedades

IMG_SIZE = (128, 128)
model = None

def load_model():
    global model
    for path in ["model.keras", "model.h5"]:
        if os.path.exists(path):
            try:
                import keras
                model = keras.models.load_model(path)
                print(f"Modelo cargado desde {path}")
                return True
            except Exception as e:
                print(f"Error cargando {path}: {e}")
    return False

model_loaded = load_model()

def predict(image):
    if not model_loaded or model is None:
        return {"⚠️ Modelo no encontrado": 1.0}
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    top5_idx = np.argsort(preds)[-5:][::-1]
    return {CLASSES[i]: float(preds[i]) for i in top5_idx}

STATUS = "✅ Modelo cargado" if model_loaded else "⚠️ Coloca model.keras en esta carpeta"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", sources=["upload", "webcam"], label="Foto de la papa"),
    outputs=gr.Label(num_top_classes=5, label="Variedades detectadas"),
    title="🥔 Clasificador de Papas Nativas Peruanas",
    description=(
        f"**{STATUS}**\n\n"
        "Sube una foto de una papa nativa peruana y el modelo identifica "
        "a cuál de las **80 variedades** pertenece.\n\n"
        "Modelo: MobileNetV2 entrenado desde cero · Accuracy: **91.67%** · "
        "Dataset: [HuggingFace](https://huggingface.co/datasets/ayayon/papas-nativas-peru-83-variedades)"
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
