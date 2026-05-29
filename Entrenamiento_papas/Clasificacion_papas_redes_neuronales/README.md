---
license: cc-by-nc-4.0
task_categories:
- image-classification
language:
- es
tags:
- potatoes
- native-species
- peru
- andes
- agriculture
- biodiversity
size_categories:
- 1K<n<10K
---

# Papas Nativas Peruanas — 83 Variedades (UNSAAC 2024)

Colección de imágenes de **83 variedades de papas nativas peruanas** para clasificación visual mediante modelos de visión computacional.

## Descripción del dataset

Dataset recopilado de forma colaborativa por estudiantes de Ingeniería Informática de la **Universidad Nacional de San Antonio Abad del Cusco (UNSAAC)** en el curso de Aprendizaje Automático (2024).

Las imágenes fueron capturadas en condiciones variadas (distintos dispositivos móviles, ángulos e iluminación), lo que proporciona diversidad natural útil para entrenar modelos robustos para clasificación en condiciones reales.

## Estructura

```
dataset/
├── dataset128x128/          # Imágenes originales 128×128 px
│   ├── V1/                  # Variedad 1 (~50 imágenes)
│   ├── V2/
│   └── ... (83 variedades)
├── dataset64x64/            # Mismas imágenes a 64×64 px
└── dataset_aumentation_128x128/  # Dataset aumentado 128×128 px
    ├── V1/                  # 300 imágenes por variedad (aug)
    └── ... (80 variedades)
```

| Split | Resolución | Variedades | Imgs/variedad | Total aprox. |
|---|---|---|---|---|
| Original | 128×128 px | 83 | ~50 | ~4,150 |
| Original | 64×64 px | 83 | ~50 | ~4,150 |
| Augmentado | 128×128 px | 80 | 300 | ~24,000 |

## Técnicas de augmentación aplicadas

- Rotaciones controladas (±15°)
- Ajustes de brillo/contraste (±20%)
- Volteo horizontal aleatorio

## Resultados de clasificación con este dataset

Entrenando 4 arquitecturas CNN con el split aumentado (70% train / 15% val / 15% test):

| Modelo | Accuracy | Parámetros |
|---|---|---|
| MobileNetV2 (custom) | **91.67%** | 2.4M |
| SqueezeNet v1.1 | 91.16% | 0.77M |
| VGG16 (transfer learning) | 85.24% | 18.9M |
| PapaNet (custom) | 84.98% | 1.3M |

Ver detalles: [github.com/chikiyu/cnn-clasificacion-papas](https://github.com/chikiyu/cnn-clasificacion-papas)

## Limitaciones conocidas

- Las imágenes fueron capturadas con diferentes dispositivos móviles bajo condiciones no controladas de iluminación y ángulo.
- Los nombres de las variedades están codificados como V1–V83 (el mapeo exacto nombre quechua / nombre científico ↔ código está documentado en el paper del repositorio).
- El dataset de augmentación cubre 80 de las 83 variedades (3 con muy pocas imágenes originales fueron excluidas del proceso).

## Cómo usar (Python)

```python
from datasets import load_dataset

ds = load_dataset("ayayon/papas-nativas-peru-83-variedades", split="train")
```

O directamente con `ImageFolder` de PyTorch / TensorFlow:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
    "dataset128x128/",
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical"
)
```

## Autoría y créditos

Dataset recopilado como proyecto del curso de Aprendizaje Automático, UNSAAC 2024.

**Autores del modelo y paper:**
> Torreblanca Paz, S. V., Pachari Lipa, M. A., & Sullcarani Diaz, B. E. (2024).
> *Evaluación comparativa de arquitecturas CNN en la clasificación de 83 variedades de papas nativas andinas.*
> Universidad Nacional San Antonio Abad del Cusco.

## Licencia

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — libre uso para investigación y educación, no comercial. Atribución requerida.
