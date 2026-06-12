from pathlib import Path
import argparse
import warnings

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")


DATASET_PATH = Path(r"C:\Users\rodri\Documents\8vo ciclo\Inteligencia Artificial\accidente_aereo_IA\Entrenamiento_papas\dataset\dataset128x128")
OUTPUT_DIR = Path(r"C:\Users\rodri\Documents\8vo ciclo\Inteligencia Artificial\accidente_aereo_IA\Entrenamiento_papas\Clasificacion_papas")


def obtener_color_dominante_variedad(carpeta_variedad: Path) -> dict | None:
    """
    Analiza todas las imágenes de una variedad y devuelve un resumen RGB/HSV.
    """
    imagenes = list(carpeta_variedad.glob("*.jpg")) + list(carpeta_variedad.glob("*.png"))
    if not imagenes:
        return None

    colores_rgb = []
    colores_hsv = []

    for img_path in imagenes:
        try:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)

            # Máscara para ignorar fondo blanco/muy claro
            mask = ~((img_array[:, :, 0] > 240) &
                     (img_array[:, :, 1] > 240) &
                     (img_array[:, :, 2] > 240))

            if not np.any(mask):
                mask = np.ones(img_array.shape[:2], dtype=bool)

            pixeles_rgb = img_array[mask]
            if pixeles_rgb.size == 0:
                continue

            color_rgb_medio = pixeles_rgb.mean(axis=0)
            colores_rgb.append(color_rgb_medio)

            img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            pixeles_hsv = img_hsv[mask]
            color_hsv_medio = pixeles_hsv.mean(axis=0)
            colores_hsv.append(color_hsv_medio)

        except Exception:
            continue

    if not colores_rgb:
        return None

    rgb_final = np.mean(colores_rgb, axis=0)
    hsv_final = np.mean(colores_hsv, axis=0)

    return {
        "variedad": carpeta_variedad.name,
        "n_imagenes": len(imagenes),
        "r_mean": float(rgb_final[0]),
        "g_mean": float(rgb_final[1]),
        "b_mean": float(rgb_final[2]),
        "h_mean": float(hsv_final[0]),
        "s_mean": float(hsv_final[1]),
        "v_mean": float(hsv_final[2]),
        "rgb_tuple": tuple(np.round(rgb_final).astype(int)),
    }


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convierte un color RGB a formato hexadecimal."""
    return "#{:02X}{:02X}{:02X}".format(int(round(r)), int(round(g)), int(round(b)))


def asignar_nombre_color(h_avg: float, s_avg: float, v_avg: float) -> str:
    """
    Etiqueta más fina de color usando H, S y V.
    """
    # Tonos muy pálidos / crema
    if s_avg < 35:
        return "Crema/Blanca" if v_avg > 150 else "Oscura/Negra"

    # Rojo / rosado
    if h_avg < 8 or h_avg >= 170:
        return "Rojo/Rosado"

    # Naranja
    if 8 <= h_avg < 18:
        if s_avg >= 110 and v_avg >= 110:
            return "Naranja intenso"
        if s_avg < 80:
            return "Naranja pálido"
        return "Naranja brillante"

    # Amarillo
    if 18 <= h_avg < 30:
        return "Amarillo intenso" if s_avg >= 100 else "Amarillo suave"

    # Verde
    if 30 <= h_avg < 55:
        return "Verde"

    # Azul / morado
    if 55 <= h_avg < 140:
        return "Azul/Morado"

    return "Morado/Violeta"


def main():
    parser = argparse.ArgumentParser(description="Explorar colores dominantes del dataset de papas")
    parser.add_argument("--k", type=int, default=5, help="Número de grupos de color a usar (por defecto 5)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {DATASET_PATH}")

    print("ETAPA 1: Extrayendo color dominante por variedad...")
    carpetas = sorted([c for c in DATASET_PATH.iterdir() if c.is_dir()])
    print(f"Variedades encontradas: {len(carpetas)}")

    registros = []
    for carpeta in tqdm(carpetas, desc="Analizando variedades"):
        resultado = obtener_color_dominante_variedad(carpeta)
        if resultado:
            registros.append(resultado)

    df_colores = pd.DataFrame(registros)
    print(f"\n✓ Variedades analizadas: {len(df_colores)}")
    print(df_colores[["variedad", "r_mean", "g_mean", "b_mean", "h_mean", "s_mean", "v_mean"]].to_string(index=False))

    print("\nETAPA 2: Determinando número óptimo de grupos de color (Elbow Method)...")
    X_cluster = df_colores[["h_mean", "s_mean", "v_mean"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    inercias = []
    k_range = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inercias.append(km.inertia_)
        print(f"  K={k} → Inercia: {km.inertia_:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(list(k_range), inercias, marker="o", linewidth=2, markersize=8, color="steelblue")
    plt.xlabel("Número de grupos (K)", fontsize=13)
    plt.ylabel("Inercia (WCSS)", fontsize=13)
    plt.title("Método del Codo — Número óptimo de grupos de color\nPapas Nativas Peruanas", fontsize=14)
    plt.xticks(list(k_range))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "elbow_method.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Gráfico guardado: elbow_method.png")

    K_OPTIMO = args.k
    print(f"\nETAPA 3: Aplicando KMeans con K={K_OPTIMO}...")
    km_final = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=10)
    df_colores["cluster"] = km_final.fit_predict(X_scaled)

    nombres_cluster = {}
    hex_por_cluster = {}
    for cluster_id in range(K_OPTIMO):
        mask_cluster = df_colores["cluster"] == cluster_id
        h_avg = df_colores.loc[mask_cluster, "h_mean"].mean()
        s_avg = df_colores.loc[mask_cluster, "s_mean"].mean()
        v_avg = df_colores.loc[mask_cluster, "v_mean"].mean()
        r_avg = df_colores.loc[mask_cluster, "r_mean"].mean()
        g_avg = df_colores.loc[mask_cluster, "g_mean"].mean()
        b_avg = df_colores.loc[mask_cluster, "b_mean"].mean()
        n_variedades = int(mask_cluster.sum())

        nombre = asignar_nombre_color(h_avg, s_avg, v_avg)
        color_hex = rgb_to_hex(r_avg, g_avg, b_avg)
        nombres_cluster[cluster_id] = nombre
        hex_por_cluster[cluster_id] = color_hex

        print(f"\n  Cluster {cluster_id}:")
        print(f"    H={h_avg:.1f}, S={s_avg:.1f}, V={v_avg:.1f}")
        print(f"    Variedades en este grupo: {n_variedades}")
        print(f"    Variedades: {list(df_colores.loc[mask_cluster, 'variedad'])}")
        print(f"    → Color asignado: {nombre}")
        print(f"    → HEX: {color_hex}")

    df_colores["color_grupo"] = df_colores["cluster"].map(nombres_cluster)
    df_colores["color_hex"] = df_colores["cluster"].map(hex_por_cluster)

    print("\nETAPA 4: Generando visualizaciones...")
    fig, axes = plt.subplots(1, K_OPTIMO, figsize=(4 * K_OPTIMO, 10))
    fig.suptitle("Paleta de Colores — Papas Nativas Peruanas por Grupo\n(dataset128x128)", fontsize=14, fontweight="bold")

    for i, (cluster_id, nombre_color) in enumerate(nombres_cluster.items()):
        ax = axes[i] if K_OPTIMO > 1 else axes
        variedades_cluster = df_colores[df_colores["cluster"] == cluster_id]

        ax.set_title(f"{nombre_color}\n({len(variedades_cluster)} variedades)", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(variedades_cluster))
        ax.axis("off")

        for j, (_, row) in enumerate(variedades_cluster.iterrows()):
            color_norm = (row["r_mean"] / 255, row["g_mean"] / 255, row["b_mean"] / 255)
            rect = mpatches.FancyBboxPatch(
                (0.05, j + 0.1), 0.6, 0.75,
                boxstyle="round,pad=0.02",
                facecolor=color_norm, edgecolor="gray", linewidth=0.5
            )
            ax.add_patch(rect)
            ax.text(0.72, j + 0.47, row["variedad"], va="center", fontsize=7, color="black")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "paleta_colores_variedades.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Gráfico guardado: paleta_colores_variedades.png")

    conteo = df_colores["color_grupo"].value_counts().reset_index()
    conteo.columns = ["Color", "Cantidad de variedades"]

    plt.figure(figsize=(10, 5))
    colores_barra = ["#F4D03F", "#8E44AD", "#E74C3C", "#F5CBA7", "#1C2833"][:K_OPTIMO]
    sns.barplot(data=conteo, x="Cantidad de variedades", y="Color", palette=colores_barra, edgecolor="black")
    plt.title("Distribución de Variedades por Grupo de Color\nPapas Nativas Peruanas — 83 variedades", fontsize=13)
    plt.xlabel("Número de variedades", fontsize=11)
    plt.ylabel("Grupo de color", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distribucion_grupos_color.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Gráfico guardado: distribucion_grupos_color.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colores_scatter = plt.cm.tab10(np.linspace(0, 1, K_OPTIMO))

    for cluster_id in range(K_OPTIMO):
        mask = df_colores["cluster"] == cluster_id
        subset = df_colores[mask]
        nombre = nombres_cluster[cluster_id]

        ax1.scatter(subset["h_mean"], subset["s_mean"], color=colores_scatter[cluster_id], label=nombre,
                    s=80, edgecolors="black", linewidth=0.5)
        ax2.scatter(subset["h_mean"], subset["v_mean"], color=colores_scatter[cluster_id], label=nombre,
                    s=80, edgecolors="black", linewidth=0.5)

    ax1.set_xlabel("Hue (H)", fontsize=11)
    ax1.set_ylabel("Saturación (S)", fontsize=11)
    ax1.set_title("Espacio H-S por grupo de color", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Hue (H)", fontsize=11)
    ax2.set_ylabel("Valor/Brillo (V)", fontsize=11)
    ax2.set_title("Espacio H-V por grupo de color", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Distribución en Espacio HSV — Papas Nativas Peruanas", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_hsv_clusters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Gráfico guardado: scatter_hsv_clusters.png")

    print("\nETAPA 5: Guardando resultados...")
    df_resultado = df_colores[["variedad", "color_grupo", "color_hex", "cluster", "r_mean", "g_mean", "b_mean", "h_mean", "s_mean", "v_mean", "n_imagenes"]]
    df_resultado = df_resultado.sort_values("cluster")
    df_resultado.to_csv(OUTPUT_DIR / "mapeo_variedades_colores.csv", index=False)
    print("✓ Guardado: mapeo_variedades_colores.csv")

    reporte = []
    reporte.append("=" * 60)
    reporte.append("REPORTE DE EXPLORACIÓN DE COLORES")
    reporte.append("Dataset: papas-nativas-peru-83-variedades (dataset128x128)")
    reporte.append("=" * 60)
    reporte.append(f"\nTotal variedades analizadas: {len(df_colores)}")
    reporte.append(f"Número de grupos de color (K óptimo): {K_OPTIMO}")
    reporte.append("\nDISTRIBUCIÓN POR GRUPO DE COLOR:")
    reporte.append("-" * 40)

    for cluster_id, nombre in nombres_cluster.items():
        mask = df_colores["cluster"] == cluster_id
        variedades = list(df_colores.loc[mask, "variedad"])
        h_avg = df_colores.loc[mask, "h_mean"].mean()
        s_avg = df_colores.loc[mask, "s_mean"].mean()
        v_avg = df_colores.loc[mask, "v_mean"].mean()
        color_hex = hex_por_cluster.get(cluster_id, "#000000")
        reporte.append(f"\n[Grupo {cluster_id}] {nombre} — HEX: {color_hex}")
        reporte.append(f"  Variedades: {len(variedades)}")
        reporte.append(f"  HSV promedio: H={h_avg:.1f}, S={s_avg:.1f}, V={v_avg:.1f}")
        reporte.append(f"  Carpetas: {', '.join(variedades)}")

    reporte.append("\n" + "=" * 60)
    reporte.append("ARCHIVOS GENERADOS:")
    reporte.append("  - elbow_method.png")
    reporte.append("  - paleta_colores_variedades.png")
    reporte.append("  - distribucion_grupos_color.png")
    reporte.append("  - scatter_hsv_clusters.png")
    reporte.append("  - mapeo_variedades_colores.csv")
    reporte.append("=" * 60)

    reporte_texto = "\n".join(reporte)
    print(reporte_texto)

    with open(OUTPUT_DIR / "reporte_colores.txt", "w", encoding="utf-8") as f:
        f.write(reporte_texto)
    print("✓ Guardado: reporte_colores.txt")

    print("\n✓ EXPLORACIÓN COMPLETA.")
    print("Revisa la curva del codo (elbow_method.png) para confirmar el K óptimo.")
    print("Luego usa mapeo_variedades_colores.csv para reentrenar los modelos con K clases.")


if __name__ == "__main__":
    main()
