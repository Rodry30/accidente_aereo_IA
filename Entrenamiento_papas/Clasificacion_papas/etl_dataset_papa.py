from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ayayon/papas-nativas-peru-83-variedades",
    repo_type="dataset",
    allow_patterns="dataset128x128/*",
    local_dir=r"C:\Users\rodri\Documents\8vo ciclo\Inteligencia Artificial\accidente_aereo_IA\Entrenamiento_papas\dataset",
    token=""
)