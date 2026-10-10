import os
import sys
from pathlib import Path

# Confini geografici della Capitanata
# (EPSG: 4326 min_lon, min_lat, max_lon, max_lat)
CAPITANATA_BBOX = [
    15.046692771569205,  # min_lon (ovest)
    41.08392693013771,  # min_lat (sud)
    16.210767777473393,  # max_lon (est)
    41.92848568568325,  # max_lat (nord)
]

# Classi da escludere a priori:
EXCLUDED_CROP_CLASSES = [
    0,  # -> all non-cropland areas
    3100,  # -> Undecided arable crop
    3200,  # -> Undecided perennial crop
    65534,  # -> water / sea
    65535,  # -> outside area
]

# Anni campionati
YEARS_TO_FETCH = ["2023", "2022", "2021"]


def is_in_capitanata(lon: float, lat: float) -> bool:
    """Verifica se una coordinata (lon, lat) ricade all'interno della BBox della Capitanata."""
    min_lon, min_lat, max_lon, max_lat = CAPITANATA_BBOX
    return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)


def setup_environment(base_dir=None, git_branch="main"):
    """
    Inizializza l'ambiente di esecuzione (Google Colab o Locale).
    Se eseguito su Colab, monta Google Drive e configura l'accesso allo storage condiviso.
    Restituisce i percorsi universali dei dati: (DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR).
    """
    if "google.colab" in sys.modules:
        print("Ambiente Colab rilevato. Inizializzazione in corso...")
        from google.colab import drive

        if not os.path.exists("/content/drive"):
            drive.mount("/content/drive")

        base_path = (
            Path("/content/drive/Shareddrives/Progetti/Progetto_MLDM")
            if base_dir is None
            else Path(base_dir)
        )
        repo_dir = Path("/content/crop-spatial-classification")

        if not repo_dir.exists():
            os.system(
                f"git clone -b {git_branch} https://github.com/SimoRinaldi/crop-spatial-classification.git {repo_dir}"
            )
        if str(repo_dir) not in sys.path:
            sys.path.append(str(repo_dir))

        req_file = repo_dir / "requirements.txt"
        if req_file.exists():
            os.system(f"pip install -q -r {req_file}")

        print(f"Setup ambiente Colab completato! Root dati: {base_path}")
    else:
        print("Ambiente Locale rilevato. Procedo con l'esecuzione...")
        if base_dir is not None:
            base_path = Path(base_dir)
        elif (Path.cwd() / "src").is_dir():
            base_path = Path.cwd()
        else:
            base_path = Path.cwd().parent

        if str(base_path) not in sys.path:
            sys.path.append(str(base_path))

    data_dir = base_path / "data"
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"

    if raw_dir.exists():
        print(f"Collegamento ai dati riuscito! Cartella raw: {raw_dir}")
    else:
        print(f"Attenzione: Cartella non trovata in {raw_dir}. Verifica il percorso o il mount.")

    return data_dir, raw_dir, interim_dir, processed_dir


# Alias per compatibilita
mount_drive = setup_environment
