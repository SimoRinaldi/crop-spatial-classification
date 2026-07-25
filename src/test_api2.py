import os
import time
import random
import socket
from datetime import datetime
import concurrent.futures
import rioxarray
import rasterio  # Aggiungiamo rasterio per la gestione dell'ambiente
from pystac_client import Client
from tqdm import tqdm

socket.setdefaulttimeout(30)

#TODO: aggiungere logger, aggiungere tracker di successi e fallimenti, 
# aggiungere logica per caricare putni e determinare quali anni andare a scaricare 
#! sentinel-2-l2a disponibile da fine 2018 circa, occhio, tagliare magari e tenere solo 2019 in avanti
# aggiungere merge finale per ogni punto: un .tif per ogni timestep
# aggiungere bande mancanti
# aggiungere in output oltre al merge un file metadata json, con info sul campione
    # dataset di provenienza, coordinate, label, data del campionamento
    # info sull'osservazione satellitare: data, cloud cover...
# ==========================================
# 1. Ottimizzazioni GDAL Globali
# ==========================================
STAC_API_URL = "https://earth-search.aws.element84.com/v1"

# Accelerazioni critiche per file COG su bucket pubblici AWS
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["VSI_CACHE"] = "TRUE"
# Ferma GDAL se il server S3 non risponde entro 30 secondi
os.environ["GDAL_HTTP_TIMEOUT"] = "30"
os.environ["GDAL_HTTP_CONNECTTIMEOUT"] = "30"
# Dici a GDAL di riprovare internamente fino a 3 volte se cade la linea
os.environ["GDAL_HTTP_MAX_RETRY"] = "3"


BAND_MAPPING = {
    "B02_10m": "blue",
    "B03_10m": "green",
    "B04_10m": "red",
    "B08_10m": "nir",
    "B11_20m": "swir16",
    "B12_20m": "swir22"
}

BUFFER_DEG = 0.005

# ==========================================
# 2. Funzioni Core
# ==========================================
def get_best_monthly_items(stac_items):
    """Trova l'osservazione con minor copertura nuvolosa per ogni mese."""
    monthly_best = {}
    for item in stac_items:
        date_str = item.properties.get("datetime")
        if not date_str:
            continue
            
        cloud_cover = item.properties.get("eo:cloud_cover", 100)
        month_key = date_str[:7] # Estrae "YYYY-MM"
        
        if month_key not in monthly_best:
            monthly_best[month_key] = item
        elif cloud_cover < monthly_best[month_key].properties.get("eo:cloud_cover", 100):
            monthly_best[month_key] = item
            
    return monthly_best

def search_stac_with_retry(client, bbox, start_date, end_date, max_retries=5):
    """Esegue la ricerca STAC gestendo i ban temporanei (HTTP 429) dell'API."""
    for attempt in range(max_retries):
        try:
            search = client.search(
                collections=["sentinel-2-l2a"], #se l2a non è disponibile provate l1c, altrimenti se vedete che l1c fa schifo usate dati del 2019, dubito cambierà tantissimo
                bbox=bbox,
                datetime=f"{start_date}/{end_date}",
                query={"eo:cloud_cover": {"lt": 60}}
            )
            return list(search.items())
        except Exception as e:
            if attempt == max_retries - 1:
                raise e # Se è l'ultimo tentativo, alza bandiera bianca
            
            # Exponential backoff: aspetta 2s, poi 4s, poi 8s + un po' di casualità
            sleep_time = (2 ** attempt) + random.uniform(0.1, 1.5)
            print(f"⚠️ [API Timeout/429] Ritento tra {sleep_time:.1f} sec... ({e})")
            time.sleep(sleep_time)

def process_point(point_id, lon, lat, years):
    """Worker isolato: usa un proprio ambiente GDAL indipendente."""
    time.sleep(random.uniform(0.1, 2.0))
    
    bbox = [lon - BUFFER_DEG, lat - BUFFER_DEG, lon + BUFFER_DEG, lat + BUFFER_DEG]
    point_dir = f"./test_aws_download/{point_id}"
    os.makedirs(point_dir, exist_ok=True)
    
    downloaded_count = 0
    failed_operations = []

    try:
        client = Client.open(STAC_API_URL)
    except Exception as e:
        return point_id, downloaded_count, [f"Connessione fallita: {e}"]

    # Configurazione di isolamento per GDAL: ogni processo ha il suo motore pulito
    gdal_config = {
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "VSI_CACHE": "FALSE",  # FONDAMENTALE: spegnere la cache condivisa nei multiprocesso
        "GDAL_HTTP_TIMEOUT": "30",
        "GDAL_HTTP_CONNECTTIMEOUT": "30",
        "GDAL_HTTP_MAX_RETRY": "3"
    }

    # Avvolgiamo tutto il lavoro geospaziale nel rasterio.Env()
    with rasterio.Env(**gdal_config):
        for year in years:
            try:
                items = search_stac_with_retry(client, bbox, f"{year}-01-01", f"{year}-12-31")
            except Exception as e:
                failed_operations.append(f"STAC Search fallita ({year}): {e}")
                continue

            if not items:
                continue

            best_monthly_items = get_best_monthly_items(items)

            for month, item in best_monthly_items.items():
                for cdse_name, aws_name in BAND_MAPPING.items():
                    if aws_name in item.assets:
                        href = item.assets[aws_name].href
                        out_path = os.path.join(point_dir, f"{month}_{item.id}_{cdse_name}.tif")
                        
                        if os.path.exists(out_path):
                            continue 
                        
                        for attempt in range(3):
                            try:
                                ds = rioxarray.open_rasterio(href)
                                cropped_ds = ds.rio.clip_box(*bbox, crs="EPSG:4326")
                                cropped_ds.rio.to_raster(out_path, compress="deflate", predictor=2, tiled=True)
                                downloaded_count += 1
                                break
                            except Exception as e:
                                if attempt == 2:
                                    failed_operations.append(f"Errore {month} {aws_name}: {e}")
                                else:
                                    time.sleep(1.5)

    return point_id, downloaded_count, failed_operations

# ==========================================
# 3. Wrapper Parallelizzato Principale
# ==========================================
if __name__ == "__main__":
    # Esempio per test. Puoi metterne centinaia qua dentro.
    points = [
        ("Vigneto_Toscana", 11.33, 43.55),
        ("Meleto_Trentino", 11.12, 46.06),
        ("Ulivi_Puglia", 17.65, 40.58),
        ("Agrumeto_Sicilia", 14.88, 37.52),
        ("Città_di_Roma", 12.48, 41.90),
        ("Città_di_Milano", 9.19, 45.46),
        ("Città_di_Venezia", 12.33, 45.43),
        ("Città_di_Firenze", 11.25, 43.77),
        ("Città_di_Napoli", 14.25, 40.85),
        ("Città_di_Bologna", 11.34, 44.49),
        ("Città_di_Genova", 9.19, 44.41),
        ("Città_di_Palermo", 13.36, 38.11),
        ("Città_di_Bari", 16.87, 41.12),
        ("Città_di_Catania", 15.09, 37.50),
        ("Città_di_Torino", 7.67, 45.05),
        ("Città_di_Verona", 10.99, 45.44),
        ("Città_di_Padova", 11.88, 45.41),
        ("Città_di_Bergamo", 9.67, 45.70),
        ("Città_di_Parma", 10.33, 44.80),
        ("Città_di_Lecce", 18.17, 40.35),
        ("Città_di_ReggioCalabria", 15.65, 38.11),
        ("Città_di_Salerno", 14.15, 40.68),
        ("Città_di_Foggia", 15.55, 41.46),
        ("Città_di_Sassari", 8.56, 40.73),
        ("Città_di_Cagliari", 9.11, 40.22),
        ("Città_di_Ancona", 13.51, 43.60),
        ("Città_di_Pescara", 14.21, 42.46),
        ("Città_di_Livorno", 10.31, 43.55),
        ("Città_di_Rimini", 12.57, 43.94),
        ("Città_di_Siena", 11.33, 43.32),
        ("Città_di_Arezzo", 11.88, 43.46),
        ("Città_di_Pisa", 10.40, 43.72),
    ]
    
    YEARS_TO_FETCH = [2020, 2021, 2022, 2023]
    MAX_WORKERS = 32
    
    total_start = time.perf_counter()
    failed_points_summary = {}
    total_tasks = len(points)
    
    print(f"Avvio in modalità MULTIPROCESSING. Elaborazione di {total_tasks} colture (Core impiegati: {MAX_WORKERS})...")

    # CAMBIAMENTO CRITICO: Usa ProcessPoolExecutor al posto di ThreadPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_point = {
            executor.submit(process_point, pid, lon, lat, YEARS_TO_FETCH): pid 
            for pid, lon, lat in points
        }
        
        with tqdm(total=total_tasks, desc="Elaborazione", unit="punto") as pbar:
            for future in concurrent.futures.as_completed(future_to_point):
                pid = future_to_point[future]
                try:
                    point_id, count, errors = future.result()
                    if errors:
                        failed_points_summary[point_id] = errors
                    
                    pbar.set_postfix({"Ultimo": str(point_id)[:10], "Tif": count})
                    
                except Exception as exc:
                    failed_points_summary[pid] = [str(exc)]
                
                pbar.update(1)

    total_end = time.perf_counter()
    
    # 4. A processo finito, stampiamo il riepilogo
    print(f"\n✅ Esecuzione Totale Terminata in {total_end - total_start:.2f} secondi")
    
    if failed_points_summary:
        print("\n⚠️ ATTENZIONE: Alcuni punti hanno registrato errori e necessitano di un retry:")
        for p, errs in failed_points_summary.items():
            print(f"  - {p}: {len(errs)} errori (es: {errs[0][:50]}...)")
    else:
        print("\nTutti i punti scaricati senza errori.")