import os
import time
import random
import socket
from datetime import datetime
import concurrent.futures
import multiprocessing
import rioxarray
import rasterio
from pystac_client import Client
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path

# %%
socket.setdefaulttimeout(30)

# %%
#TODO: aggiungere logger, aggiungere tracker di successi e fallimenti, 
# aggiungere logica per caricare putni e determinare quali anni andare a scaricare 
# # ! sentinel-2-l2a disponibile da fine 2018 circa, occhio, tagliare magari e tenere solo 2019 in avanti
# aggiungere merge finale per ogni punto: un .tif per ogni timestep
# aggiungere bande mancanti
# aggiungere in output oltre al merge un file metadata json, con info sul campione
    # dataset di provenienza, coordinate, label, data del campionamento
    # info sull'osservazione satellitare: data, cloud cover...
# ==========================================
# 1. Ottimizzazioni GDAL Globali
# ==========================================
STAC_API_URL = "https://earth-search.aws.element84.com/v1"

# %%
BAND_MAPPING = {
    "B02_10m": "blue",
    "B03_10m": "green",
    "B04_10m": "red",
    "B08_10m": "nir",
    "B11_20m": "swir16",
    "B12_20m": "swir22"
}

# %%
BUFFER_DEG = 0.005

# %%
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

# %%
def search_stac_with_retry(client, bbox, start_date, end_date, max_retries=5):
    """Esegue la ricerca STAC gestendo i ban temporanei (HTTP 429) dell'API."""
    for attempt in range(max_retries):
        try:
            search = client.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{start_date}/{end_date}",
                query={"eo:cloud_cover": {"lt": 60}}
            )
            return list(search.items())
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            sleep_time = (2 ** attempt) + random.uniform(0.1, 1.5)
            time.sleep(sleep_time)

# %%
def process_point(point_id, lon, lat, years, base_out_dir="./test_aws_download", progress_queue=None):
    """Worker isolato: usa un proprio ambiente GDAL indipendente."""
    time.sleep(random.uniform(0.05, 0.5))
    
    bbox = [lon - BUFFER_DEG, lat - BUFFER_DEG, lon + BUFFER_DEG, lat + BUFFER_DEG]
    point_dir = os.path.join(base_out_dir, str(point_id))
    os.makedirs(point_dir, exist_ok=True)
    
    downloaded_count = 0
    failed_operations = []

    try:
        client = Client.open(STAC_API_URL)
    except Exception as e:
        return point_id, downloaded_count, [f"Connessione fallita: {e}"]

    gdal_config = {
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "VSI_CACHE": "FALSE",
        "GDAL_HTTP_TIMEOUT": "30",
        "GDAL_HTTP_CONNECTTIMEOUT": "30",
        "GDAL_HTTP_MAX_RETRY": "3"
    }

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
                            downloaded_count += 1
                            if progress_queue is not None:
                                progress_queue.put(('file', 1, point_id))
                            continue 
                        
                        for attempt in range(3):
                            try:
                                ds = rioxarray.open_rasterio(href)
                                cropped_ds = ds.rio.clip_box(*bbox, crs="EPSG:4326")
                                cropped_ds.rio.to_raster(out_path, compress="deflate", predictor=2, tiled=True)
                                downloaded_count += 1
                                if progress_queue is not None:
                                    progress_queue.put(('file', 1, point_id))
                                break
                            except Exception as e:
                                if attempt == 2:
                                    failed_operations.append(f"Errore {month} {aws_name}: {e}")
                                    if progress_queue is not None:
                                        progress_queue.put(('file', 1, point_id))
                                else:
                                    time.sleep(1.5)

    return point_id, downloaded_count, failed_operations

# %%
# ==========================================
# 3. Wrapper Parallelizzato Principale
# ==========================================
def download_sentinel_data(points_df, years_to_fetch=[2023], max_workers=32, out_dir="./test_aws_download"):
    # Accelerazioni critiche per file COG su bucket pubblici AWS
    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
    os.environ["VSI_CACHE"] = "TRUE"
    os.environ["GDAL_HTTP_TIMEOUT"] = "30"
    os.environ["GDAL_HTTP_CONNECTTIMEOUT"] = "30"
    os.environ["GDAL_HTTP_MAX_RETRY"] = "3"

    points = []
    for idx, row in points_df.iterrows():
        lat_val = float(row['lat'])
        lon_val = float(row['lon'])
        
        # Auto-correzione se lat e lon sono stati invertiti (in Italia: lat ~ 41° N, lon ~ 15° E)
        if abs(lat_val) < abs(lon_val):
            lat_val, lon_val = lon_val, lat_val

        point_id = f"point_{idx}"
        
        points.append((point_id, lon_val, lat_val))

    total_expected_files = len(points) * len(years_to_fetch) * 12 * len(BAND_MAPPING)
    total_start = time.perf_counter()
    failed_points_summary = {}
    completed_points_count = 0
    
    print(f"Avvio MULTIPROCESSING: {len(points)} punti, {max_workers} worker.")
    print(f"Stima totale file TIF da scaricare: ~{total_expected_files} file.")

    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_point = {
            executor.submit(process_point, pid, lon, lat, years_to_fetch, out_dir, progress_queue): pid 
            for pid, lon, lat in points
        }
        
        total_futures = len(future_to_point)
        
        with tqdm(total=total_expected_files, desc="Scaricamento TIF", unit="file") as pbar:
            completed_futures = 0
            
            while completed_futures < total_futures:
                # Svuota i messaggi dalla queue in tempo reale
                while not progress_queue.empty():
                    try:
                        msg_type, count, pid_item = progress_queue.get_nowait()
                        if msg_type == 'file':
                            pbar.update(count)
                            pbar.set_postfix({"Punti": f"{completed_points_count}/{total_futures}", "Ultimo": str(pid_item)[:15]})
                    except Exception:
                        break
                
                # Controlla quanti punti/future hanno terminato completamente
                done_futures = [f for f in future_to_point if f.done()]
                if len(done_futures) > completed_futures:
                    for f in done_futures:
                        pid = future_to_point[f]
                        if not hasattr(f, '_processed_result'):
                            setattr(f, '_processed_result', True)
                            completed_points_count += 1
                            try:
                                point_id, count_res, errors = f.result()
                                if errors:
                                    failed_points_summary[point_id] = errors
                            except Exception as exc:
                                failed_points_summary[pid] = [str(exc)]
                    completed_futures = len(done_futures)
                
                time.sleep(0.1)

            # Processa eventuali messaggi residui nella queue
            while not progress_queue.empty():
                try:
                    msg_type, count, pid_item = progress_queue.get_nowait()
                    if msg_type == 'file':
                        pbar.update(count)
                except Exception:
                    break

    total_end = time.perf_counter()
    
    print(f"\n✅ Esecuzione Totale Terminata in {total_end - total_start:.2f} secondi")
    
    if failed_points_summary:
        print("\n⚠️ ATTENZIONE: Alcuni punti hanno registrato errori e necessitano di un retry:")
        for p, errs in failed_points_summary.items():
            print(f"  - {p}: {len(errs)} errori (es: {errs[0][:50]}...)")
    else:
        print("\nTutti i punti scaricati senza errori.")
        
    return failed_points_summary
