"""
Modulo per il calcolo unificato delle Feature Spazio-Temporali (Fenologiche e Spettrali).
Condiviso tra il Notebook 03 (Dataset Creation) e il Notebook 05 (Spatial Visualization).
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling

# Elenco ufficiale e ordinato delle 51 feature del modello
FEATURE_NAMES = [
    # 1. Medie annuali delle 6 bande grezze
    "Blu_B02",
    "Verde_B03",
    "Rosso_B04",
    "NIR_B08",
    "SWIR1_B11",
    "SWIR2_B12",
    # 2. Serie temporale NDVI mensile (12 mesi)
    *[f"NDVI_{m:02d}" for m in range(1, 13)],
    # 3. Serie temporale NDWI mensile (12 mesi)
    *[f"NDWI_{m:02d}" for m in range(1, 13)],
    # 4. Indicatori statistici di vigore vegetativo (NDVI)
    "NDVI_max",
    "NDVI_min",
    "NDVI_amp",
    "Peak_Month",
    "NDVI_mean",
    "NDVI_std",
    # 5. Differenziali stagionali e fenologici (NDVI)
    "NDVI_diff_lug_apr",
    "NDVI_diff_apr_gen",
    "NDVI_diff_set_lug",
    "Orzo_Wheat_Ratio",
    "NDVI_senescence_rate",
    # 6. Indicatori di contenuto idrico (NDWI)
    "NDWI_mean",
    "NDWI_diff_lug_gen",
    # 7. Rapporti spettrali SWIR / Struttura / Lignina
    "SWIR_NIR_ratio",
    "SWIR_Cellulose_ratio",
    # 8. Indicatori fenologici avanzati
    "NDVI_winter",
    "NDVI_summer_winter_diff",
    "NDVI_AUC",
    "Active_Months_Count",
    "Senescence_May_Apr",
    "Greenup_Mar_Feb",
]


def fill_missing_series(series):
    """
    Interpola i valori mancanti (zero o NaN) lungo l'asse temporale (asse 0).
    Equivalente a .interpolate().bfill().ffill() di pandas ma ottimizzato in NumPy.
    
    series: array di forma (12, ...)
    """
    filled = series.copy()
    n_months = filled.shape[0]
    
    # Maschera dei valori non validi (zero o NaN)
    invalid = np.isnan(filled) | (filled == 0.0)
    
    # 1. Forward-fill lungo i mesi
    for m in range(1, n_months):
        filled[m] = np.where(invalid[m], filled[m - 1], filled[m])
    
    # 2. Backward-fill lungo i mesi
    for m in range(n_months - 2, -1, -1):
        filled[m] = np.where(np.isnan(filled[m]) | (filled[m] == 0.0), filled[m + 1], filled[m])
        
    # Se rimangono pixel ancora NaN (tutti e 12 i mesi vuoti), sostituisce con 0
    filled = np.nan_to_num(filled, nan=0.0)
    return filled


def extract_features_from_monthly_stack(monthly_stack):
    """
    Estrae le 51 feature in modo vettorizzato NumPy a partire dallo stack dei 12 mesi.
    
    Parametri:
    -----------
    monthly_stack : np.ndarray
        Array NumPy di forma (12, 6, H, W) oppure (12, 6, N_pixel).
        - Asse 0: 12 Mesi (da Gennaio a Dicembre)
        - Asse 1: 6 Bande ordinate:
            0: Blu (B02)
            1: Verde (B03)
            2: Rosso (B04)
            3: NIR (B08)
            4: SWIR1 (B11)
            5: SWIR2 (B12)
        - Asse 2 e 3 (o solo 2): Dimensioni spaziali (H, W) o Pixel piatti.
        
    Ritorna:
    --------
    features_dict : dict
        Dizionario contenente ciascuna delle 51 feature calcolata spazialmente.
    features_matrix : np.ndarray
        Matrice 2D di forma (H*W, 51) oppure (N_pixel, 51), pronta per model.predict().
    """
    orig_shape = monthly_stack.shape
    # Normalizza la forma a (12, 6, N_pixel) per calcoli ultra-rapidi
    if len(orig_shape) == 4:
        n_months, n_bands, h, w = orig_shape
        flat_stack = monthly_stack.reshape(n_months, n_bands, -1).astype(np.float32)
        is_spatial = True
    elif len(orig_shape) == 3:
        n_months, n_bands, n_pixels = orig_shape
        flat_stack = monthly_stack.astype(np.float32)
        is_spatial = False
    else:
        raise ValueError("monthly_stack deve avere 4 dimensioni (12, 6, H, W) o 3 (12, 6, N_pixel)")

    if n_months != 12 or n_bands != 6:
        raise ValueError(f"Attesi 12 mesi e 6 bande, trovati: {n_months} mesi e {n_bands} bande.")

    n_pixels = flat_stack.shape[2]
    features_dict = {}

    # -------------------------------------------------------------
    # 1. MEDIE ANNUALI DI RIFLETTANZA PER LE 6 BANDE
    # -------------------------------------------------------------
    band_keys = ["Blu_B02", "Verde_B03", "Rosso_B04", "NIR_B08", "SWIR1_B11", "SWIR2_B12"]
    for b_idx, b_name in enumerate(band_keys):
        band_vals = flat_stack[:, b_idx, :]  # shape: (12, N_pixels)
        valid = band_vals > 0.0
        valid_counts = np.sum(valid, axis=0)
        sums = np.sum(np.where(valid, band_vals, 0.0), axis=0)
        mean_band = np.where(valid_counts > 0, sums / np.maximum(valid_counts, 1), 0.0)
        features_dict[b_name] = mean_band

    # -------------------------------------------------------------
    # 2. SERIE TEMPORALI MENSILI (NDVI e NDWI)
    # -------------------------------------------------------------
    red_series = flat_stack[:, 2, :]    # B04
    nir_series = flat_stack[:, 3, :]    # B08
    swir1_series = flat_stack[:, 4, :]  # B11
    swir2_series = flat_stack[:, 5, :]  # B12

    # NDVI: (NIR - Red) / (NIR + Red)
    ndvi_denom = nir_series + red_series
    raw_ndvi = np.where(
        (red_series > 0) & (nir_series > 0) & (ndvi_denom > 0),
        (nir_series - red_series) / np.maximum(ndvi_denom, 1e-6),
        0.0
    )
    ndvi_series = fill_missing_series(raw_ndvi)

    # NDWI: (NIR - SWIR1) / (NIR + SWIR1)
    ndwi_denom = nir_series + swir1_series
    raw_ndwi = np.where(
        (nir_series > 0) & (swir1_series > 0) & (ndwi_denom > 0),
        (nir_series - swir1_series) / np.maximum(ndwi_denom, 1e-6),
        0.0
    )
    ndwi_series = fill_missing_series(raw_ndwi)

    for m in range(1, 13):
        m_idx = m - 1
        features_dict[f"NDVI_{m:02d}"] = ndvi_series[m_idx]
        features_dict[f"NDWI_{m:02d}"] = ndwi_series[m_idx]

    # -------------------------------------------------------------
    # 3. INDICATORI FENOLOGICI E STATISTICI (NDVI)
    # -------------------------------------------------------------
    features_dict["NDVI_max"] = np.max(ndvi_series, axis=0)
    features_dict["NDVI_min"] = np.min(ndvi_series, axis=0)
    features_dict["NDVI_amp"] = features_dict["NDVI_max"] - features_dict["NDVI_min"]
    features_dict["Peak_Month"] = (np.argmax(ndvi_series, axis=0) + 1).astype(np.float32)
    features_dict["NDVI_mean"] = np.mean(ndvi_series, axis=0)
    features_dict["NDVI_std"] = np.std(ndvi_series, axis=0)

    # -------------------------------------------------------------
    # 4. DIFFERENZIALI STAGIONALI (NDVI)
    # ndvi_series indici 0..11 corrispondono a Gennaio..Dicembre
    # -------------------------------------------------------------
    features_dict["NDVI_diff_lug_apr"] = ndvi_series[6] - ndvi_series[3]   # Luglio (6) - Aprile (3)
    features_dict["NDVI_diff_apr_gen"] = ndvi_series[3] - ndvi_series[0]   # Aprile (3) - Gennaio (0)
    features_dict["NDVI_diff_set_lug"] = ndvi_series[8] - ndvi_series[6]   # Settembre (8) - Luglio (6)
    features_dict["Orzo_Wheat_Ratio"] = ndvi_series[3] / (ndvi_series[4] + 0.01) # Aprile / Maggio
    features_dict["NDVI_senescence_rate"] = ndvi_series[5] - ndvi_series[4] # Giugno (5) - Maggio (4)

    # -------------------------------------------------------------
    # 5. INDICATORI NDWI E RAPPORTI SWIR
    # -------------------------------------------------------------
    features_dict["NDWI_mean"] = np.mean(ndwi_series, axis=0)
    features_dict["NDWI_diff_lug_gen"] = ndwi_series[6] - ndwi_series[0]

    nir_b8 = features_dict["NIR_B08"]
    swir1_b11 = features_dict["SWIR1_B11"]
    swir2_b12 = features_dict["SWIR2_B12"]

    features_dict["SWIR_NIR_ratio"] = np.where(nir_b8 > 0, swir1_b11 / (nir_b8 + 0.001), 0.0)
    features_dict["SWIR_Cellulose_ratio"] = np.where(swir1_b11 > 0, swir2_b12 / (swir1_b11 + 0.001), 0.0)

    # -------------------------------------------------------------
    # 6. FENOLOGIA AVANZATA
    # -------------------------------------------------------------
    ndvi_winter = (ndvi_series[0] + ndvi_series[1] + ndvi_series[11]) / 3.0 # Gen + Feb + Dic
    ndvi_summer = (ndvi_series[6] + ndvi_series[7]) / 2.0                   # Lug + Ago
    features_dict["NDVI_winter"] = ndvi_winter
    features_dict["NDVI_summer_winter_diff"] = ndvi_summer - ndvi_winter
    features_dict["NDVI_AUC"] = np.sum(ndvi_series, axis=0)
    features_dict["Active_Months_Count"] = np.sum(ndvi_series > 0.35, axis=0).astype(np.float32)
    features_dict["Senescence_May_Apr"] = ndvi_series[4] - ndvi_series[3]   # Maggio (4) - Aprile (3)
    features_dict["Greenup_Mar_Feb"] = ndvi_series[2] - ndvi_series[1]       # Marzo (2) - Febbraio (1)

    # -------------------------------------------------------------
    # COSTRUZIONE MATRICE ORDINATA RIGOROSAMENTE SECONDO FEATURE_NAMES
    # -------------------------------------------------------------
    feature_columns = [features_dict[name] for name in FEATURE_NAMES]
    features_matrix = np.column_stack(feature_columns).astype(np.float32)

    return features_dict, features_matrix


def load_monthly_capitanata_rasters(folder_path, year=2023, max_size=None):
    """
    Carica i 12 file GeoTIFF mensili della Capitanata dalla cartella specificata.
    Opzionalmente ridimensiona (ricampiona) per contenere l'uso della RAM.
    
    Parametri:
    -----------
    folder_path : str or Path
        Cartella contenente i file 'capitanata_{year}_{01..12}.tif'
    year : int or str
        Anno di interesse (default 2023)
    max_size : int, optional
        Se specificato, ricampiona l'immagine in modo che la dimensione massima (H o W) sia max_size.
        Utile per testare la predizione senza saturare la RAM.
        
    Ritorna:
    --------
    monthly_stack : np.ndarray
        Array di forma (12, 6, H, W)
    profile : dict
        Metadati rasterio del primo file utile (per georeferenziazione o export)
    """
    folder = Path(folder_path)
    monthly_arrays = []
    base_profile = None

    for m in range(1, 13):
        candidates = list(folder.glob(f"*{year}*_{m:02d}.tif"))
        if not candidates:
            # Fallback generico
            candidates = list(folder.glob(f"*_{m:02d}.tif"))
        
        if not candidates:
            raise FileNotFoundError(f"Impossibile trovare il file TIF per il mese {m:02d} in {folder}")

        tif_path = candidates[0]
        with rasterio.open(tif_path) as src:
            if base_profile is None:
                base_profile = src.profile.copy()
            
            if max_size is not None:
                orig_h, orig_w = src.shape
                scale = max_size / max(orig_h, orig_w)
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)
                data = src.read(
                    out_shape=(src.count, new_h, new_w),
                    resampling=Resampling.bilinear
                )
            else:
                data = src.read()

            if data.shape[0] != 6:
                raise ValueError(f"Il file {tif_path.name} ha {data.shape[0]} bande (attese 6).")

            monthly_arrays.append(data)

    monthly_stack = np.stack(monthly_arrays, axis=0) # Forma: (12, 6, H, W)
    return monthly_stack, base_profile
