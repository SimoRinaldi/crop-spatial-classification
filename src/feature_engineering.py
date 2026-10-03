# Script per il calcolo delle feature.

from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling

# elenco ordinato e dettagliato delle feature
FEATURE_LIST = [
    # Colori e bande satellitari (medie di tutto l'anno)
    "Blu_B02",  # riflettanza della luce blu
    "Verde_B03",  # riflettanza della luce verde
    "Rosso_B04",  # riflettanza della luce rossa (assorbita dalle piante vive)
    "NIR_B08",  # vicino infrarosso (le foglie sane e dense ne riflettono tantissimo)
    "SWIR1_B11",  # infrarosso a onde corte 1 (sensibile all'umidità del suolo e delle foglie)
    "SWIR2_B12",  # infrarosso a onde corte 2 (sensibile a residui secchi, rami e cellulosa)
    # Vigore vegetativo mese per mese (ndvi da gennaio a dicembre)
    # l'ndvi misura quanto la pianta è verde e rigogliosa (vicino a 1 = molto verde, vicino a 0 = terra nuda)
    "NDVI_01",
    "NDVI_02",
    "NDVI_03",
    "NDVI_04",
    "NDVI_05",
    "NDVI_06",
    "NDVI_07",
    "NDVI_08",
    "NDVI_09",
    "NDVI_10",
    "NDVI_11",
    "NDVI_12",
    # Contenuto d'acqua mese per mese (ndwi da gennaio a dicembre)
    # l'ndwi misura il contenuto idrico delle foglie (più è alto, più la pianta è idratata)
    "NDWI_01",
    "NDWI_02",
    "NDWI_03",
    "NDWI_04",
    "NDWI_05",
    "NDWI_06",
    "NDWI_07",
    "NDWI_08",
    "NDWI_09",
    "NDWI_10",
    "NDWI_11",
    "NDWI_12",
    # Statistiche annuali del vigore verde (ndvi)
    "NDVI_max",  # il massimo picco di verde raggiunto durante l'anno
    "NDVI_min",  # il minimo di verde nell'anno (inverno o dopo il raccolto)
    "NDVI_amp",  # escursione annuale (differenza tra massimo e minimo vigore)
    "Peak_Month",  # il mese dell'anno (1-12) in cui la pianta è stata al massimo splendore
    "NDVI_mean",  # media del vigore sull'anno (alta per sempreverdi come ulivi, bassa per colture brevi)
    "NDVI_std",  # quanto varia il vigore nel tempo (stabile per boschi/ulivi, altalenante per cereali)
    # Differenze tra mesi chiave (aiutano a distinguere le colture)
    "NDVI_diff_lug_apr",  # differenza luglio-aprile (il grano a luglio è secco/raccolto, il pomodoro è verde)
    "NDVI_diff_apr_gen",  # crescita primaverile (differenza tra aprile e gennaio)
    "NDVI_diff_set_lug",  # comportamento a fine estate (differenza tra settembre e luglio)
    "Orzo_Wheat_Ratio",  # confronto tra aprile e maggio (l'orzo matura e ingiallisce prima del grano duro)
    "NDVI_senescence_rate",  # velocità con cui la pianta si secca tra maggio e giugno
    # Statistiche sull'idratazione (ndwi)
    "NDWI_mean",  # idratazione media durante tutto l'anno
    "NDWI_diff_lug_gen",  # disidratazione estiva rispetto all'inverno
    # Struttura della pianta e residui secchi
    "SWIR_NIR_ratio",  # rapporto tra secchezza e vigore fogliare
    "SWIR_Cellulose_ratio",  # presenza di parti legnose o paglia (distingue alberi da piante erbacee)
    # Indicatori del ciclo di vita (fenologia)
    "NDVI_winter",  # vigore medio invernale (gennaio, febbraio, dicembre)
    "NDVI_summer_winter_diff",  # contrasto tra estate e inverno
    "NDVI_AUC",  # biomassa totale prodotta nell'anno (area totale sotto la curva di crescita)
    "Active_Months_Count",  # quanti mesi all'anno il campo è rimasto effettivamente verde
    "Senescence_May_Apr",  # ingiallimento tra aprile e maggio
    "Greenup_Mar_Feb",  # risveglio vegetativo tra febbraio e marzo
]


def fill_missing_series(series):
    """
    Interpola i valori mancanti (zero o nan) lungo l'asse temporale dei 12 mesi
    """
    filled = series.copy()
    n_months = filled.shape[0]

    def is_missing(data):
        return np.isnan(data) | (data == 0.0)

    # fill-forward
    # se il mese corrente è vuoto, copia il valore del mese precedente
    for m in range(1, n_months):
        is_empty = is_missing(filled[m])
        filled[m] = np.where(is_empty, filled[m - 1], filled[m])

    # fill-backward
    # se il mese corrente è vuoto, copia il valore del mese successivo
    for m in range(n_months - 2, -1, -1):
        is_empty = is_missing(filled[m])
        filled[m] = np.where(is_empty, filled[m + 1], filled[m])

    # sostituisce eventuali pixel sempre vuoti tutto l'anno con zero
    return np.nan_to_num(filled, nan=0.0)


def extract_features_from_monthly_stack(monthly_stack):
    """
    Estrae le feature in modo vettorizzato NumPy a partire dallo stack dei 12 mesi.
    Prende i 12 mesi di immagini satellitari e calcola tutte le feature per ciascun pixel in parallelo

    Input: np.ndarray
        array NumPy di forma (12, 6, h, w) oppure (12, 6, n_points).
        - indice 0: 12 mesi
        - indice 1: 6 bande ordinate:
            0: Blu (B02)
            1: Verde (B03)
            2: Rosso (B04)
            3: NIR (B08)
            4: SWIR1 (B11)
            5: SWIR2 (B12)
        - indici 2 e/o 3: dimensioni (h, w) o pixel.

    Output: dict, np.ndarray
        - dizionario contenente tutte le feature calcolate.
        - matrice 2D di forma (h*w, n_feature) oppure (n_points, n_feature).
    """
    orig_shape = monthly_stack.shape
    # normalizza la forma a (12, 6, n_points) per calcoli rapidi
    if len(orig_shape) == 4:
        n_months, n_bands = orig_shape[:2]
        flat_stack = monthly_stack.reshape(n_months, n_bands, -1).astype(np.float32)
    elif len(orig_shape) == 3:
        n_months, n_bands = orig_shape[:2]
        flat_stack = monthly_stack.astype(np.float32)
    else:
        raise ValueError(
            "Il parametro di input deve avere 4 dimensioni (12, 6, h, w) o 3 (12, 6, n_points)"
        )

    if n_months != 12 or n_bands != 6:
        raise ValueError(
            f"Attesi 12 mesi e 6 bande, trovati: {n_months} mesi e {n_bands} bande."
        )

    features_dict = {}

    # ------------------------------------------------------
    # Calcolo del valore medio annuale per ciascuna banda,
    # escludendo eventuali valori mancanti o pari a zero.
    band_list = [
        "Blu_B02",
        "Verde_B03",
        "Rosso_B04",
        "NIR_B08",
        "SWIR1_B11",
        "SWIR2_B12",
    ]
    for b_idx, b_name in enumerate(band_list):
        band_values = flat_stack[:, b_idx, :]
        valid = band_values > 0.0
        valid_counts = np.sum(valid, axis=0)
        sums = np.sum(np.where(valid, band_values, 0.0), axis=0)
        mean_band = np.where(valid_counts > 0, sums / np.maximum(valid_counts, 1), 0.0)
        features_dict[b_name] = mean_band

    # ------------------------------------------------------
    # NDVI = (NIR - Rosso) / (NIR + Rosso)
    # NDWI = (NIR - SWIR1) / (NIR + SWIR1)

    red_series = flat_stack[:, 2, :]  # B04
    nir_series = flat_stack[:, 3, :]  # B08
    swir1_series = flat_stack[:, 4, :]  # B11

    # NDVI mensile
    ndvi_denom = nir_series + red_series
    raw_ndvi = np.where(
        (red_series > 0) & (nir_series > 0) & (ndvi_denom > 0),
        (nir_series - red_series) / np.maximum(ndvi_denom, 1e-6),
        0.0,
    )
    ndvi_series = fill_missing_series(raw_ndvi)

    # NDWI mensile
    ndwi_denom = nir_series + swir1_series
    raw_ndwi = np.where(
        (nir_series > 0) & (swir1_series > 0) & (ndwi_denom > 0),
        (nir_series - swir1_series) / np.maximum(ndwi_denom, 1e-6),
        0.0,
    )
    ndwi_series = fill_missing_series(raw_ndwi)

    for m in range(1, 13):
        features_dict[f"NDVI_{m:02d}"] = ndvi_series[m - 1]
        features_dict[f"NDWI_{m:02d}"] = ndwi_series[m - 1]

    # -------------------------------------------------------------
    # Statistiche annuali del vigore vegetativo (NDVI)

    features_dict["NDVI_max"] = np.max(ndvi_series, axis=0)
    features_dict["NDVI_min"] = np.min(ndvi_series, axis=0)
    features_dict["NDVI_amp"] = features_dict["NDVI_max"] - features_dict["NDVI_min"]
    features_dict["Peak_Month"] = (np.argmax(ndvi_series, axis=0) + 1).astype(
        np.float32
    )
    features_dict["NDVI_mean"] = np.mean(ndvi_series, axis=0)
    features_dict["NDVI_std"] = np.std(ndvi_series, axis=0)

    # -------------------------------------------------------------
    # Differenziali stagionali tra mesi

    features_dict["NDVI_diff_lug_apr"] = ndvi_series[6] - ndvi_series[3]
    features_dict["NDVI_diff_apr_gen"] = ndvi_series[3] - ndvi_series[0]
    features_dict["NDVI_diff_set_lug"] = ndvi_series[8] - ndvi_series[6]

    denom_orzo = np.where(
        np.abs(ndvi_series[4] + 0.01) < 1e-5, 1e-5, ndvi_series[4] + 0.01
    )
    features_dict["Orzo_Wheat_Ratio"] = np.clip(
        ndvi_series[3] / denom_orzo, -50.0, 50.0
    )
    features_dict["NDVI_senescence_rate"] = ndvi_series[5] - ndvi_series[4]

    # -------------------------------------------------------------
    # Statistiche NDWI SWIR
    features_dict["NDWI_mean"] = np.mean(ndwi_series, axis=0)
    features_dict["NDWI_diff_lug_gen"] = ndwi_series[6] - ndwi_series[0]

    nir_b8 = features_dict["NIR_B08"]
    swir1_b11 = features_dict["SWIR1_B11"]
    swir2_b12 = features_dict["SWIR2_B12"]

    features_dict["SWIR_NIR_ratio"] = np.where(
        nir_b8 > 0, np.clip(swir1_b11 / (nir_b8 + 0.001), 0.0, 50.0), 0.0
    )
    features_dict["SWIR_Cellulose_ratio"] = np.where(
        swir1_b11 > 0, np.clip(swir2_b12 / (swir1_b11 + 0.001), 0.0, 50.0), 0.0
    )

    # -------------------------------------------------------------
    # Indicatori avanzati del ciclo di vita della pianta

    ndvi_winter = (ndvi_series[0] + ndvi_series[1] + ndvi_series[11]) / 3.0
    ndvi_summer = (ndvi_series[6] + ndvi_series[7]) / 2.0
    features_dict["NDVI_winter"] = ndvi_winter
    features_dict["NDVI_summer_winter_diff"] = ndvi_summer - ndvi_winter
    features_dict["NDVI_AUC"] = np.sum(ndvi_series, axis=0)
    features_dict["Active_Months_Count"] = np.sum(ndvi_series > 0.35, axis=0).astype(
        np.float32
    )
    features_dict["Senescence_May_Apr"] = ndvi_series[4] - ndvi_series[3]
    features_dict["Greenup_Mar_Feb"] = ndvi_series[2] - ndvi_series[1]

    # -------------------------------------------------------------
    # Costruzione della matrice

    feature_columns = [features_dict[name] for name in FEATURE_LIST]
    features_matrix = np.column_stack(feature_columns).astype(np.float32)

    # pulizia finale di eventuali NaN, inf e overflow
    features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    features_matrix = np.clip(features_matrix, -1e5, 1e5).astype(np.float32)

    return features_dict, features_matrix


def load_monthly_capitanata_rasters(directory_path, year=2023, max_size=None):
    """
    Carica i 12 file GeoTIFF mensili della Capitanata e li impila insieme in un unico
    blocco 4D di forma (12 mesi, 6 bande, altezza, larghezza)

    Input: str or Path, int or str, int or optional
        - cartella contenente i file 'capitanata_{year}_{01..12}.tif'
        - anno di interesse (default 2023)
        - se specificato, ricampiona l'immagine in modo che la dimensione massima (h o w) sia max_size.
          (utile per testare la predizione senza saturare la RAM).

    Output: np.ndarray, dict
        - array di forma (12, 6, h, w)
        - metadati rasterio del primo file utile
    """
    directory = Path(directory_path)
    monthly_arrays = []
    tif_profile = None

    for m in range(1, 13):
        candidates = list(directory.glob(f"*{year}*_{m:02d}.tif"))

        if not candidates:
            raise FileNotFoundError(
                f"Impossibile trovare il file TIF per il mese {m:02d} in {directory}"
            )

        tif_path = candidates[0]
        with rasterio.open(tif_path) as src:
            if tif_profile is None:
                tif_profile = src.profile.copy()  # salva metadati (coordinate, crs...)

            orig_h, orig_w = src.shape
            if max_size is not None and max(orig_h, orig_w) > max_size:
                # calcola le nuove dimensioni proporzionate
                scale = max_size / max(orig_h, orig_w)
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)
                data = src.read(
                    out_shape=(src.count, new_h, new_w), resampling=Resampling.bilinear
                )
            else:
                data = src.read()

            if data.shape[0] != 6:
                raise ValueError(
                    f"Il file {tif_path.name} ha {data.shape[0]} bande (attese 6)."
                )

            monthly_arrays.append(data)

    monthly_stack = np.stack(monthly_arrays, axis=0)
    return monthly_stack, tif_profile
