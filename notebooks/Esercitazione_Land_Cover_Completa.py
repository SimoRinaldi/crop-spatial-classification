# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv (3.12.3)
#     language: python
#     name: python3
# ---

# %% [markdown] id="6ca839ba"
# # Esercitazione: Classificazione Land Cover con Machine Learning
#
# **Obiettivo:** Costruire un modello di classificazione supervisionata multiclasse (Acqua, Foresta, Suolo) trasformando dati spaziali in formato tabellare.
#
# Area di studio: **Brescia e Iseo (Test)**.
#
# ### **Sommario delle attività**
# 1. **Download STAC API**: Scaricare porzioni di Sentinel-2 per l'area di addestramento e di test.
# 2. **Data Preparation**: Estrarre pixel spaziali da GeoTIFF, incrociarli con le etichette fornite e convertire in tabular data (`pandas`).
# 3. **Machine Learning Supervisionato**: Addestrare `RandomForestClassifier` (o altro) e generare mappe predittive su nuovi dati.
# 4. **Machine Learning Non Supervisionato**: Confrontare i risultati con clustering `KMeans`.
# 5. **Esportazione GIS**: Salvare la mappa finale in formato GeoTIFF compatibile con QGIS/ArcGIS.

# %% colab={"base_uri": "https://localhost:8080/"} id="YSciXAR7VU_1" executionInfo={"status": "ok", "timestamp": 1780472936967, "user_tz": -120, "elapsed": 22729, "user": {"displayName": "Ivan SERINA", "userId": "13730370536643547392"}} outputId="1909b6bf-2c3f-4698-d385-18ca490526d1"
# SETUP INIZIALE
# Installa le librerie gis e di machine learning necessarie
# !pip install rasterio rioxarray pystac-client scikit-learn matplotlib pandas numpy xarray

# %% id="07397f8c"
import os
import rasterio
import rioxarray
import xarray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pystac_client import Client
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')

# Configurazione GDAL/AWS per scaricare velocemente Sentinel-2
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"

# %% [markdown] id="b724f68d"
# ### 0. Download Dati (Tramite STAC API)
# Scarichiamo due immagini satellitari (Train e Test) da AWS Earth Search usando Bounding Box (BBox).

# %% colab={"base_uri": "https://localhost:8080/"} id="2f700357" executionInfo={"status": "ok", "timestamp": 1780472958639, "user_tz": -120, "elapsed": 7953, "user": {"displayName": "Ivan SERINA", "userId": "13730370536643547392"}} outputId="6e7e86e6-1ed3-425d-e484-48c426aca6c6"
STAC_API_URL = "https://earth-search.aws.element84.com/v1"
client = Client.open(STAC_API_URL)

PUNTI = {
    "Train_Brescia": {"lon": 10.27, "lat": 45.50},
    "Test_Iseo": {"lon": 10.05, "lat": 45.65}
}

BUFFER_DEG = 0.05 # Dimensione ritaglio
BANDS = ["blue", "green", "red", "nir"]
YEAR = "2023"

# Funzione per scaricare e fondere le bande
def download_point(name, lon, lat):
    print(f"Scaricando {name}...")
    file_out = f"{name}.tif"
    if os.path.exists(file_out):
        print("File già presente.")
        return file_out

    bbox = [lon - BUFFER_DEG, lat - BUFFER_DEG, lon + BUFFER_DEG, lat + BUFFER_DEG]
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{YEAR}-06-01/{YEAR}-08-31", # Estate per massima vegetazione
        query={"eo:cloud_cover": {"lt": 10}}
    )
    items = list(search.items())
    best_item = sorted(items, key=lambda x: x.properties["eo:cloud_cover"])[0]

    # Scarica e concatena bande (Blu, Verde, Rosso, NIR)
    band_arrays = []
    for b in BANDS:
        href = best_item.assets[b].href
        ds = rioxarray.open_rasterio(href)
        cropped = ds.rio.clip_box(*bbox, crs="EPSG:4326")
        band_arrays.append(cropped)

    # Merge nel singolo TIF
    merged = xarray.concat(band_arrays, dim="band")
    merged.rio.to_raster(file_out, compress="deflate")
    print(f"Salvato {file_out}")
    return file_out

# Esegui download
train_img_path = download_point("Train_Brescia", PUNTI["Train_Brescia"]["lon"], PUNTI["Train_Brescia"]["lat"])
test_img_path = download_point("Test_Iseo", PUNTI["Test_Iseo"]["lon"], PUNTI["Test_Iseo"]["lat"])

# %% [markdown] id="69149f89"
# ---
# #### Fase 1: Esplorazione Visiva

# %% colab={"base_uri": "https://localhost:8080/", "height": 692} id="cf28d5fe" executionInfo={"status": "ok", "timestamp": 1780472962439, "user_tz": -120, "elapsed": 3126, "user": {"displayName": "Ivan SERINA", "userId": "13730370536643547392"}} outputId="a8d20079-fdb3-43fd-c55d-35d151d4aa7d"
# Leggiamo l'immagine di training
src_train = rasterio.open(train_img_path)
img_train = src_train.read()

print("Shape immagine (Bande, Altezza, Larghezza):", img_train.shape)

# Plot RGB (Bande 3, 2, 1 che corrispondono a R, G, B nel nostro cubo 4-bande)
# Rasterio ordina 1-based: [Blu, Verde, Rosso, NIR] -> [2, 1, 0] per RGB
rgb_img = np.dstack((img_train[2], img_train[1], img_train[0]))

# Normalizzazione per visualizzazione
rgb_img = rgb_img / np.percentile(rgb_img, 98)
rgb_img = np.clip(rgb_img, 0, 1)

plt.figure(figsize=(8,8))
plt.imshow(rgb_img)
plt.title("Immagine di Training (RGB)")
plt.axis('off')
plt.show()

# %% id="zoQAt2JH1kBZ"
# Leggiamo l'immagine di training
src_train = rasterio.open(train_img_path)
img_train = src_train.read()

print("Shape immagine (Bande, Altezza, Larghezza):", img_train.shape)

# Plot RGB (Bande 3, 2, 1 che corrispondono a R, G, B nel nostro cubo 4-bande)
# Rasterio ordina 1-based: [Blu, Verde, Rosso, NIR] -> [2, 1, 0] per RGB
rgb_img = np.dstack((img_train[2], img_train[1], img_train[0]))

# Normalizzazione per visualizzazione
rgb_img = rgb_img / np.percentile(rgb_img, 98)
rgb_img = np.clip(rgb_img, 0, 1)

plt.figure(figsize=(8,8))
plt.imshow(rgb_img)
plt.title("Immagine di Training (RGB)")
plt.axis('off')
plt.show()

# %% [markdown] id="c52a002e"
# ---
# #### Fase 2: Costruzione del Dataset

# %% id="8a83ed65"
df = pd.read_csv("training_points.csv")

# 0 > Acqua
# 1 > Foresta/Erba
# 2 > Urbano/Suolo non coltivato

# Estraiamo le firme spettrali (valori pixel) per ogni coordinata (Row, Col)
features = []
for index, row in df.iterrows():
    # img_train ha forma (4, H, W)
    r = int(row['Row'])
    c = int(row['Col'])
    pixel_values = img_train[:, r, c]
    features.append(pixel_values)

# Aggiungiamo le feature al DataFrame
features_df = pd.DataFrame(features, columns=['blue', 'green', 'red', 'nir'])
dataset = pd.concat([features_df, df['Class']], axis=1)

print("Distribuzione classi:")
print(dataset['Class'].value_counts().rename(index={0: 'Acqua (0)', 1: 'Foresta/Erba (1)', 2: 'Suolo non coltivato/Urbano (2)'}))
print("\nAnteprima dataset:")
print(dataset.head())

# %% [markdown] id="3dca7cba"
# ---
# #### Fase 3: Addestramento del Modello

# %% colab={"base_uri": "https://localhost:8080/"} id="7a8a3443" executionInfo={"status": "ok", "timestamp": 1780472962487, "user_tz": -120, "elapsed": 12, "user": {"displayName": "Ivan SERINA", "userId": "13730370536643547392"}} outputId="8abae50a-477e-440d-d0cc-a545f69907b8"
X = dataset[['blue', 'green', 'red', 'nir']]
y = dataset['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Random Forest non ha bisogno di feature scaling
model = RandomForestClassifier(n_estimators=1, random_state=42)
model.fit(X_train, y_train)

# Valutazione
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Acqua', 'Foresta', 'Suolo/Urbano']))


# %% [markdown] id="abb6506e"
# ---
# #### Fase 4: Inferenza Spaziale
# Flattening dell'immagine -> Predizione -> Reshape mappa

# %% colab={"base_uri": "https://localhost:8080/"} id="45cf3874" executionInfo={"status": "ok", "timestamp": 1780472962502, "user_tz": -120, "elapsed": 12, "user": {"displayName": "Ivan SERINA", "userId": "13730370536643547392"}} outputId="9cfb19e7-8473-425a-b2a7-67263195508f"
def elabora_e_classifica(tif_path, model):
    with rasterio.open(tif_path) as src:
        img_data = src.read()
        n_bands, h, w = img_data.shape

        # Flattening: (4, H, W) -> (H*W, 4)
        flat_img = img_data.reshape(n_bands, -1).T

        # Generiamo le predizioni
        print(f"Predizione su {h*w} pixel per {tif_path}...")
        pred_flat = model.predict(flat_img)

        # Reshape: (H*W) -> (H, W)
        pred_map = pred_flat.reshape(h, w)
        return pred_map

map_train = elabora_e_classifica(train_img_path, model)
map_test = elabora_e_classifica(test_img_path, model)

# %% colab={"base_uri": "https://localhost:8080/", "height": 598} id="4b8716d9" executionInfo={"status": "ok", "timestamp": 1780472963091, "user_tz": -120, "elapsed": 586, "user": {"displayName": "Ivan SERINA", "userId": "13730370536643547392"}} outputId="63f3cf9f-8152-46aa-83ad-2b8c9396c35c"
# Plot Risultati Custom
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['blue', 'forestgreen', 'saddlebrown'])

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# TIF 1: Train
axes[0].imshow(map_train, cmap=cmap)
axes[0].set_title('Predizione Brescia (Train Area)')
axes[0].axis('off')

# TIF 2: Test
axes[1].imshow(map_test, cmap=cmap)
axes[1].set_title('Predizione Iseo (Test Area)')
axes[1].axis('off')

plt.show()

# %% [markdown] id="5c304f4d"
# ---
# #### Fase 5: Clustering & Esportazione
# Applichiamo `KMeans` senza etichette per raggruppare i pixel in 3 cluster. Poi esportiamo il risultato come nuovo `.tif`.

# %% colab={"base_uri": "https://localhost:8080/", "height": 667} id="e705f2fa" executionInfo={"status": "ok", "timestamp": 1780472975307, "user_tz": -120, "elapsed": 12210, "user": {"displayName": "Ivan SERINA", "userId": "13730370536643547392"}} outputId="620380e7-1621-4985-f389-1e4681ae836a"
from sklearn.cluster import KMeans

print("Esecuzione KMeans su area di Test...")

# Leggiamo immagine di test
with rasterio.open(test_img_path) as src:
    test_data = src.read()
    h = test_data.shape[1]
    w = test_data.shape[2]

    # Flattening per KMeans
    test_flat = test_data.reshape(4, -1).T

# Training & Inferenza Non Supervisionata
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_flat = kmeans.fit_predict(test_flat)

# Reshape mappa cluster
map_cluster = clusters_flat.reshape(h, w)

# 1. PLOT CONFRONTO
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].imshow(map_test, cmap=cmap)
axes[0].set_title('Supervisionato (Random Forest)')
axes[0].axis('off')

# I colori di KMeans sono arbitrari (non mappati a classi specifiche)
axes[1].imshow(map_cluster, cmap='viridis')
axes[1].set_title('Non Supervisionato (KMeans)')
axes[1].axis('off')
plt.show()

# 2. ESPORTAZIONE GEOTIFF
out_tif = 'iseo_kmeans_clusters.tif'
print(f"Salvataggio mappa clustering in {out_tif}...")

with rasterio.open(test_img_path) as src:
    profile = src.profile

    # Aggiorniamo info profilo (1 banda, intero a 8 bit)
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='deflate'
    )

    # Scrittura nuovo file
    with rasterio.open(out_tif, 'w', **profile) as dst:
        dst.write(map_cluster.astype(rasterio.uint8), 1)

print("Esportazione completata!")
