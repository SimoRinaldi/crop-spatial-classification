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
#     display_name: .venv (3.14.6.final.0)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

main_directory = Path("../data/crops_types_yearly_capitanata_03035")

tifs_3035 = {}

for d in main_directory.iterdir():
    if d.is_dir():
        year = d.name
        
        lista_tifs = list(d.rglob("*.tif"))
        
        tifs_3035[year] = [str(tif) for tif in lista_tifs]

# %%
from pathlib import Path
import rioxarray

tifs_4326 = {}

for key, value in tifs_3035.items():
    year_file_list = []
    for v in value:
        file_name = v.replace("03035", "4326")
        file_path = Path(file_name)
        
        # Se il file riproiettato esiste già, salta la riproiezione
        if file_path.is_file():
            year_file_list.append(file_name)
            continue
        
        # Crea la cartella di destinazione se non esiste
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File GeoTIFF originale (EPSG:3035)
        raster = rioxarray.open_rasterio(v)
        # Riproiezione dell'intero raster in EPSG:4326
        raster_4326 = raster.rio.reproject("EPSG:4326")
        
        raster_4326.rio.to_raster(file_name)
        
        year_file_list.append(file_name)
        
    tifs_4326[key] = year_file_list

# %%
from pathlib import Path
import rasterio
import numpy as np

MAX_SAMPLE_NUMBER = 100

points = []

# Prende la lista di file dell'ultimo gruppo
last_file_list = list(tifs_4326.values())[-1]

for tif in last_file_list:
    with rasterio.open(tif) as dataset:
        full_map = dataset.read(1)

        valid_mask = (full_map > 0) & (full_map < 65534)
        rows, cols = np.where(valid_mask)

        print(full_map.shape)

        indices = np.random.choice(len(rows), size=min(MAX_SAMPLE_NUMBER, len(rows)), replace=False)

        for index in indices:
            lat, lon = dataset.xy(cols[index], rows[index])
            points.append({
                "lat": lat.item(),
                "lon": lon.item(),
                "code": full_map[rows[index], cols[index]].item()
            })

# %%
import pandas as pd
df = pd.read_json('../data/points.json')
conteggio = df['code'].value_counts()
print(conteggio)

# %%
import json
from pathlib import Path

# Percorso del file JSON nella cartella di lavoro
json_path = Path("../data/points.json")

# Salva i punti (formato: [["nome_file", lon, lat], ...])
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(points, f, indent=4, ensure_ascii=False)

print(f"✅ Salvati {len(points)} punti in {json_path.resolve()}")
