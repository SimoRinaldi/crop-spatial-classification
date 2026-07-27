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
#     display_name: env (3.13.5.final.0)
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
import rioxarray

tifs_4326 = {}

for key, value in tifs_3035.items():
    list = []
    for v in value:
        file_name = v.replace("03035", "4326")
        file_path = Path(file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File GeoTIFF originale (EPSG:3035)
        raster = rioxarray.open_rasterio(v)
        # Riproiezione dell'intero raster in EPSG:4326
        raster_4326 = raster.rio.reproject("EPSG:4326")
        
        raster_4326.rio.to_raster(file_name)
        
        list.append(file_name)
        
    tifs_4326[key] = list

# %%
print(tifs_4326)
