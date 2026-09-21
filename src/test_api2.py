import os
import calendar
from pathlib import Path
import rioxarray
from rioxarray.merge import merge_arrays
import xarray
from pystac_client import Client

# Accelerazioni e accesso pubblico per bucket AWS S3
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"

STAC_API_URL = "https://earth-search.aws.element84.com/v1"
client = Client.open(STAC_API_URL)

# Le 6 bande agronomiche Sentinel-2 L2A (10m e 20m)
BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]


def download_area_month(
    name, bbox, year, month, out_dir="./data/processed/sentinel2_capitanata_area"
):
    """Scarica e fonde le bande di un mese per l'area specificata."""
    os.makedirs(out_dir, exist_ok=True)
    file_out = os.path.join(out_dir, f"{name}_{year}_{month:02d}.tif")

    if os.path.exists(file_out) and os.path.getsize(file_out) > 50_000_000:
        print(f"Mese {month:02d} già presente: {file_out}")
        return file_out

    print(f"Scaricando {name} per {year}-{month:02d}...")
    last_day = calendar.monthrange(int(year), int(month))[1]

    # Ricerca scene limpide nel mese
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day:02d}",
        query={"eo:cloud_cover": {"lt": 30}},
    )
    items = list(search.items())

    # Se non trova nulla sotto il 30% di nuvole, rilassa la soglia
    if not items:
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day:02d}",
            query={"eo:cloud_cover": {"lt": 70}},
        )
        items = list(search.items())

    if not items:
        print(f"⚠️ Nessuna immagine trovata per {year}-{month:02d}")
        return None

    # Raggruppa le tile per data (giorno di passaggio) e prende la data più limpida
    items_by_date = {}
    for it in items:
        d = it.properties.get("datetime", "")[:10]
        items_by_date.setdefault(d, []).append(it)

    best_date = min(
        items_by_date.keys(),
        key=lambda d: sum(
            x.properties.get("eo:cloud_cover", 100) for x in items_by_date[d]
        )
        / len(items_by_date[d]),
    )
    day_items = items_by_date[best_date]
    print(f"  • Data selezionata: {best_date} ({len(day_items)} scene)")

    # Scarica, ritaglia e allinea ciascuna banda
    band_arrays = []
    for b in BANDS:
        tile_crops = []
        for it in day_items:
            if b in it.assets:
                href = it.assets[b].href
                ds = rioxarray.open_rasterio(href)
                cropped = ds.rio.clip_box(*bbox, crs="EPSG:4326")
                if "band" in cropped.dims and cropped.sizes["band"] == 1:
                    cropped = cropped.squeeze("band", drop=True)
                tile_crops.append(cropped)

        if not tile_crops:
            continue

        # Unisce le tile adiacenti (mosaico) se l'area attraversa 2 scene
        merged_band = merge_arrays(tile_crops) if len(tile_crops) > 1 else tile_crops[0]

        # Ricampiona le bande a 20m (swir16, swir22) sulla griglia della prima banda a 10m (blue)
        if band_arrays and (
            merged_band.rio.shape != band_arrays[0].rio.shape
            or merged_band.rio.crs != band_arrays[0].rio.crs
        ):
            merged_band = merged_band.rio.reproject_match(band_arrays[0])

        band_arrays.append(merged_band)

    if not band_arrays:
        print(f"⚠️ Nessuna banda scaricata per {year}-{month:02d}")
        return None

    # Merge delle 6 bande nel singolo TIF multibanda
    merged = xarray.concat(band_arrays, dim="band")
    merged.coords["band"] = list(range(1, len(band_arrays) + 1))
    if band_arrays[0].rio.crs:
        merged.rio.write_crs(band_arrays[0].rio.crs, inplace=True)

    merged.rio.to_raster(file_out, compress="deflate", predictor=2, tiled=True)
    print(f"✅ Salvato {file_out}")
    return file_out


def download_area(
    name,
    bbox,
    year="2023",
    months=range(1, 13),
    out_dir="./data/processed/sentinel2_capitanata_area",
):
    """Scarica tutti i mesi richiesti per l'area data."""
    print(f"Inizio download area '{name}' per l'anno {year}...")
    downloaded = []
    for m in months:
        f = download_area_month(name, bbox, year, m, out_dir=out_dir)
        if f:
            downloaded.append(f)
    print(f"Finito! Scaricati {len(downloaded)}/{len(list(months))} file.")
    return downloaded
