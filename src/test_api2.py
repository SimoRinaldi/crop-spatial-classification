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
    """
    Scarica e fonde le bande di un mese per l'area specificata.
    Per garantire la copertura completa dell'area senza buchi, seleziona
    la scena più limpida del mese per ciascuna tile MGRS che interseca la BBox.
    """
    os.makedirs(out_dir, exist_ok=True)
    file_out = os.path.join(out_dir, f"{name}_{year}_{month:02d}.tif")

    # Resume: se il file è già completo (> 200 MB), salta
    if os.path.exists(file_out) and os.path.getsize(file_out) > 200_000_000:
        print(
            f"Mese {month:02d} già presente e completo: {file_out} ({os.path.getsize(file_out)/(1024*1024):.1f} MB)"
        )
        return file_out

    print(f"\nScaricando {name} per {year}-{month:02d}...")
    last_day = calendar.monthrange(int(year), int(month))[1]

    # 1. Ricerca scene nel mese
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day:02d}",
        query={"eo:cloud_cover": {"lt": 40}},
    )
    items = list(search.items())

    # Se troppe nuvole, rilassa la soglia per garantire la presenza di tutte le tile
    if len(items) < 3:
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day:02d}",
            query={"eo:cloud_cover": {"lt": 80}},
        )
        items = list(search.items())

    if not items:
        print(f"⚠️ Nessuna immagine trovata per {year}-{month:02d}")
        return None

    # 2. Per ciascuna tile MGRS dell'area, seleziona la scena con minor copertura nuvolosa del mese
    best_items_by_tile = {}
    for it in items:
        tile_key = (
            it.properties.get("grid:code")
            or it.properties.get("s2:mgrs_tile")
            or it.id[:10]
        )
        cloud = it.properties.get("eo:cloud_cover", 100)
        if tile_key not in best_items_by_tile:
            best_items_by_tile[tile_key] = it
        elif cloud < best_items_by_tile[tile_key].properties.get("eo:cloud_cover", 100):
            best_items_by_tile[tile_key] = it

    selected_items = list(best_items_by_tile.values())
    tile_names = list(best_items_by_tile.keys())
    print(
        f"  • Scene selezionate per coprire l'intera area: {len(selected_items)} tile ({tile_names})"
    )

    # 3. Scarica, ritaglia e unisce ciascuna banda
    band_arrays = []
    for b in BANDS:
        tile_crops = []
        for it in selected_items:
            if b in it.assets:
                href = it.assets[b].href
                ds = rioxarray.open_rasterio(href)
                cropped = ds.rio.clip_box(*bbox, crs="EPSG:4326")
                if "band" in cropped.dims and cropped.sizes["band"] == 1:
                    cropped = cropped.squeeze("band", drop=True)
                tile_crops.append(cropped)

        if not tile_crops:
            continue

        # Mosaico di tutte le tile dell'area per questa banda
        merged_band = merge_arrays(tile_crops) if len(tile_crops) > 1 else tile_crops[0]

        # Allinea risoluzione (20m -> 10m) sulla prima banda
        if band_arrays and (
            merged_band.rio.shape != band_arrays[0].rio.shape
            or merged_band.rio.crs != band_arrays[0].rio.crs
        ):
            merged_band = merged_band.rio.reproject_match(band_arrays[0])

        band_arrays.append(merged_band)

    if not band_arrays:
        print(f"⚠️ Nessuna banda scaricata per {year}-{month:02d}")
        return None

    # 4. Merge delle 6 bande nel singolo GeoTIFF multibanda
    merged = xarray.concat(band_arrays, dim="band")
    merged.coords["band"] = list(range(1, len(band_arrays) + 1))
    if band_arrays[0].rio.crs:
        merged.rio.write_crs(band_arrays[0].rio.crs, inplace=True)

    temp_out = file_out + ".tmp.tif"
    merged.rio.to_raster(temp_out, compress="deflate", predictor=2, tiled=True)
    os.replace(temp_out, file_out)

    file_size_mb = os.path.getsize(file_out) / (1024 * 1024)
    print(f"✅ Salvato {file_out} ({file_size_mb:.1f} MB)")
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
    print(f"\nFinito! Scaricati {len(downloaded)}/{len(list(months))} file.")
    return downloaded
