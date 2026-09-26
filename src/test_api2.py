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
    name,
    bbox,
    year,
    month,
    out_dir="./data/processed/sentinel2_capitanata_area",
    overwrite=False,
):
    """
    Scarica e fonde le bande di un mese per l'area specificata.
    Per garantire la copertura completa dell'area senza buchi, seleziona
    la scena più limpida del mese per ciascuna tile MGRS che interseca la BBox.
    """
    os.makedirs(out_dir, exist_ok=True)
    file_out = os.path.join(out_dir, f"{name}_{year}_{month:02d}.tif")

    # Resume: se il file è già completo (> 200 MB), salta (a meno di overwrite=True)
    if not overwrite and os.path.exists(file_out) and os.path.getsize(file_out) > 200_000_000:
        print(f"Mese {month:02d} già presente e completo: {file_out} ({os.path.getsize(file_out)/(1024*1024):.1f} MB)")
        return file_out

    print(f"\nScaricando {name} per {year}-{month:02d}...")
    last_day = calendar.monthrange(int(year), int(month))[1]
    
    # 1. Ricerca scene nel mese per la sola BBox specificata
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day:02d}",
        query={"eo:cloud_cover": {"lt": 40}}
    )
    items = list(search.items())

    # Se troppe nuvole, rilassa la soglia per garantire la presenza di tutte le tile
    if len(items) < 2:
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day:02d}",
            query={"eo:cloud_cover": {"lt": 80}}
        )
        items = list(search.items())

    if not items:
        print(f"⚠️ Nessuna immagine trovata per {year}-{month:02d}")
        return None

    # 2. Per ciascuna tile MGRS, seleziona la scena a copertura intera con minor nuvolosità.
    # Evita di selezionare passaggi orbitali parziali che hanno un cloud_cover basso solo perché
    # il satellite ha sfiorato l'8% della tile (lasciando l'altro 92% vuoto/nero).
    items_by_tile = {}
    for it in items:
        tile_key = (
            it.properties.get("grid:code")
            or it.properties.get("s2:mgrs_tile")
            or it.id[:10]
        )
        items_by_tile.setdefault(tile_key, []).append(it)

    selected_items = []
    tile_names = []
    for tile_key, tile_items in items_by_tile.items():
        min_nodata = min(it.properties.get("s2:nodata_pixel_percentage", 0) for it in tile_items)
        # Filtra solo le scene che hanno copertura massima per quella tile (entro +15% dal minimo nodata)
        candidates = [
            it for it in tile_items
            if it.properties.get("s2:nodata_pixel_percentage", 0) <= min_nodata + 15
        ]
        best_it = min(candidates, key=lambda it: it.properties.get("eo:cloud_cover", 100))
        selected_items.append(best_it)
        tile_names.append(tile_key)

    print(f"  • Scene selezionate per coprire la Capitanata: {len(selected_items)} tile ({tile_names})")

    # 3. Scarica, ritaglia e unisce ciascuna banda con feedback visivo
    band_arrays = []
    for idx_b, b in enumerate(BANDS, 1):
        print(f"    -> [{idx_b}/{len(BANDS)}] Elaborazione banda '{b}'...")
        tile_crops = []
        for it in selected_items:
            if b in it.assets:
                href = it.assets[b].href
                try:
                    ds = rioxarray.open_rasterio(href)
                    cropped = ds.rio.clip_box(*bbox, crs="EPSG:4326")
                    if "band" in cropped.dims and cropped.sizes["band"] == 1:
                        cropped = cropped.squeeze("band", drop=True)
                    cropped.rio.write_nodata(0, inplace=True)
                    tile_crops.append(cropped)
                except Exception:
                    pass

        if not tile_crops:
            continue

        # Mosaico di tutte le tile dell'area per questa banda (nodata=0 evita che i bordi neri sovrascrivano i dati validi)
        merged_band = merge_arrays(tile_crops, nodata=0) if len(tile_crops) > 1 else tile_crops[0]

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
    merged.rio.write_nodata(0, inplace=True)

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
    overwrite=False,
):
    """Scarica tutti i mesi richiesti per l'area data."""
    print(f"Inizio download area '{name}' per l'anno {year}...")
    downloaded = []
    for m in months:
        f = download_area_month(name, bbox, year, m, out_dir=out_dir, overwrite=overwrite)
        if f:
            downloaded.append(f)
    print(f"\nFinito! Scaricati {len(downloaded)}/{len(list(months))} file.")
    return downloaded
