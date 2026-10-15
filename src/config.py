import os
import sys
from pathlib import Path

# Confini geografici della Capitanata
# (EPSG: 4326 min_lon, min_lat, max_lon, max_lat)
CAPITANATA_BBOX = [
    15.046692771569205,  # min_lon (ovest)
    41.08392693013771,  # min_lat (sud)
    16.210767777473393,  # max_lon (est)
    41.92848568568325,  # max_lat (nord)
]

VICINO_CAPITANATA_BBOX = [
    14.5536,  # min_lon (ovest)
    41.8387,  # min_lat (sud)
    15.8382,  # max_lon (est)
    42.8017,  # max_lat (nord)
]

# Classi da escludere a priori:
EXCLUDED_CROP_CLASSES = [
    0,  # -> all non-cropland areas
    3100,  # -> Undecided arable crop
    3200,  # -> Undecided perennial crop
    65534,  # -> water / sea
    65535,  # -> outside area
]

# Anni campionati
YEARS_TO_FETCH = ["2023", "2022", "2021"]


def is_in_capitanata(lon: float, lat: float) -> bool:
    """Verifica se una coordinata (lon, lat) ricade all'interno della BBox della Capitanata."""
    min_lon, min_lat, max_lon, max_lat = CAPITANATA_BBOX
    return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)
