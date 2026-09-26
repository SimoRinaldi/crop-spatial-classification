# confini geografici della Capitanata
# (EPSG: 4326 min_lon, min_lat, max_lon, max_lat)
CAPITANATA_BBOX = [
    15.046692771569205,  # min_lon (ovest)
    41.08392693013771,  # min_lat (sud)
    16.210767777473393,  # max_lon (est)
    41.92848568568325,  # max_lat (nord)
]

# classi da escludere a priori:
EXCLUDED_CROP_CLASSES = [
    0,  # -> all non-cropland areas
    3100,  # -> Undecided arable crop
    3200,  # -> Undecided perennial crop
    65534,  # -> water / sea
    65535,  # -> outside area
]

# anni campionati
YEARS_TO_FETCH = ["2023", "2022", "2021"]


def is_in_capitanata(lon: float, lat: float) -> bool:
    min_lon, min_lat, max_lon, max_lat = CAPITANATA_BBOX
    return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)
