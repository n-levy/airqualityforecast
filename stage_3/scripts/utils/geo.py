from math import cos, radians


def bbox_from_point_km(lat, lon, halfwidth_km):
    lat_km = 111.32
    lon_km = 111.32 * cos(radians(lat if abs(lat) > 1e-6 else 0.0))
    dlat = halfwidth_km / lat_km
    dlon = halfwidth_km / lon_km if lon_km else halfwidth_km / 111.32
    return (lat + dlat, lon - dlon, lat - dlat, lon + dlon)
