"""Shared map-frame model for the mapfix tools.

IMPORTANT: this mod's map is NOT a single equirectangular projection. It is a
world map and a separate Antarctica map merged vertically, then squashed to fit
the engine's pixel budget. So the vertical scale (degrees-of-latitude per pixel)
is different in each band -- that is the "two aspect ratios" in the map.

We model it as a stack of equirectangular *bands*. Each band maps a normalized
row range [y0..y1] (0=top, 1=bottom) to a latitude range [lat_top..lat_bot],
with longitude always equirectangular across the full width.

Detected for bakasekai (see bands.csv):
  world      rows 0.000..0.918  ->  +90 .. -60 deg   (~15.4 px/deg vertical)
  antarctica rows 0.918..1.000  ->  -65 .. -90 deg   (~8.4  px/deg vertical)
"""
import csv
import os

from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BANDS = os.path.join(HERE, "bands.csv")

# fallback if bands.csv is missing
_FALLBACK = [("world", 0.0, 0.918, 90.0, -60.0),
             ("antarctica", 0.918, 1.0, -65.0, -90.0)]


def load_bands(path=DEFAULT_BANDS):
    """Return [(name, y0n, y1n, lat_top, lat_bot), ...] (normalized rows)."""
    if not path or not os.path.exists(path):
        return list(_FALLBACK)
    bands = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            name, y0, y1, lt, lb = row[:5]
            bands.append((name, float(y0), float(y1), float(lt), float(lb)))
    return bands or list(_FALLBACK)


def gis_land_mask(shp_path, w, h, bands=None):
    """Rasterize Natural Earth land polygons into the piecewise-band frame.

    Each band renders all polygons with its own lat->row scale, then is pasted
    into its row range. Returns a uint8 0/255 array (h, w).
    """
    if bands is None:
        bands = load_bands()
    try:
        import shapefile  # pyshp
    except ImportError:
        raise SystemExit("pyshp required: pip install pyshp")
    shapes = shapefile.Reader(shp_path).shapes()

    out = Image.new("L", (w, h), 0)
    for _name, y0n, y1n, lat0, lat1 in bands:
        ry0, ry1 = int(round(y0n * h)), int(round(y1n * h))
        if ry1 <= ry0:
            continue

        def X(lon):
            return (lon + 180.0) / 360.0 * w

        def Y(lat, ry0=ry0, ry1=ry1, lat0=lat0, lat1=lat1):
            return ry0 + (lat0 - lat) / (lat0 - lat1) * (ry1 - ry0)

        layer = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(layer)
        for sh in shapes:
            pts = sh.points
            parts = list(sh.parts) + [len(pts)]
            for i in range(len(parts) - 1):
                ring = pts[parts[i]:parts[i + 1]]
                poly = [(X(lon), Y(lat)) for lon, lat in ring]
                if len(poly) >= 3:
                    d.polygon(poly, fill=255)
        band_crop = layer.crop((0, ry0, w, ry1))
        out.paste(band_crop, (0, ry0))
    return out
