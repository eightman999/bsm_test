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
import math
import os

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BANDS = os.path.join(HERE, "bands.csv")

# fallback if bands.csv is missing
_FALLBACK = [("world", 0.0, 0.918, 90.0, -60.0, "equirect"),
             ("antarctica", 0.918, 1.0, -65.0, -90.0, "equirect")]


def province_types(defs_path):
    """Map enc(rgb) -> province type from definition.csv.

    definition.csv columns: id;r;g;b;type;coastal;terrain;continent.
    type is one of land / sea / lake (this mod). Returns {enc: type}.
    """
    types = {}
    with open(defs_path, newline="") as f:
        for row in csv.reader(f, delimiter=";"):
            if len(row) < 5 or not row[0].strip().lstrip("-").isdigit():
                continue
            r, g, b = int(row[1]), int(row[2]), int(row[3])
            types[(r << 16) | (g << 8) | b] = row[4].strip().lower()
    return types


def province_defs(defs_path):
    """Parse definition.csv -> (enc_by_id, type_by_id).

    id;r;g;b;type;...  ->  {pid: enc}, {pid: type}. Lets callers turn the
    province-ID lists used by strategic regions / states into pixel colours.
    """
    enc_by_id, type_by_id = {}, {}
    with open(defs_path, newline="") as f:
        for row in csv.reader(f, delimiter=";"):
            if len(row) < 5 or not row[0].strip().lstrip("-").isdigit():
                continue
            pid = int(row[0])
            r, g, b = int(row[1]), int(row[2]), int(row[3])
            enc_by_id[pid] = (r << 16) | (g << 8) | b
            type_by_id[pid] = row[4].strip().lower()
    return enc_by_id, type_by_id


def province_land_mask(prov_path, defs_path, w, h, sea_types=("sea",)):
    """Authoritative land/sea mask from provinces.bmp + definition.csv types.

    A pixel is SEA iff its province's type is in sea_types; everything else
    (land, lake, unknown) counts as land. This is the ground truth HOI4 itself
    uses, and is far more accurate than thresholding heightmap/terrain (the
    painted terrain colours do not reliably encode the coastline). Natural
    Earth's land layer keeps inland lakes as land, so we treat HOI4 'lake'
    provinces as land too for a like-for-like coastline comparison.

    Returns float32 (h, w), 1.0 = land, 0.0 = sea.
    """
    types = province_types(defs_path)
    sea_enc = np.array(sorted(e for e, t in types.items() if t in sea_types),
                       dtype=np.int64)
    im = Image.open(prov_path).convert("RGB").resize((w, h), Image.NEAREST)
    a = np.asarray(im, dtype=np.uint32)
    enc = ((a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]).astype(np.int64)
    is_sea = np.isin(enc, sea_enc)
    return (~is_sea).astype(np.float32)


def find_land_shapes(gis_dir):
    """Return the finest available Natural Earth land shapefiles in gis_dir.

    Natural Earth ships at three scales: 10m (finest, keeps small islands),
    50m, and 110m (coarsest -- small islands vanish). We prefer the finest set
    present and, at that scale, fold in minor-island / reef layers too so the
    reference coastline isn't missing islands the painted map actually has.

    Returns (paths, resolution_label). paths is [] if nothing is found.
    """
    for res in ("10m", "50m", "110m"):
        land = os.path.join(gis_dir, f"ne_{res}_land.shp")
        if os.path.exists(land):
            paths = [land]
            for extra in (f"ne_{res}_minor_islands.shp",
                          f"ne_{res}_reefs.shp"):
                p = os.path.join(gis_dir, extra)
                if os.path.exists(p):
                    paths.append(p)
            return paths, res
    return [], None


def load_bands(path=DEFAULT_BANDS):
    """Return [(name, y0n, y1n, lat_top, lat_bot, proj), ...] (normalized rows).

    proj is 'equirect' or 'mercator' (optional 6th column; default equirect).
    """
    if not path or not os.path.exists(path):
        return list(_FALLBACK)
    bands = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            name, y0, y1, lt, lb = row[:5]
            proj = row[5].strip() if len(row) > 5 and row[5].strip() else "equirect"
            bands.append((name, float(y0), float(y1), float(lt), float(lb), proj))
    return bands or list(_FALLBACK)


def _merc(lat):
    lat = max(min(lat, 89.99), -89.99)
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def _miller(lat):
    """Miller cylindrical: like Mercator but the poles are at a finite y, so the
    map can include +/-90 (or near it). y = 1.25 * ln(tan(pi/4 + 0.4*phi))."""
    lat = max(min(lat, 89.99), -89.99)
    return 1.25 * math.log(math.tan(math.pi / 4 + 0.4 * math.radians(lat)))


_PROJ = {"equirect": None, "mercator": _merc, "miller": _miller}


def gis_land_mask(shp_path, w, h, bands=None):
    """Rasterize Natural Earth land polygons into the piecewise-band frame.

    shp_path may be a single shapefile path or a list of paths (e.g. land +
    minor islands); all are rasterized into the same mask. Each band renders all
    polygons with its own lat->row scale, then is pasted into its row range.
    Returns a uint8 0/255 array (h, w).
    """
    if bands is None:
        bands = load_bands()
    try:
        import shapefile  # pyshp
    except ImportError:
        raise SystemExit("pyshp required: pip install pyshp")
    paths = [shp_path] if isinstance(shp_path, str) else list(shp_path)
    shapes = []
    for p in paths:
        shapes.extend(shapefile.Reader(p).shapes())

    out = Image.new("L", (w, h), 0)
    for band in bands:
        _name, y0n, y1n, lat0, lat1 = band[:5]
        proj = band[5] if len(band) > 5 else "equirect"
        ry0, ry1 = int(round(y0n * h)), int(round(y1n * h))
        if ry1 <= ry0:
            continue

        def X(lon):
            return (lon + 180.0) / 360.0 * w

        fwd = _PROJ.get(proj)
        if fwd is not None:                    # cylindrical (mercator/miller)
            m0, m1 = fwd(lat0), fwd(lat1)

            def Y(lat, ry0=ry0, ry1=ry1, m0=m0, m1=m1, fwd=fwd):
                return ry0 + (m0 - fwd(lat)) / (m0 - m1) * (ry1 - ry0)
        else:                                  # equirectangular
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
