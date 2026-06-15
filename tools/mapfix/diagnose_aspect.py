#!/usr/bin/env python3
"""Diagnose map distortion by comparing the current HOI4 map land/sea mask
against real-world geography (Natural Earth) rendered in the same
equirectangular frame.

The HOI4 map in this mod is 5120x2560 = 2:1, i.e. a full equirectangular
(-180..180 lon, -90..90 lat) projection. Real geography projected into the
exact same frame should line up with the painted coastline. Where it does
not, the hand-drawn map has geometric distortion ("aspect ratio off").

Outputs (written to OUT_DIR):
  - gis_land_mask.png      : real land mask in the map frame
  - map_land_mask.png      : current map land mask (from heightmap)
  - overlay.png            : current terrain with GIS coastline drawn on top
  - diff.png               : per-pixel land/sea disagreement (red=only map land,
                             blue=only GIS land)

Usage:
  python3 diagnose_aspect.py [--width 1280] [--sealevel 71]
"""
import argparse
import os

from PIL import Image, ImageFilter

try:
    import shapefile  # pyshp
except ImportError:
    raise SystemExit("pyshp required: pip install pyshp")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MAP_DIR = os.path.join(REPO, "bakasekai", "map")
GIS_DIR = os.environ.get("GIS_DIR", "/tmp/gis")
OUT_DIR = os.path.join(HERE, "out")


def lonlat_to_px(lon, lat, w, h):
    x = (lon + 180.0) / 360.0 * w
    y = (90.0 - lat) / 180.0 * h
    return x, y


def render_gis_land(shp_path, w, h):
    """Rasterize Natural Earth land polygons into an equirectangular mask."""
    from PIL import ImageDraw

    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    sf = shapefile.Reader(shp_path)
    for shape in sf.shapes():
        pts = shape.points
        parts = list(shape.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            ring = pts[parts[i]:parts[i + 1]]
            poly = [lonlat_to_px(lon, lat, w, h) for lon, lat in ring]
            if len(poly) >= 3:
                draw.polygon(poly, fill=255)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280,
                    help="preview width (height = width/2)")
    ap.add_argument("--sealevel", type=int, default=71,
                    help="heightmap value at/below which is sea")
    args = ap.parse_args()

    w = args.width
    h = w // 2
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Real geography mask
    shp = os.path.join(GIS_DIR, "ne_110m_land.shp")
    gis = render_gis_land(shp, w, h)
    gis.save(os.path.join(OUT_DIR, "gis_land_mask.png"))

    # 2) Current map land mask from heightmap
    hm = Image.open(os.path.join(MAP_DIR, "heightmap.bmp")).convert("L")
    hm = hm.resize((w, h), Image.NEAREST)
    map_mask = hm.point(lambda v: 255 if v > args.sealevel else 0)
    map_mask.save(os.path.join(OUT_DIR, "map_land_mask.png"))

    # 3) Overlay GIS coastline onto current terrain
    terr = Image.open(os.path.join(MAP_DIR, "terrain.bmp")).convert("RGB")
    terr = terr.resize((w, h), Image.NEAREST)
    edge = gis.filter(ImageFilter.FIND_EDGES)
    overlay = terr.copy()
    px = overlay.load()
    ep = edge.load()
    for y in range(h):
        for x in range(w):
            if ep[x, y] > 40:
                px[x, y] = (255, 0, 0)
    overlay.save(os.path.join(OUT_DIR, "overlay.png"))

    # 4) Land/sea disagreement
    diff = Image.new("RGB", (w, h), (0, 0, 0))
    dp = diff.load()
    mp = map_mask.load()
    gp = gis.load()
    only_map = only_gis = 0
    for y in range(h):
        for x in range(w):
            m = mp[x, y] > 127
            g = gp[x, y] > 127
            if m and not g:
                dp[x, y] = (220, 40, 40)   # map says land, reality says sea
                only_map += 1
            elif g and not m:
                dp[x, y] = (40, 90, 220)   # reality land, map says sea
                only_gis += 1
            elif m and g:
                dp[x, y] = (40, 40, 40)
    diff.save(os.path.join(OUT_DIR, "diff.png"))

    total = w * h
    print(f"resolution: {w}x{h}")
    print(f"map-only land (over-painted): {only_map} px ({100*only_map/total:.1f}%)")
    print(f"gis-only land (missing):      {only_gis} px ({100*only_gis/total:.1f}%)")
    print(f"outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
