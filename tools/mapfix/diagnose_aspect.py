#!/usr/bin/env python3
"""Diagnose map distortion by comparing the current HOI4 map land/sea mask
against real-world geography (Natural Earth) rendered in the same
piecewise-band frame (see mapframe.py / bands.csv).

The HOI4 map in this mod is 5120x2560 but is NOT a single equirectangular
projection: it is a world band (Miller, +85..-57) stacked on an Antarctica
band (equirect, -57..-90). The current land/sea mask is taken from the
authoritative source -- provinces.bmp + definition.csv types (land/sea/lake),
NOT the heightmap/terrain colours. Real geography projected into the same band
frame should line up with that mask; where it does not, the hand-drawn map has
geometric distortion ("aspect ratio off").

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

import numpy as np
from PIL import Image, ImageFilter

from mapframe import (find_land_shapes, gis_land_mask, load_bands,
                      province_land_mask)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MAP_DIR = os.path.join(REPO, "bakasekai", "map")
GIS_DIR = os.environ.get("GIS_DIR", "/tmp/gis")
OUT_DIR = os.path.join(HERE, "out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280,
                    help="preview width (height = width/2)")
    ap.add_argument("--sealevel", type=int, default=71,
                    help="heightmap value at/below which is sea (--landsea heightmap)")
    ap.add_argument("--landsea", choices=["province", "heightmap"],
                    default="province",
                    help="land/sea source: province = provinces.bmp + "
                         "definition.csv types (accurate, default); heightmap = "
                         "legacy threshold")
    args = ap.parse_args()

    w = args.width
    h = w // 2
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Real geography mask in the map's piecewise-band frame
    shp, res = find_land_shapes(GIS_DIR)
    if not shp:
        raise SystemExit(
            f"no Natural Earth land shapefile in {GIS_DIR}. "
            "Download ne_10m_land.* (see README).")
    bands = load_bands()
    print(f"GIS reference: {res} ({len(shp)} layer(s): "
          + ", ".join(os.path.basename(p) for p in shp) + ")")
    print("bands:", ", ".join(f"{b[0]}[{b[3]:g}..{b[4]:g}]" for b in bands))
    gis = gis_land_mask(shp, w, h, bands)
    gis.save(os.path.join(OUT_DIR, "gis_land_mask.png"))

    # 2) Current map land mask -- authoritative from provinces.bmp + definition
    print(f"land/sea source: {args.landsea}")
    if args.landsea == "province":
        m = province_land_mask(os.path.join(MAP_DIR, "provinces.bmp"),
                               os.path.join(MAP_DIR, "definition.csv"), w, h)
        map_mask = Image.fromarray((m * 255).astype(np.uint8), "L")
    else:
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
