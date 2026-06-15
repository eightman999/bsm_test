#!/usr/bin/env python3
"""Visualize protect_provinces.csv: paint each protected region a distinct
colour on the map frame so you can confirm the region IDs select the right
landmasses (positions are baka-distorted, so eyeball the SHAPE, not real-world
location). Writes out/protect_regions_colored.png and prints a legend with each
region's province count and centroid.

Usage:  python3 show_protect.py [--protect-provinces protect_provinces.csv] [--width 1024]
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from PIL import Image

import warp_map as W
import mapframe as M

# stable, visually distinct colours; regions beyond this cycle through them
_COLORS = [
    (255, 60, 60), (60, 120, 255), (60, 220, 60), (255, 0, 200), (255, 160, 40),
    (0, 230, 230), (230, 230, 0), (180, 180, 180), (150, 80, 255), (255, 120, 180),
    (120, 200, 120), (200, 120, 60),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protect-provinces",
                    default=os.path.join(W.HERE, "protect_provinces.csv"))
    ap.add_argument("--width", type=int, default=1024)
    args = ap.parse_args()
    w = args.width
    h = w // 2

    enc_by_id, type_by_id = M.province_defs(
        os.path.join(W.MAP_DIR, "definition.csv"))
    im = Image.open(os.path.join(W.MAP_DIR, "provinces.bmp")).convert("RGB") \
        .resize((w, h), Image.NEAREST)
    a = np.asarray(im, dtype=np.uint32)
    enc = ((a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]).astype(np.int64)

    regions = defaultdict(list)
    for row in csv.reader(open(args.protect_provinces)):
        row = [c.strip() for c in row if c.strip() != ""]
        if not row or row[0].startswith("#"):
            continue
        name, kind, ids = row[0], row[1].lower(), W._expand_ids(row[2:])
        pids = []
        if kind == "prov":
            pids = ids
        elif kind in ("sr", "state"):
            d = W.SR_DIR if kind == "sr" else W.STATE_DIR
            for rid in ids:
                fp = W._region_file(d, rid)
                if fp:
                    pids += W._provinces_in_file(fp)
        regions[name] += [p for p in pids
                          if type_by_id.get(p) != "sea" and p in enc_by_id]

    out = np.zeros((h, w, 3), np.uint8)
    print(f"{'region':20s} {'prov':>5}  centroid")
    for i, (name, pids) in enumerate(regions.items()):
        sel = np.array(sorted(enc_by_id[p] for p in set(pids)), dtype=np.int64)
        mask = np.isin(enc, sel)
        ys, xs = np.where(mask)
        col = _COLORS[i % len(_COLORS)]
        out[mask] = col
        if len(xs):
            lon = xs.mean() / w * 360 - 180
            lat = 90 - ys.mean() / h * 180
            print(f"{name:20s} {len(set(pids)):5d}  lon~{lon:+.0f} lat~{lat:+.0f}  "
                  f"rgb{col}")
        else:
            print(f"{name:20s} {0:5d}  (no land provinces resolved!)  rgb{col}")

    path = os.path.join(W.OUT_DIR, "protect_regions_colored.png")
    Image.fromarray(out).save(path)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
