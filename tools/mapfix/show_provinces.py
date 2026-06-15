#!/usr/bin/env python3
"""Render province maps (with borders) so you can eyeball the warp result.

Draws the live map (bakasekai/map/provinces.bmp) and the corrected one
(out/corrected/provinces.bmp) at a viewable size: land/sea/lake tinted by
definition.csv type, province borders in black. Use it to judge whether the
warp kept the province mesh clean (no mangling/fragmentation) and moved the
coastlines the way you want.

Outputs out/provinces_before.png (live) and out/provinces_after.png (corrected).

Usage:  python3 show_provinces.py [--width 2048]
"""
import argparse
import os

import numpy as np
from PIL import Image

import warp_map as W
import mapframe as M

Image.MAX_IMAGE_PIXELS = None

SEA = (28, 40, 72)
LAND = (202, 196, 176)
LAKE = (90, 150, 190)
BORDER = (0, 0, 0)


def render(prov_path, w, h, type_by_id, enc_to_type_arr=None):
    im = Image.open(prov_path).convert("RGB").resize((w, h), Image.NEAREST)
    a = np.asarray(im, dtype=np.uint32)
    enc = ((a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]).astype(np.int64)

    # classify each pixel land/sea/lake via definition.csv
    enc_by_id, tmap = type_by_id
    sea = np.zeros(enc.shape, bool)
    lake = np.zeros(enc.shape, bool)
    uniq = np.unique(enc)
    # map enc -> type using a dict lookup over unique values (fast)
    type_of = {}
    inv_enc = {v: k for k, v in enc_by_id.items()}   # enc -> id
    for e in uniq.tolist():
        pid = inv_enc.get(e)
        type_of[e] = tmap.get(pid, "land") if pid is not None else "land"
    sea_set = np.array(sorted(e for e in uniq.tolist() if type_of[e] == "sea"),
                       dtype=np.int64)
    lake_set = np.array(sorted(e for e in uniq.tolist() if type_of[e] == "lake"),
                        dtype=np.int64)
    sea = np.isin(enc, sea_set)
    lake = np.isin(enc, lake_set)

    out = np.empty((h, w, 3), np.uint8)
    out[...] = LAND
    out[sea] = SEA
    out[lake] = LAKE

    # province borders: pixel differs from right or down neighbour
    b = np.zeros(enc.shape, bool)
    b[:, :-1] |= enc[:, :-1] != enc[:, 1:]
    b[:-1, :] |= enc[:-1, :] != enc[1:, :]
    out[b] = BORDER
    return Image.fromarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--corrected", default=os.path.join(W.OUT_DIR, "corrected",
                                                        "provinces.bmp"))
    args = ap.parse_args()
    w = args.width
    h = w // 2

    defs = M.province_defs(os.path.join(W.MAP_DIR, "definition.csv"))

    live = os.path.join(W.MAP_DIR, "provinces.bmp")
    print(f"rendering live    -> out/provinces_before.png ({w}x{h})")
    render(live, w, h, defs).save(os.path.join(W.OUT_DIR, "provinces_before.png"))

    if os.path.exists(args.corrected):
        print(f"rendering warped  -> out/provinces_after.png ({w}x{h})")
        render(args.corrected, w, h, defs).save(
            os.path.join(W.OUT_DIR, "provinces_after.png"))
    else:
        print(f"(no corrected provinces at {args.corrected}; run warp_map.py --apply)")


if __name__ == "__main__":
    main()
