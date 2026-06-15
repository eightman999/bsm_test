#!/usr/bin/env python3
"""Cosmetic restyle of the corrected map (out/corrected/), in place.

Two author-requested touch-ups that do NOT add or remove province IDs (so
definition.csv / states / strategic regions stay valid -- only shapes change):

1. Shrink Antarctica: vertically COMPRESS the Antarctica band (lat -57..-90) so
   its land ends around -85 instead of -90. No province is deleted -- the
   provinces that only existed in -85..-90 are squeezed upward with the rest.
   The freed strip at the very bottom becomes sea.

2. De-crush the polar boundary sea: the warp/projection leaves the sea along the
   top edge (and the new bottom strip) as thin slivers. Re-tile those zones into
   compact Voronoi cells, reusing the EXISTING sea province IDs there (no new
   IDs), so the crushed look is disguised.

Run AFTER `warp_map.py --apply`; re-running warp_map overwrites out/corrected/
so run this again afterwards. Then verify_apply.py.
"""
import argparse
import os

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

import warp_map as W
import mapframe as M

Image.MAX_IMAGE_PIXELS = None

OCEAN = {"terrain.bmp": 15, "heightmap.bmp": 71, "rivers.bmp": 254}
NORMAL_FLAT = (128, 128, 255)
SEAM = 0.8562          # world/Antarctica seam (normalized row)


def enc_of(rgb):
    a = rgb.astype(np.uint32)
    return ((a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]).astype(np.int64)


def voronoi(shape, seed_ys, seed_xs, seed_vals, region_mask):
    """Assign every pixel in region_mask the value of the nearest seed."""
    mark = np.ones(shape, bool)
    mark[seed_ys, seed_xs] = False
    _, (iy, ix) = distance_transform_edt(mark, return_indices=True)
    valgrid = np.zeros(shape, np.int64)
    valgrid[seed_ys, seed_xs] = seed_vals
    out = valgrid[iy, ix]
    return np.where(region_mask, out, -1)


def write_enc(prov, enc, mask, value_grid):
    """Write encoded province colours back into prov RGB where mask is set."""
    v = value_grid[mask]
    ys, xs = np.where(mask)
    prov[ys, xs, 0] = (v >> 16) & 255
    prov[ys, xs, 1] = (v >> 8) & 255
    prov[ys, xs, 2] = v & 255


def compress_band(arr, r_seam, r_h, r_keep):
    """Vertically squeeze rows [r_seam:r_h] into [r_seam:r_keep] (NEAREST)."""
    band = arr[r_seam:r_h]
    new_h = r_keep - r_seam
    if band.ndim == 2:
        im = Image.fromarray(band)
    else:
        im = Image.fromarray(band)
    comp = np.asarray(im.resize((arr.shape[1], new_h), Image.NEAREST))
    arr[r_seam:r_keep] = comp
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(W.OUT_DIR, "corrected"))
    ap.add_argument("--cut-lat", type=float, default=-85.0,
                    help="latitude to shrink Antarctica's bottom to (was -90)")
    ap.add_argument("--top-zone", type=float, default=0.14,
                    help="fraction of height at the top edge to de-crush")
    ap.add_argument("--cell-px", type=int, default=5000,
                    help="approx target pixel area per re-tiled sea cell")
    args = ap.parse_args()
    cdir = args.dir
    rng = np.random.default_rng(12345)

    enc_by_id, type_by_id = M.province_defs(os.path.join(W.MAP_DIR, "definition.csv"))
    sea_enc = set(enc_by_id[p] for p in enc_by_id if type_by_id.get(p) == "sea")

    # --- provinces (drives the sea re-tiling) ------------------------------
    pim = Image.open(os.path.join(cdir, "provinces.bmp")).convert("RGB")
    prov = np.array(pim)
    H, Wd = prov.shape[:2]
    r_seam = int(round(SEAM * H))
    r_h = H
    frac = (args.cut_lat - (-57.0)) / (-90.0 - (-57.0))      # -85 -> 0.848
    r_keep = r_seam + int(round(frac * (r_h - r_seam)))
    print(f"map {Wd}x{H}  antarctica band {r_seam}..{r_h} -> compress to "
          f"{r_seam}..{r_keep} (cut {args.cut_lat:g}), sea strip {r_keep}..{r_h}")

    enc0 = enc_of(prov)                     # corrected provinces before restyle
    prov = compress_band(prov, r_seam, r_h, r_keep)
    enc = enc_of(prov)

    # zone A: freed bottom strip (becomes sea); zone B: crushed top edge sea
    top_h = int(round(args.top_zone * H))
    strip = np.zeros((H, Wd), bool); strip[r_keep:r_h] = True
    topsea = np.zeros((H, Wd), bool)
    topsea[:top_h] = np.isin(enc[:top_h], np.array(sorted(sea_enc), dtype=np.int64))

    # pool of sea IDs to seed the bottom strip: coastal seas in the (compressed)
    # Antarctica band + the southern ocean just above the seam
    pool_zone = enc[max(0, r_seam - 200):r_keep]
    pool = sorted(set(np.unique(pool_zone).tolist()) & sea_enc)
    if not pool:
        pool = sorted(sea_enc)
    pool = np.array(pool, dtype=np.int64)

    # --- 1. fill the freed strip with compact sea cells (existing IDs) ------
    n_seeds = max(8, (strip.sum() // args.cell_px))
    sy = rng.integers(r_keep, r_h, size=n_seeds)
    sx = rng.integers(0, Wd, size=n_seeds)
    sv = pool[rng.integers(0, len(pool), size=n_seeds)]
    # guarantee every pool id survives somewhere: it already has pixels in the
    # band, so reuse here is safe; cells just extend those provinces.
    vg = voronoi((H, Wd), sy, sx, sv, strip)
    write_enc(prov, enc, strip, vg)

    # --- 2. de-crush the top-edge sea: one compact cell per existing sea id --
    if topsea.any():
        ids = np.unique(enc[topsea])
        sy2, sx2, sv2 = [], [], []
        for v in ids.tolist():
            yy, xx = np.where(topsea & (enc == v))
            if len(yy):
                sy2.append(int(yy.mean())); sx2.append(int(xx.mean())); sv2.append(v)
        if sv2:
            vg2 = voronoi((H, Wd), np.array(sy2), np.array(sx2),
                          np.array(sv2, np.int64), topsea)
            write_enc(prov, enc, topsea, vg2)

    enc = enc_of(prov)
    # restamp: compression/re-tiling can squeeze a tiny province out -> put it
    # back (1px stolen from a neighbour with >1px) so every ID survives.
    lost = set(np.unique(enc0).tolist()) - set(np.unique(enc).tolist())
    if lost:
        vals, cnts = np.unique(enc, return_counts=True)
        cnt = dict(zip(vals.tolist(), cnts.tolist()))

        def maprow(y):
            return r_seam + int(round((y - r_seam) * frac)) if r_seam <= y < r_h else y

        fixed = 0
        for idv in lost:
            ys, xs = np.where(enc0 == idv)
            for k in range(len(ys)):
                ny = min(max(maprow(int(ys[k])), 0), H - 1)
                nx = int(xs[k])
                occ = int(enc[ny, nx])
                if cnt.get(occ, 0) > 1:
                    cnt[occ] -= 1
                    prov[ny, nx] = [(idv >> 16) & 255, (idv >> 8) & 255, idv & 255]
                    enc[ny, nx] = idv
                    cnt[idv] = cnt.get(idv, 0) + 1
                    fixed += 1
                    break
        print(f"    restamped {fixed}/{len(lost)} squeezed provinces")
    now_ids = len(np.unique(enc))
    print(f"    provinces present after restyle: {now_ids}")
    Image.fromarray(prov, "RGB").save(os.path.join(cdir, "provinces.bmp"))

    # changed-to-sea pixels (need ocean values on the other layers): the strip
    sea_change = strip

    # --- other layers: compress the same way, ocean-fill the strip ---------
    for fname in ("terrain.bmp", "rivers.bmp", "heightmap.bmp", "world_normal.bmp"):
        p = os.path.join(cdir, fname)
        if not os.path.exists(p):
            continue
        im = Image.open(p)
        mode = im.mode
        a = np.array(im)
        fh, fw = a.shape[:2]
        rs = int(round(SEAM * fh)); rk = rs + int(round(frac * (fh - rs)))
        a = compress_band(a, rs, fh, rk)
        sc = np.zeros((fh, fw), bool); sc[rk:fh] = True
        if fname == "world_normal.bmp":
            a[sc] = NORMAL_FLAT if a.ndim == 3 else NORMAL_FLAT[0]
        else:
            a[sc] = OCEAN[fname]
        if mode == "P":
            outim = Image.fromarray(a.astype(np.uint8), "P"); outim.putpalette(im.getpalette())
        else:
            outim = Image.fromarray(a, mode)
        outim.save(p)
        print(f"    restyled {fname} ({mode} {fw}x{fh})")

    print("done. now run: python3 verify_apply.py")


if __name__ == "__main__":
    main()
