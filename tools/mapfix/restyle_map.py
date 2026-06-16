#!/usr/bin/env python3
"""Cosmetic restyle of the corrected map (out/corrected/), in place.

Author-requested touch-ups that do NOT add or remove province IDs (so
definition.csv / states / strategic regions stay valid -- only shapes change):

1. Re-fill Antarctica at the world scale (fixes the "floating" Antarctica).
   The map is a world band + a separate Antarctica band stacked vertically with
   DIFFERENT vertical scales ("two aspect ratios"): at the seam the world band
   is ~17.2 px/deg (Miller, stretched toward -57) while the Antarctica band is
   only ~11.2 px/deg. The earlier shrink left a sea strip at the very bottom, so
   Antarctica floated off the bottom edge.

   Instead we VERTICALLY RE-MAP the Antarctica band so its COAST (lat -57..
   --coast-lat, default -75) is drawn at the world band's seam scale -- the seam
   becomes continuous (no scale jump) -- and the polar INTERIOR (--coast-lat..
   -90, just ice plateau) is compressed into whatever band space remains, so the
   band stays the same height and Antarctica reaches the bottom edge again (no
   float, no sea gap). Columns are untouched and the resample is nearest, so
   province colours/IDs are preserved; any interior province squeezed to 0 px is
   restamped back (1 px stolen from a neighbour).

2. De-crush the polar boundary sea: the warp/projection leaves the sea along the
   top edge as thin slivers. Re-tile that zone into compact Voronoi cells,
   reusing the EXISTING sea province IDs there (no new IDs), so the crushed look
   is disguised.

Run AFTER `warp_map.py --apply` (it overwrites out/corrected/, so re-run this
afterwards). Then verify_apply.py.
"""
import argparse
import os

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

import warp_map as W
import mapframe as M

Image.MAX_IMAGE_PIXELS = None

SEAM = 0.8562          # world/Antarctica seam (normalized row)
LAT_TOP = -57.0        # Antarctica band top latitude (the seam)
LAT_BOT = -90.0        # Antarctica band bottom latitude (south pole)


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


def world_seam_scale(H, bands):
    """px-per-degree of the WORLD band at its bottom edge (the seam latitude)."""
    for band in bands:
        name, y0, y1, lt, lb = band[:5]
        if name != "world":
            continue
        ry0, ry1 = round(y0 * H), round(y1 * H)
        fwd = M._PROJ.get(band[5] if len(band) > 5 else "equirect")
        if fwd is not None:
            m0, m1 = fwd(lt), fwd(lb)
            Y = lambda lat, m0=m0, m1=m1: ry0 + (m0 - fwd(lat)) / (m0 - m1) * (ry1 - ry0)
        else:
            Y = lambda lat: ry0 + (lt - lat) / (lt - lb) * (ry1 - ry0)
        eps = 0.1
        return abs(Y(lb) - Y(lb + eps)) / eps
    return 17.25


def ant_remap_fracs(H, r_seam, bands, coast_lat, coast_scale=None):
    """A 2-segment dest->src fraction map for the Antarctica band.

    Coast [LAT_TOP..coast_lat] is drawn at the world band's seam scale
    (coast_scale px/deg, auto by default) so the seam is continuous; the interior
    [coast_lat..LAT_BOT] is compressed into the remaining band space so the band
    fills to the bottom edge. Returns (f_split, g_split, s_w):
      f_split = dest fraction where coast ends / interior begins
      g_split = src  fraction at the same latitude (coast_lat)
    Resolution-independent (fractions), so it applies to any layer height.
    """
    band_px = H - r_seam
    s_w = coast_scale if coast_scale else world_seam_scale(H, bands)
    tot_deg = abs(LAT_BOT) - abs(LAT_TOP)            # 33
    coast_deg = abs(coast_lat) - abs(LAT_TOP)        # e.g. -75 -> 18
    f_split = min((coast_deg * s_w) / band_px, 0.999)
    g_split = coast_deg / tot_deg
    return f_split, g_split, s_w


def remap_src_rows(fh, r_seam, f_split, g_split):
    """For each dest row in [r_seam:fh], the source row to copy (nearest)."""
    band = fh - r_seam
    f = np.arange(band) / max(band - 1, 1)           # dest fraction 0..1
    denom = max(1.0 - f_split, 1e-6)
    g = np.where(f <= f_split,
                 (f / max(f_split, 1e-6)) * g_split,
                 g_split + (f - f_split) / denom * (1.0 - g_split))
    src = r_seam + np.clip(g, 0.0, 1.0) * (band - 1)
    return np.clip(np.round(src).astype(int), r_seam, fh - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(W.OUT_DIR, "corrected"))
    ap.add_argument("--coast-lat", type=float, default=-75.0,
                    help="latitude down to which Antarctica's COAST is drawn at "
                         "the world band's scale; below this the polar interior "
                         "is compressed to fill the band to the bottom edge")
    ap.add_argument("--coast-scale", type=float, default=None,
                    help="px/deg for the coast (default: auto = world band seam "
                         "scale, so the seam is continuous)")
    ap.add_argument("--top-zone", type=float, default=0.14,
                    help="fraction of height at the top edge to de-crush")
    args = ap.parse_args()
    cdir = args.dir
    bands = M.load_bands()

    enc_by_id, type_by_id = M.province_defs(os.path.join(W.MAP_DIR, "definition.csv"))
    sea_enc = set(enc_by_id[p] for p in enc_by_id if type_by_id.get(p) == "sea")

    # --- provinces (drives the sea re-tiling) ------------------------------
    pim = Image.open(os.path.join(cdir, "provinces.bmp")).convert("RGB")
    prov = np.array(pim)
    H, Wd = prov.shape[:2]
    r_seam = int(round(SEAM * H))
    f_split, g_split, s_w = ant_remap_fracs(H, r_seam, bands,
                                            args.coast_lat, args.coast_scale)
    print(f"map {Wd}x{H}  seam row {r_seam}  antarctica band {H - r_seam}px")
    print(f"  coast {LAT_TOP:g}..{args.coast_lat:g} drawn at {s_w:.2f} px/deg "
          f"(world scale) -> dest frac 0..{f_split:.3f}")
    print(f"  interior {args.coast_lat:g}..{LAT_BOT:g} compressed into the rest; "
          f"band fills to the bottom edge (no float)")

    enc0 = enc_of(prov)                     # warp output (source) before restyle
    src_rows = remap_src_rows(H, r_seam, f_split, g_split)
    prov[r_seam:] = prov[src_rows]          # fancy index -> copy, safe in place
    enc = enc_of(prov)

    # --- de-crush the top-edge sea: one compact cell per existing sea id ----
    top_h = int(round(args.top_zone * H))
    topsea = np.zeros((H, Wd), bool)
    topsea[:top_h] = np.isin(enc[:top_h], np.array(sorted(sea_enc), dtype=np.int64))
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

    # restamp: interior compression can squeeze a tiny province out -> put it
    # back (1px stolen from a neighbour with >1px) so every ID survives.
    lost = set(np.unique(enc0).tolist()) - set(np.unique(enc).tolist())
    if lost:
        vals, cnts = np.unique(enc, return_counts=True)
        cnt = dict(zip(vals.tolist(), cnts.tolist()))

        def src_to_dest(y):
            if y < r_seam:
                return y
            d = int(np.searchsorted(src_rows, y))    # src_rows ascending
            return min(r_seam + d, H - 1)

        fixed = 0
        for idv in lost:
            ys, xs = np.where(enc0 == idv)
            for k in range(len(ys)):
                ny = min(max(src_to_dest(int(ys[k])), 0), H - 1)
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

    # --- other layers: same vertical remap (no ocean fill; fills to bottom) --
    for fname in ("terrain.bmp", "rivers.bmp", "heightmap.bmp", "world_normal.bmp"):
        p = os.path.join(cdir, fname)
        if not os.path.exists(p):
            continue
        im = Image.open(p)
        mode = im.mode
        a = np.array(im)
        fh, fw = a.shape[:2]
        rs = int(round(SEAM * fh))
        sr = remap_src_rows(fh, rs, f_split, g_split)
        a[rs:] = a[sr]
        if mode == "P":
            outim = Image.fromarray(a.astype(np.uint8), "P"); outim.putpalette(im.getpalette())
        else:
            outim = Image.fromarray(a, mode)
        outim.save(p)
        print(f"    restyled {fname} ({mode} {fw}x{fh})")

    print("done. now run: python3 verify_apply.py")


if __name__ == "__main__":
    main()
