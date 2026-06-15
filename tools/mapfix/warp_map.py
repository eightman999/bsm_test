#!/usr/bin/env python3
"""Distortion-correcting warp for the HOI4 map.

Goal: keep the mod's *design* (which province/country sits where) but fix the
geometric distortion ("aspect ratio off") by gently warping every map layer so
that the painted coastline lines up with real-world geography (Natural Earth)
in the equirectangular frame.

Why a warp (and not a redraw): warping the existing layers preserves every
province's unique colour, so province IDs stay valid. definition.csv, states,
history, supply etc. keep working -- only the shapes/positions are corrected.
provinces.bmp / terrain.bmp are resampled with NEAREST so colours/indices are
never blended into invalid values; heightmap / normals use bilinear.

Pipeline:
  1. Build land/sea masks for the current map and for real geography.
  2. Estimate a smooth displacement field via FFT block-matching that maps
     each output (corrected) location back to the source map location whose
     local coastline matches real geography there.
  3. Resample every layer through that field.
  4. Re-run the land/sea diff to quantify the improvement.

By default it only writes previews + corrected copies under out/; it never
overwrites the live map unless you pass --apply.

Usage:
  python3 warp_map.py                      # estimate + preview (safe)
  python3 warp_map.py --work 1536          # finer field estimation
  python3 warp_map.py --apply              # write corrected full-res layers
                                           # into out/corrected/
"""
import argparse
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from scipy.signal import fftconvolve

try:
    import shapefile  # pyshp
except ImportError:
    raise SystemExit("pyshp required: pip install pyshp")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MAP_DIR = os.path.join(REPO, "bakasekai", "map")
GIS_DIR = os.environ.get("GIS_DIR", "/tmp/gis")
OUT_DIR = os.path.join(HERE, "out")

Image.MAX_IMAGE_PIXELS = None

# layer file -> resampling order (0 = nearest, 1 = bilinear)
LAYERS = {
    "provinces.bmp": 0,
    "terrain.bmp": 0,
    "rivers.bmp": 0,
    "heightmap.bmp": 1,
    "world_normal.bmp": 1,
}


def lonlat_to_px(lon, lat, w, h):
    return (lon + 180.0) / 360.0 * w, (90.0 - lat) / 180.0 * h


def gis_land_mask(w, h):
    from PIL import ImageDraw
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    sf = shapefile.Reader(os.path.join(GIS_DIR, "ne_110m_land.shp"))
    for shape in sf.shapes():
        pts = shape.points
        parts = list(shape.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            ring = pts[parts[i]:parts[i + 1]]
            poly = [lonlat_to_px(lon, lat, w, h) for lon, lat in ring]
            if len(poly) >= 3:
                draw.polygon(poly, fill=255)
    return np.asarray(img, dtype=np.float32) / 255.0


def map_land_mask(w, h, sealevel):
    hm = Image.open(os.path.join(MAP_DIR, "heightmap.bmp")).convert("L")
    hm = hm.resize((w, h), Image.NEAREST)
    return (np.asarray(hm) > sealevel).astype(np.float32)


def estimate_field(src, ref, grid, win, search):
    """Displacement field d so that src[y+dy, x+dx] ~ ref[y, x].

    src, ref: float masks, same HxW. Returns (dy, dx) on a node grid plus the
    node coordinate vectors.
    """
    h, w = ref.shape
    ys = np.arange(grid // 2, h, grid)
    xs = np.arange(grid // 2, w, grid)
    dy = np.zeros((len(ys), len(xs)), np.float32)
    dx = np.zeros((len(ys), len(xs)), np.float32)
    conf = np.zeros((len(ys), len(xs)), np.float32)

    src0 = src - src.mean()
    ref0 = ref - ref.mean()
    pad = win + search
    sP = np.pad(src0, pad)
    rP = np.pad(ref0, pad)
    for iy, cy in enumerate(ys):
        for ix, cx in enumerate(xs):
            Y, X = cy + pad, cx + pad
            patch = rP[Y - win:Y + win, X - win:X + win]
            if patch.std() < 1e-3:           # featureless (open ocean / interior)
                continue
            area = sP[Y - win - search:Y + win + search,
                      X - win - search:X + win + search]
            corr = fftconvolve(area, patch[::-1, ::-1], mode="valid")
            k = int(np.argmax(corr))
            cyk, cxk = divmod(k, corr.shape[1])
            dy[iy, ix] = cyk - search
            dx[iy, ix] = cxk - search
            m = corr.max()
            conf[iy, ix] = m / (np.linalg.norm(area) * np.linalg.norm(patch) + 1e-6)
    return ys, xs, dy, dx, conf


def densify(ys, xs, dy, dx, conf, shape, smooth):
    """Confidence-weighted smoothing + upsample to full pixel field."""
    h, w = shape
    cw = np.clip(conf, 0, None)
    num_y = gaussian_filter(dy * cw, smooth)
    num_x = gaussian_filter(dx * cw, smooth)
    den = gaussian_filter(cw, smooth) + 1e-6
    fy = num_y / den
    fx = num_x / den
    zy = h / fy.shape[0]
    zx = w / fy.shape[1]
    DY = zoom(fy, (zy, zx), order=1)[:h, :w]
    DX = zoom(fx, (zy, zx), order=1)[:h, :w]
    return DY.astype(np.float32), DX.astype(np.float32)


def warp_array(arr, DY, DX, order):
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.array([yy + DY, xx + DX])
    if arr.ndim == 2:
        return map_coordinates(arr, coords, order=order, mode="nearest")
    out = np.empty_like(arr)
    for c in range(arr.shape[2]):
        out[..., c] = map_coordinates(arr[..., c], coords, order=order, mode="nearest")
    return out


def colorize_diff(mp, gp):
    h, w = mp.shape
    out = np.zeros((h, w, 3), np.uint8)
    m = mp > 0.5
    g = gp > 0.5
    out[m & g] = (40, 40, 40)
    out[m & ~g] = (220, 40, 40)
    out[~m & g] = (40, 90, 220)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=int, default=1024,
                    help="working width for field estimation (height=work/2)")
    ap.add_argument("--sealevel", type=int, default=71)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--win", type=int, default=40)
    ap.add_argument("--search", type=int, default=36)
    ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--apply", action="store_true",
                    help="write corrected full-res layers to out/corrected/")
    args = ap.parse_args()

    w = args.work
    h = w // 2
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[1/4] masks @ {w}x{h}")
    ref = gis_land_mask(w, h)            # target (real geography)
    src = map_land_mask(w, h, args.sealevel)  # current map

    before = colorize_diff(src, ref)
    Image.fromarray(before).save(os.path.join(OUT_DIR, "diff_before.png"))
    b_only_map = int(((src > .5) & (ref < .5)).sum())
    b_only_gis = int(((src < .5) & (ref > .5)).sum())

    print("[2/4] estimating displacement field (block matching)")
    ys, xs, dy, dx, conf = estimate_field(src, ref, args.grid, args.win, args.search)
    DY, DX = densify(ys, xs, dy, dx, conf, (h, w), args.smooth)
    mag = np.sqrt(DY ** 2 + DX ** 2)
    print(f"      displacement: mean={mag.mean():.1f}px max={mag.max():.1f}px "
          f"(@ {w}px-wide frame)")

    # visualize field magnitude
    mvis = (np.clip(mag / (args.search), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(mvis).save(os.path.join(OUT_DIR, "field_magnitude.png"))

    print("[3/4] warping preview + measuring improvement")
    warped = warp_array(src, DY, DX, order=0)
    after = colorize_diff(warped, ref)
    Image.fromarray(after).save(os.path.join(OUT_DIR, "diff_after.png"))
    a_only_map = int(((warped > .5) & (ref < .5)).sum())
    a_only_gis = int(((warped < .5) & (ref > .5)).sum())

    # corrected terrain preview
    terr = np.asarray(Image.open(os.path.join(MAP_DIR, "terrain.bmp"))
                      .convert("RGB").resize((w, h), Image.NEAREST))
    Image.fromarray(warp_array(terr, DY, DX, 0)).save(
        os.path.join(OUT_DIR, "terrain_corrected_preview.png"))

    tot = w * h
    print("\n  land/sea disagreement (lower = better):")
    print(f"    before: map-only {100*b_only_map/tot:.2f}%  "
          f"gis-only {100*b_only_gis/tot:.2f}%  "
          f"total {100*(b_only_map+b_only_gis)/tot:.2f}%")
    print(f"    after : map-only {100*a_only_map/tot:.2f}%  "
          f"gis-only {100*a_only_gis/tot:.2f}%  "
          f"total {100*(a_only_map+a_only_gis)/tot:.2f}%")

    if args.apply:
        print("\n[4/4] applying to full-resolution layers -> out/corrected/")
        cdir = os.path.join(OUT_DIR, "corrected")
        os.makedirs(cdir, exist_ok=True)
        for fname, order in LAYERS.items():
            path = os.path.join(MAP_DIR, fname)
            if not os.path.exists(path):
                print(f"    skip {fname} (missing)")
                continue
            im = Image.open(path)
            mode = im.mode
            fw, fh = im.size
            # scale field to this layer's resolution
            DYf = zoom(DY, (fh / h, fw / w), order=1) * (fh / h)
            DXf = zoom(DX, (fw / w,) if False else (fh / h, fw / w), order=1) * (fw / w)
            arr = np.asarray(im)
            out = warp_array(arr, DYf.astype(np.float32), DXf.astype(np.float32), order)
            Image.fromarray(out, mode=mode).save(os.path.join(cdir, fname))
            print(f"    wrote {fname} ({mode} {fw}x{fh})")
        print("    NOTE: review out/corrected/ before copying over bakasekai/map/")
    else:
        print("\n[4/4] preview only (pass --apply to write full-res corrected layers)")

    print(f"\noutputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
