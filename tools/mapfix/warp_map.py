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

import csv

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from scipy.signal import fftconvolve

from mapframe import gis_land_mask, load_bands

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


def gis_mask(w, h, bands):
    shp = os.path.join(GIS_DIR, "ne_110m_land.shp")
    return np.asarray(gis_land_mask(shp, w, h, bands), dtype=np.float32) / 255.0


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


def load_gcps(path, w, h):
    """Ground control points that steer/override the warp.

    CSV columns: name,src_x,src_y,dst_x,dst_y  (normalized 0..1 frame coords)
    src = where the feature currently sits on the painted map.
    dst = where it should be (real geography).
    Returns list of (dst_y, dst_x, dy, dx) in pixels of the (h,w) frame, where
    the displacement moves output->source: source = output + d.
    """
    gcps = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            _name, sx, sy, dx_, dy_ = row[:5]
            src_x, src_y = float(sx) * w, float(sy) * h
            dst_x, dst_y = float(dx_) * w, float(dy_) * h
            gcps.append((dst_y, dst_x, src_y - dst_y, src_x - dst_x))
    return gcps


def load_protect(path, mask_path, w, h):
    """Mask (0..1) of regions to leave UNWARPED (preserve intentional design).

    From a CSV of normalized rectangles (name,x0,y0,x1,y1) and/or a PNG mask
    (white = protected). Returns a feathered float mask, or None.
    """
    mask = None
    if path and os.path.exists(path):
        mask = np.zeros((h, w), np.float32)
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if not row or row[0].lstrip().startswith("#"):
                    continue
                _name, x0, y0, x1, y1 = row[:5]
                a = slice(int(float(y0) * h), int(float(y1) * h))
                b = slice(int(float(x0) * w), int(float(x1) * w))
                mask[a, b] = 1.0
    if mask_path and os.path.exists(mask_path):
        m = np.asarray(Image.open(mask_path).convert("L").resize((w, h))) / 255.0
        mask = m if mask is None else np.maximum(mask, m)
    if mask is None or mask.max() == 0:
        return None
    return np.clip(gaussian_filter(mask, max(w, h) / 256.0), 0, 1).astype(np.float32)


def densify(ys, xs, dy, dx, conf, shape, smooth, gcps=None, protect=None,
            gcp_radius=0.06):
    """Confidence-weighted smoothing + upsample, then apply GCPs and protection.

    GCPs are applied as LOCAL nudges: each control point adds its residual
    (desired - auto displacement) with a Gaussian falloff (radius = gcp_radius
    of the frame width), decaying to zero away from the point. This fixes
    regions the automatic matcher missed without disturbing the rest of the map.
    Protected regions have their displacement feathered back to zero.
    """
    h, w = shape
    cw = np.clip(conf, 0, None)
    num_y = gaussian_filter(dy * cw, smooth)
    num_x = gaussian_filter(dx * cw, smooth)
    den = gaussian_filter(cw, smooth) + 1e-6
    fy = num_y / den
    fx = num_x / den
    zy = h / fy.shape[0]
    zx = w / fy.shape[1]
    DY = zoom(fy, (zy, zx), order=1)[:h, :w].astype(np.float32)
    DX = zoom(fx, (zy, zx), order=1)[:h, :w].astype(np.float32)

    if gcps:
        yy, xx = np.mgrid[0:h, 0:w]
        sigma = max(gcp_radius * w, 1.0)
        for dst_y, dst_x, ddy, ddx in gcps:
            iy = int(np.clip(dst_y, 0, h - 1))
            ix = int(np.clip(dst_x, 0, w - 1))
            res_y = ddy - DY[iy, ix]      # how far auto is from the desired nudge
            res_x = ddx - DX[iy, ix]
            g = np.exp(-((yy - dst_y) ** 2 + (xx - dst_x) ** 2) / (2 * sigma ** 2))
            DY += (res_y * g).astype(np.float32)
            DX += (res_x * g).astype(np.float32)

    if protect is not None:
        keep = 1.0 - protect
        DY *= keep
        DX *= keep

    return DY, DX


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
    ap.add_argument("--gcps", default=os.path.join(HERE, "gcps.csv"),
                    help="CSV of ground control points (steers/overrides warp)")
    ap.add_argument("--protect", default=os.path.join(HERE, "protect.csv"),
                    help="CSV of normalized rectangles to leave unwarped")
    ap.add_argument("--protect-mask", default=None,
                    help="optional PNG mask (white=protected) to leave unwarped")
    ap.add_argument("--gcp-radius", type=float, default=0.06,
                    help="GCP influence radius as fraction of frame width")
    ap.add_argument("--apply", action="store_true",
                    help="write corrected full-res layers to out/corrected/")
    args = ap.parse_args()

    w = args.work
    h = w // 2
    os.makedirs(OUT_DIR, exist_ok=True)

    bands = load_bands()
    print(f"[1/4] masks @ {w}x{h}")
    print("      bands:", ", ".join(f"{b[0]}[{b[3]:g}..{b[4]:g}]" for b in bands))
    ref = gis_mask(w, h, bands)               # target (real geography, band frame)
    src = map_land_mask(w, h, args.sealevel)  # current map

    before = colorize_diff(src, ref)
    Image.fromarray(before).save(os.path.join(OUT_DIR, "diff_before.png"))
    b_only_map = int(((src > .5) & (ref < .5)).sum())
    b_only_gis = int(((src < .5) & (ref > .5)).sum())

    gcps = load_gcps(args.gcps, w, h) if os.path.exists(args.gcps) else []
    protect = load_protect(args.protect, args.protect_mask, w, h)
    print(f"      control points: {len(gcps)}  "
          f"protected regions: {'yes' if protect is not None else 'none'}")

    print("[2/4] estimating displacement field (block matching)")
    ys, xs, dy, dx, conf = estimate_field(src, ref, args.grid, args.win, args.search)
    DY, DX = densify(ys, xs, dy, dx, conf, (h, w), args.smooth,
                     gcps=gcps, protect=protect, gcp_radius=args.gcp_radius)
    mag = np.sqrt(DY ** 2 + DX ** 2)
    print(f"      displacement: mean={mag.mean():.1f}px max={mag.max():.1f}px "
          f"(@ {w}px-wide frame)")

    # visualize field magnitude
    mvis = (np.clip(mag / (args.search), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(mvis).save(os.path.join(OUT_DIR, "field_magnitude.png"))

    # visualize control points + protected regions on the current terrain
    from PIL import ImageDraw
    terr_ctrl = Image.open(os.path.join(MAP_DIR, "terrain.bmp")).convert("RGB") \
        .resize((w, h), Image.NEAREST)
    d = ImageDraw.Draw(terr_ctrl, "RGBA")
    if protect is not None:
        pm = (protect > 0.5)
        ov = np.zeros((h, w, 4), np.uint8)
        ov[pm] = (255, 255, 0, 90)
        terr_ctrl = Image.alpha_composite(terr_ctrl.convert("RGBA"),
                                          Image.fromarray(ov, "RGBA")).convert("RGB")
        d = ImageDraw.Draw(terr_ctrl)
    for dst_y, dst_x, ddy, ddx in gcps:
        sx, sy = dst_x + ddx, dst_y + ddy        # source (current) location
        d.line([(sx, sy), (dst_x, dst_y)], fill=(255, 0, 0), width=2)
        d.ellipse([sx - 3, sy - 3, sx + 3, sy + 3], outline=(255, 0, 0), width=2)
        d.ellipse([dst_x - 3, dst_y - 3, dst_x + 3, dst_y + 3], fill=(0, 255, 0))
    terr_ctrl.save(os.path.join(OUT_DIR, "control_overlay.png"))

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
