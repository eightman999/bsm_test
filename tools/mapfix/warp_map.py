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
import re

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from scipy.signal import fftconvolve

from mapframe import (find_land_shapes, gis_land_mask, load_bands,
                      province_defs, province_land_mask)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MAP_DIR = os.path.join(REPO, "bakasekai", "map")
SR_DIR = os.path.join(MAP_DIR, "strategicregions")
STATE_DIR = os.path.join(REPO, "bakasekai", "history", "states")
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
    paths, res = find_land_shapes(GIS_DIR)
    if not paths:
        raise SystemExit(
            f"no Natural Earth land shapefile in {GIS_DIR}. "
            "Download ne_10m_land.* (see README).")
    print(f"      GIS reference: {res} ({len(paths)} layer(s): "
          + ", ".join(os.path.basename(p) for p in paths) + ")")
    return np.asarray(gis_land_mask(paths, w, h, bands), dtype=np.float32) / 255.0


def map_land_mask(w, h, sealevel, source="province"):
    """Current-map land/sea mask. source='province' (accurate, from
    provinces.bmp + definition.csv types) or 'heightmap' (legacy threshold)."""
    if source == "province":
        return province_land_mask(
            os.path.join(MAP_DIR, "provinces.bmp"),
            os.path.join(MAP_DIR, "definition.csv"), w, h)
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


def _expand_ids(tokens):
    """Expand a list of id tokens into ints. A token 'A-B' means the inclusive
    range A..B (e.g. '1009-1087'); a plain token is a single id."""
    out = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if "-" in t:
            a, b = t.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(t))
    return out


def _region_file(dirpath, rid):
    """Find the file in dirpath whose name starts with rid then a non-digit
    (e.g. '154-Hokkaido.txt', '1002 - Diego Garcia.txt', '519.txt')."""
    pat = re.compile(rf"^{int(rid)}(\D|$)")
    for fn in sorted(os.listdir(dirpath)):
        if fn.endswith(".txt") and pat.match(fn):
            return os.path.join(dirpath, fn)
    return None


def _provinces_in_file(path):
    """Extract the province IDs from a strategic-region or state file's
    provinces = { ... } block."""
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        txt = f.read()
    m = re.search(r"provinces\s*=\s*{([^}]*)}", txt)
    return [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []


def resolve_protect_regions(path):
    """Parse protect_provinces.csv -> [(name, [enc,...], total_provs), ...].

    CSV rows: name, kind, id, id, ...   kind = sr | state | prov
      sr    -> strategic region file(s); use its province list
      state -> state file(s); use its province list
      prov  -> the ids ARE province ids
    An id token may be a range A-B (inclusive). By default only land/lake
    provinces are kept (sea ignored); append '+sea' (or '+all') to the kind to
    keep sea too -- needed to pin a water feature like a channel/strait in place.
    Shared by load_protect_provinces() and show_protect.py so the two never drift.
    """
    enc_by_id, type_by_id = province_defs(os.path.join(MAP_DIR, "definition.csv"))
    regions = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            row = [c.strip() for c in row if c.strip() != ""]
            if not row or row[0].startswith("#"):
                continue
            name, kind = row[0], row[1].lower()
            include_sea = kind.endswith("+sea") or kind.endswith("+all")
            base = kind.split("+", 1)[0]
            ids = _expand_ids(row[2:])
            pids = []
            if base == "prov":
                pids = ids
            elif base in ("sr", "state"):
                d = SR_DIR if base == "sr" else STATE_DIR
                for rid in ids:
                    fp = _region_file(d, rid)
                    if fp is None:
                        print(f"      (!) {name}: {base} {rid} file not found")
                        continue
                    pids += _provinces_in_file(fp)
            else:
                print(f"      (!) {name}: unknown kind '{kind}' (sr|state|prov[+sea])")
                continue
            encs = [enc_by_id[p] for p in pids
                    if (include_sea or type_by_id.get(p) != "sea") and p in enc_by_id]
            regions.append((name, encs, len(pids)))
    return regions


def load_protect_provinces(path, w, h):
    """Build a protection mask from province membership (exact painted shapes).

    The protected provinces' pixels on provinces.bmp become the mask, so
    intentional baka-distortions (Japan, the Shikoku 'continent', Australia, ...)
    are preserved at their exact painted extent -- not an eyeballed rectangle.
    Returns a feathered float mask (h, w) or None.
    """
    if not path or not os.path.exists(path):
        return None
    protected = set()
    summary = []
    for name, encs, total in resolve_protect_regions(path):
        protected.update(encs)
        summary.append(f"{name}={len(encs)}/{total}")
    if not protected:
        return None
    print("      protect-by-province: " + "; ".join(summary))

    im = Image.open(os.path.join(MAP_DIR, "provinces.bmp")).convert("RGB") \
        .resize((w, h), Image.NEAREST)
    a = np.asarray(im, dtype=np.uint32)
    enc = ((a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]).astype(np.int64)
    sel = np.array(sorted(protected), dtype=np.int64)
    mask = np.isin(enc, sel).astype(np.float32)
    return np.clip(gaussian_filter(mask, max(w, h) / 256.0), 0, 1).astype(np.float32)


def _poly_terms(degree):
    """Total-degree 2D polynomial term exponents (i over y, j over x), i+j<=deg."""
    return [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]


def fit_poly_field(ys, xs, dy, dx, conf, shape, degree):
    """Fit a single smooth low-order polynomial displacement field.

    Block matching gives noisy per-node displacement samples; fitting them to a
    global low-degree polynomial (confidence-weighted least squares) yields a
    field that can only bend/stretch the map at large scale. This removes the
    high-frequency "wobble" the dense field produces and, by construction,
    leaves small local shapes (intentional baka-distortions, single islands)
    unchanged -- a smooth field cannot reshape one island on its own.
    """
    h, w = shape
    YS, XS = np.meshgrid(ys, xs, indexing="ij")
    yn = (YS / (h - 1)) * 2 - 1                  # normalize node coords to [-1,1]
    xn = (XS / (w - 1)) * 2 - 1
    wts = np.clip(conf, 0, None).ravel()
    sel = wts > 1e-6
    terms = _poly_terms(degree)
    A = np.stack([(yn ** i * xn ** j).ravel() for (i, j) in terms], axis=1)
    Aw = A[sel] * wts[sel, None]

    def solve(vals):
        bw = vals.ravel()[sel] * wts[sel]
        coef, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        return coef

    cy, cx = solve(dy), solve(dx)
    yy, xx = np.mgrid[0:h, 0:w]
    ynf = ((yy / (h - 1)) * 2 - 1).astype(np.float32)
    xnf = ((xx / (w - 1)) * 2 - 1).astype(np.float32)
    DY = np.zeros((h, w), np.float32)
    DX = np.zeros((h, w), np.float32)
    for (i, j), c_y, c_x in zip(terms, cy, cx):
        basis = (ynf ** i * xnf ** j)
        DY += (c_y * basis).astype(np.float32)
        DX += (c_x * basis).astype(np.float32)
    return DY, DX


def densify(ys, xs, dy, dx, conf, shape, smooth, gcps=None, protect=None,
            gcp_radius=0.06, model="field", degree=4):
    """Build the displacement field, then apply GCPs and protection.

    model='field' (default): confidence-weighted Gaussian smoothing + upsample
      of the dense node displacements. Follows local coastlines closely but can
      introduce visible wobble.
    model='poly': fit one smooth low-degree polynomial to the node samples
      (see fit_poly_field). Corrects only the large-scale (aspect-ratio) warp,
      no wobble.

    GCPs are applied as LOCAL nudges: each control point adds its residual
    (desired - auto displacement) with a Gaussian falloff (radius = gcp_radius
    of the frame width), decaying to zero away from the point. This fixes
    regions the automatic matcher missed without disturbing the rest of the map.
    Protected regions have their displacement feathered back to zero.
    """
    h, w = shape
    if model == "poly":
        DY, DX = fit_poly_field(ys, xs, dy, dx, conf, shape, degree)
    else:
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


def _encode_rgb(a):
    a = a.astype(np.uint32)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


def restamp_lost_provinces(out_rgb, src_rgb, DYf, DXf):
    """Ensure every province colour (= ID) that existed in the source survives
    the warp. Backward warping can squeeze provinces out of existence, which
    HOI4 rejects (missing definition.csv IDs). For each lost province we claim a
    single pixel near its warped location, but ONLY from a victim that still has
    other pixels -- so the operation never eliminates any other province (it is
    provably non-destructive: every previously-present ID stays present and each
    lost ID becomes present). Returns (out_rgb, n_lost, n_fixed)."""
    s = _encode_rgb(src_rgb)
    o = _encode_rgb(out_rgb)
    before = set(np.unique(s).tolist())
    vals, cnts = np.unique(o, return_counts=True)
    count = dict(zip(vals.tolist(), cnts.tolist()))
    lost = [v for v in before if v not in count]
    if not lost:
        return out_rgb, 0, 0
    h, w = s.shape

    def claim(idv, yy, xx):
        victim = int(o[yy, xx])
        count[victim] -= 1
        out_rgb[yy, xx, 0] = (idv >> 16) & 255
        out_rgb[yy, xx, 1] = (idv >> 8) & 255
        out_rgb[yy, xx, 2] = idv & 255
        o[yy, xx] = idv
        count[idv] = count.get(idv, 0) + 1

    fixed = 0
    for idv in lost:
        ys, xs = np.where(s == idv)
        oy = np.clip(np.round(ys - DYf[ys, xs]).astype(int), 0, h - 1)
        ox = np.clip(np.round(xs - DXf[ys, xs]).astype(int), 0, w - 1)
        placed = False
        for k in range(len(oy)):               # prefer the warped footprint
            y, x = int(oy[k]), int(ox[k])
            if count.get(int(o[y, x]), 0) > 1:
                claim(idv, y, x)
                placed = True
                break
        if not placed:                         # spiral out for a safe victim
            cy, cx = int(oy[0]), int(ox[0])
            for r in range(1, 32):
                for ddy in range(-r, r + 1):
                    for ddx in range(-r, r + 1):
                        y = min(max(cy + ddy, 0), h - 1)
                        x = min(max(cx + ddx, 0), w - 1)
                        if count.get(int(o[y, x]), 0) > 1:
                            claim(idv, y, x)
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
        fixed += placed
    return out_rgb, len(lost), fixed


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
    ap.add_argument("--landsea", choices=["province", "heightmap"],
                    default="province",
                    help="how to classify the current map's land/sea: "
                         "province = provinces.bmp + definition.csv types "
                         "(accurate, default); heightmap = legacy threshold")
    ap.add_argument("--sealevel", type=int, default=71,
                    help="sea threshold for --landsea heightmap only")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--win", type=int, default=40)
    ap.add_argument("--search", type=int, default=36)
    ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--model", choices=["poly", "field"], default="poly",
                    help="poly = smooth low-order warp (no wobble, large-scale "
                         "only, default); field = dense block-match field "
                         "(follows coastlines closely but can wobble)")
    ap.add_argument("--poly-degree", type=int, default=4,
                    help="polynomial degree for --model poly (3-5 typical)")
    ap.add_argument("--gcps", default=os.path.join(HERE, "gcps.csv"),
                    help="CSV of ground control points (steers/overrides warp)")
    ap.add_argument("--protect", default=os.path.join(HERE, "protect.csv"),
                    help="CSV of normalized rectangles to leave unwarped")
    ap.add_argument("--protect-provinces",
                    default=os.path.join(HERE, "protect_provinces.csv"),
                    help="CSV of regions defined by strategic-region/state/"
                         "province IDs to leave unwarped (exact painted shapes)")
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
    src = map_land_mask(w, h, args.sealevel, args.landsea)  # current map
    print(f"      land/sea source: {args.landsea}")

    before = colorize_diff(src, ref)
    Image.fromarray(before).save(os.path.join(OUT_DIR, "diff_before.png"))
    b_only_map = int(((src > .5) & (ref < .5)).sum())
    b_only_gis = int(((src < .5) & (ref > .5)).sum())

    gcps = load_gcps(args.gcps, w, h) if os.path.exists(args.gcps) else []
    protect = load_protect(args.protect, args.protect_mask, w, h)
    protect_prov = load_protect_provinces(args.protect_provinces, w, h)
    if protect_prov is not None:
        protect = protect_prov if protect is None else np.maximum(protect, protect_prov)
    print(f"      control points: {len(gcps)}  "
          f"protected regions: {'yes' if protect is not None else 'none'}")
    if protect is not None:
        Image.fromarray((protect * 255).astype(np.uint8)).save(
            os.path.join(OUT_DIR, "protect_mask.png"))

    print("[2/4] estimating displacement field (block matching)")
    ys, xs, dy, dx, conf = estimate_field(src, ref, args.grid, args.win, args.search)
    DY, DX = densify(ys, xs, dy, dx, conf, (h, w), args.smooth,
                     gcps=gcps, protect=protect, gcp_radius=args.gcp_radius,
                     model=args.model, degree=args.poly_degree)
    print(f"      field model: {args.model}"
          + (f" (degree {args.poly_degree})" if args.model == "poly" else ""))
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
            # scale the displacement field to this layer's resolution
            DYf = (zoom(DY, (fh / h, fw / w), order=1) * (fh / h)).astype(np.float32)
            DXf = (zoom(DX, (fh / h, fw / w), order=1) * (fw / w)).astype(np.float32)
            arr = np.asarray(im)
            out = warp_array(arr, DYf, DXf, order)

            note = ""
            if fname == "provinces.bmp" and arr.ndim == 3:
                # province IDs are encoded as colours; restamp any squeezed out
                out = out.copy()
                out, lost, fixed = restamp_lost_provinces(out, arr, DYf, DXf)
                kept = len(np.unique(_encode_rgb(out)))
                src_n = len(np.unique(_encode_rgb(arr)))
                note = (f"  provinces: {src_n} IDs, {lost} squeezed -> "
                        f"{fixed} restamped, {kept}/{src_n} present")
                if kept < src_n:
                    note += "  (!) still missing -- inspect manually"

            # Save preserving the exact HOI4 BMP format (see map-modding rules):
            #  - 8-bit indexed (terrain/rivers): KEEP the original colormap.
            #  - 8-bit greyscale (heightmap): plain L.
            #  - 24-bit RGB (provinces/world_normal): direct colours = IDs.
            if mode == "P":
                outim = Image.fromarray(out.astype(np.uint8), mode="P")
                outim.putpalette(im.getpalette())
            else:
                outim = Image.fromarray(out, mode=mode)
            outim.save(os.path.join(cdir, fname))
            print(f"    wrote {fname} ({mode} {fw}x{fh}){note}")
        print("    NOTE: review out/corrected/ before copying over bakasekai/map/")
        print("    NOTE: provinces.bmp must stay 24-bit uncompressed; terrain/")
        print("          rivers must keep their colormap; then regenerate")
        print("          positions (buildings.txt/unitstacks.txt) via the nudger.")
    else:
        print("\n[4/4] preview only (pass --apply to write full-res corrected layers)")

    print(f"\noutputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
