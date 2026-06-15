#!/usr/bin/env python3
"""Verify that warped/corrected map layers are HOI4-loadable.

Runs after `warp_map.py --apply` (which writes corrected layers into
out/corrected/). It does NOT touch bakasekai/map/. It checks the things that
make HOI4 reject a map at load time, and reports quality issues that only
produce warnings:

HARD checks (any failure -> exit 1, i.e. the map would not load cleanly):
  * dimensions of every corrected layer match the live layer.
  * provinces.bmp is 24-bit RGB (colour = province ID).
  * terrain.bmp / rivers.bmp keep their 8-bit indexed palette (unchanged).
  * heightmap.bmp is 8-bit greyscale; world_normal.bmp is 24-bit RGB.
  * ID survival: every colour in definition.csv is still present in the
    corrected provinces.bmp (a missing ID = HOI4 "province not found" crash).
  * no invalid colours: every colour in provinces.bmp maps to a definition.csv
    entry (a stray/blended colour = undefined province).

QUALITY checks (reported, never fail the run):
  * province fragmentation: provinces split into multiple disconnected blobs by
    the warp (HOI4 logs these and they look wrong in-game).
  * tiny provinces: provinces reduced to a handful of pixels (often the ones the
    restamper had to rescue; flag for manual cleanup in nudge).
  * before/after deltas vs the live provinces.bmp so you can see what the warp
    introduced rather than what the map already had.

Usage:
  python3 verify_apply.py                       # verify out/corrected/
  python3 verify_apply.py --dir out/corrected   # explicit
  python3 verify_apply.py --no-compare          # skip live-map comparison (faster)
  python3 verify_apply.py --tiny 8              # tiny-province pixel threshold
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MAP_DIR = os.path.join(REPO, "bakasekai", "map")
OUT_DIR = os.path.join(HERE, "out")

Image.MAX_IMAGE_PIXELS = None

# Expected formats per HOI4 map-modding rules.
#   mode_ok: acceptable PIL modes; palette: must the colormap be preserved?
EXPECT = {
    "provinces.bmp":    {"mode_ok": ("RGB",),       "palette": False},
    "terrain.bmp":      {"mode_ok": ("P",),         "palette": True},
    "rivers.bmp":       {"mode_ok": ("P",),         "palette": True},
    "heightmap.bmp":    {"mode_ok": ("L",),         "palette": False},
    "world_normal.bmp": {"mode_ok": ("RGB",),       "palette": False},
}

EIGHT = np.ones((3, 3), dtype=bool)   # 8-connectivity structuring element


class Report:
    def __init__(self):
        self.hard_fail = False

    def ok(self, msg):
        print(f"  \033[32m[ OK ]\033[0m {msg}")

    def fail(self, msg):
        self.hard_fail = True
        print(f"  \033[31m[FAIL]\033[0m {msg}")

    def warn(self, msg):
        print(f"  \033[33m[warn]\033[0m {msg}")

    def info(self, msg):
        print(f"         {msg}")


def load_definition(path):
    """definition.csv: id;r;g;b;type;coastal;terrain;continent.
    Returns (enc_set, id_by_enc) where enc = (r<<16)|(g<<8)|b."""
    enc_set = set()
    id_by_enc = {}
    with open(path, newline="") as f:
        for row in csv.reader(f, delimiter=";"):
            if len(row) < 4 or not row[0].strip().lstrip("-").isdigit():
                continue
            pid, r, g, b = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
            enc = (r << 16) | (g << 8) | b
            enc_set.add(enc)
            id_by_enc[enc] = pid
    return enc_set, id_by_enc


def encode(rgb):
    a = rgb.astype(np.uint32)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


def component_stats(enc, cap_bbox=8_000_000):
    """Per-province connected-component analysis on an encoded (h,w) image.

    Returns dict: enc_value -> (n_pixels, n_components, largest_component_px).
    Provinces whose bounding box exceeds cap_bbox px are recorded with
    n_components = -1 (skipped to bound runtime; large provinces are not the
    fragmentation cases we care about).
    """
    uniq, inv = np.unique(enc, return_inverse=True)
    lab = inv.reshape(enc.shape).astype(np.int32)
    objs = ndimage.find_objects(lab + 1)        # label i -> objs[i] for lab==i
    stats = {}
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        sub = (lab[sl] == i)
        npix = int(sub.sum())
        bbox_area = sub.size
        if bbox_area > cap_bbox:
            stats[int(uniq[i])] = (npix, -1, npix)
            continue
        cc, ncomp = ndimage.label(sub, structure=EIGHT)
        if ncomp <= 1:
            largest = npix
        else:
            sizes = np.bincount(cc.ravel())[1:]
            largest = int(sizes.max())
        stats[int(uniq[i])] = (npix, ncomp, largest)
    return stats


def verify_provinces(rep, corrected_path, live_path, defs_path, tiny, compare):
    enc_set, id_by_enc = load_definition(defs_path)
    print(f"\ndefinition.csv: {len(enc_set)} province colours")

    im = Image.open(corrected_path)
    if im.mode != "RGB":
        rep.fail(f"provinces.bmp mode is {im.mode}, expected RGB (24-bit)")
        im = im.convert("RGB")
    rgb = np.asarray(im)
    enc = encode(rgb)
    present = set(np.unique(enc).tolist())

    # --- load the live (source) map: the warp's INPUT ----------------------
    # The warp's guarantee is "no painted province is lost", which must be
    # measured against the SOURCE image, not definition.csv: definition.csv may
    # list colours that were never painted (e.g. ID 0 = (0,0,0)), and those
    # absences are pre-existing, not introduced by the warp.
    src_present = None
    live_rgb = None
    if os.path.exists(live_path):
        live_rgb = np.asarray(Image.open(live_path).convert("RGB"))
        if live_rgb.shape == rgb.shape:
            src_present = set(np.unique(encode(live_rgb)).tolist())
        else:
            rep.warn(f"live provinces.bmp size {live_rgb.shape[1::-1]} != corrected "
                     f"{rgb.shape[1::-1]}; ID-survival falls back to definition.csv")
            live_rgb = None

    # --- ID survival -------------------------------------------------------
    if src_present is not None:
        lost = src_present - present
        if lost:
            rep.fail(f"{len(lost)} provinces painted in the live map are MISSING after "
                     f"the warp (HOI4 would crash). e.g. IDs "
                     + ", ".join(str(id_by_enc.get(m, '?')) for m in list(lost)[:10]))
        else:
            rep.ok(f"all {len(src_present)} painted provinces survived the warp")
        never = enc_set - src_present
        if never:
            rep.info(f"{len(never)} definition.csv IDs are not painted in the live map "
                     f"either (pre-existing, not caused by the warp): IDs "
                     + ", ".join(str(id_by_enc[m]) for m in sorted(never)[:10]))
    else:
        missing = enc_set - present
        if missing:
            rep.fail(f"{len(missing)} definition.csv IDs MISSING from provinces.bmp "
                     f"(no usable live map to tell warp-loss from never-painted; "
                     f"HOI4 may crash). e.g. IDs "
                     + ", ".join(str(id_by_enc[m]) for m in list(missing)[:10]))
        else:
            rep.ok(f"all {len(enc_set)} definition.csv IDs present")

    # --- invalid colours ---------------------------------------------------
    invalid = present - enc_set
    if invalid:
        rep.fail(f"{len(invalid)} colours in provinces.bmp are NOT in "
                 f"definition.csv (undefined provinces). e.g. "
                 + ", ".join(f"#{v:06x}" for v in list(invalid)[:10]))
    else:
        rep.ok("no undefined colours (every colour maps to definition.csv)")

    # --- fragmentation / tiny ---------------------------------------------
    print("\n  connected-component analysis (corrected)...")
    cs = component_stats(enc)
    frag = {v: s for v, s in cs.items() if s[1] > 1}
    tinyp = {v: s for v, s in cs.items() if 0 < s[0] <= tiny}
    rep.info(f"provinces analysed: {len(cs)}")
    if frag:
        rep.warn(f"{len(frag)} provinces are fragmented (>1 disconnected blob)")
    else:
        rep.ok("no fragmented provinces")
    if tinyp:
        rep.warn(f"{len(tinyp)} provinces are tiny (<= {tiny}px) -- nudge cleanup advised")
    else:
        rep.ok(f"no tiny provinces (<= {tiny}px)")

    # --- before/after delta vs the live map -------------------------------
    if compare and live_rgb is not None:
        print("\n  comparing against live bakasekai/map/provinces.bmp...")
        cs0 = component_stats(encode(live_rgb))
        frag0 = sum(1 for s in cs0.values() if s[1] > 1)
        tiny0 = sum(1 for s in cs0.values() if 0 < s[0] <= tiny)
        rep.info(f"fragmented: live {frag0} -> corrected {len(frag)} "
                 f"(delta {len(frag) - frag0:+d})")
        rep.info(f"tiny (<= {tiny}px): live {tiny0} -> corrected {len(tinyp)} "
                 f"(delta {len(tinyp) - tiny0:+d})")

    # worst offenders, most fragments first
    if frag:
        worst = sorted(frag.items(), key=lambda kv: kv[1][1], reverse=True)[:12]
        print("\n  most-fragmented provinces (id: blobs, px, largest-blob px):")
        for v, (npix, nc, largest) in worst:
            print(f"    id {id_by_enc.get(v, '?'):>6}: {nc:>3} blobs, "
                  f"{npix:>6} px, largest {largest} px")


def verify_format(rep, corrected_dir):
    print("\nformat & dimension checks:")
    for fname, spec in EXPECT.items():
        cpath = os.path.join(corrected_dir, fname)
        lpath = os.path.join(MAP_DIR, fname)
        if not os.path.exists(cpath):
            rep.warn(f"{fname}: not in corrected dir (was it applied?)")
            continue
        cim = Image.open(cpath)
        if cim.mode not in spec["mode_ok"]:
            rep.fail(f"{fname}: mode {cim.mode}, expected {spec['mode_ok']}")
        else:
            rep.ok(f"{fname}: mode {cim.mode}")
        if os.path.exists(lpath):
            lim = Image.open(lpath)
            if cim.size != lim.size:
                rep.fail(f"{fname}: size {cim.size} != live {lim.size}")
            if spec["palette"]:
                if cim.getpalette() != lim.getpalette():
                    rep.fail(f"{fname}: palette changed (HOI4 needs the original "
                             f"colormap intact)")
                else:
                    rep.ok(f"{fname}: palette preserved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(OUT_DIR, "corrected"),
                    help="directory of corrected layers to verify")
    ap.add_argument("--definition", default=os.path.join(MAP_DIR, "definition.csv"))
    ap.add_argument("--tiny", type=int, default=8,
                    help="pixel count at/below which a province is 'tiny'")
    ap.add_argument("--no-compare", action="store_true",
                    help="skip the before/after comparison vs the live map")
    args = ap.parse_args()

    cdir = args.dir
    if not os.path.isdir(cdir):
        print(f"error: {cdir} does not exist. Run: warp_map.py --apply", file=sys.stderr)
        return 2
    print(f"verifying corrected layers in: {cdir}")

    rep = Report()
    verify_format(rep, cdir)

    cprov = os.path.join(cdir, "provinces.bmp")
    if os.path.exists(cprov):
        verify_provinces(rep, cprov, os.path.join(MAP_DIR, "provinces.bmp"),
                         args.definition, args.tiny, not args.no_compare)
    else:
        rep.fail("provinces.bmp not found in corrected dir")

    print("\n" + "=" * 60)
    if rep.hard_fail:
        print("RESULT: \033[31mFAIL\033[0m -- hard checks failed; this map would "
              "NOT load cleanly in HOI4. Fix before copying to bakasekai/map/.")
        return 1
    print("RESULT: \033[32mPASS\033[0m -- hard checks passed. Review the quality "
          "warnings above, then (if happy) copy out/corrected/ to bakasekai/map/")
    print("        and regenerate derived data (positions/buildings) via nudge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
