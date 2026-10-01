#!/usr/bin/env python3
"""Check the readability score with synthetic degradations of known-good dial photos.

    python3 ml/check_readability.py photos_big --labels labels.csv border_labels.csv --n 60

Takes photos a human labeled "yes" (dial clearly visible), and for each one
builds degraded copies in memory: the camera farther away, the dial off
center, an oblique view, defocus blur, glare on the crystal, and the dial
cut off at the image border. Every copy goes through the same detector and
pixel statistics as the real photos (detect_dials.py), and each original +
copy pair is scored as a two-photo album (rank_dials.py). A sound score
prefers the original in every pair. Nothing is written to disk except the
optional --out table; no image leaves memory.
"""

import argparse
import io
import os
import random
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_dials import box_stats, detect, load_detector  # noqa: E402
from rank_dials import WEIGHTS, features, score  # noqa: E402

EXPECT = {  # degradation -> the term that should drop (or flag that should rise)
    "far": "closeness", "off_center": "centered", "oblique": "frontal",
    "blur": "sharp", "glare": "clear", "cut": "cut",
}


def fill_color(img):
    a = np.asarray(img)
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return tuple(int(v) for v in np.median(edge, axis=0))


def canvas_paste(img, part, xy):
    out = Image.new("RGB", img.size, fill_color(img))
    out.paste(part, xy)
    return out


def degrade(img, box, kind):
    w, h = img.size
    x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    if kind == "far":  # same scene from ~1.7x the distance
        s = 0.6
        small = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        return canvas_paste(img, small, ((w - small.width) // 2, (h - small.height) // 2))
    if kind == "off_center":  # move the dial toward a corner, keeping it in frame
        dx = (w - 2 - x1) if cx < w / 2 else -(x0 - 2)
        dy = (h - 2 - y1) if cy < h / 2 else -(y0 - 2)
        return canvas_paste(img, img, (int(dx * 0.9), int(dy * 0.9)))
    if kind == "oblique":  # squeeze across the dial: a round dial seen at ~53 degrees
        s = 0.6
        narrow = img.resize((int(w * s), h), Image.LANCZOS)
        return canvas_paste(img, narrow, (int(cx - cx * s), 0))
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=3))
    if kind == "glare":  # a blown-out reflection over the middle of the dial
        out = img.copy()
        ImageDraw.Draw(out).ellipse([cx - 0.3 * bw, cy - 0.22 * bh, cx + 0.3 * bw, cy + 0.22 * bh],
                                    fill=(255, 255, 255))
        return out
    if kind == "cut":  # 40% of the dial pushed out past the left border
        return canvas_paste(img, img, (int(-(x0 + 0.4 * bw)), 0))
    raise ValueError(kind)


def jpeg(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def measure(detector, img):
    pixels = detector[0].image_processor(images=img, return_tensors="pt").pixel_values
    (box, det_score), = detect(detector, pixels, [img.size])
    return {"width": img.width, "height": img.height, "box_x0": box[0], "box_y0": box[1],
            "box_x1": box[2], "box_y1": box[3], "det_score": det_score,
            **box_stats(np.asarray(img), box)}, box


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photos_dir")
    parser.add_argument("--labels", nargs="+", required=True, help="label logs with yes labels")
    parser.add_argument("--n", type=int, default=60, help="originals to degrade")
    parser.add_argument("--min-det", type=float, default=0.10)
    parser.add_argument("--out", default=None, help="optional per-pair results table (parquet)")
    args = parser.parse_args()

    lab = pd.concat([pd.read_csv(p) for p in args.labels]).sort_values("ts", kind="stable")
    lab = lab.drop_duplicates("path", keep="last")
    yes = sorted(p for p in lab[lab.label == "yes"].path
                 if os.path.exists(os.path.join(args.photos_dir, p)))
    picked = random.Random(11).sample(yes, min(args.n, len(yes)))
    print(f"{len(yes)} human-'yes' photos available; degrading {len(picked)}", flush=True)

    detector = load_detector()
    rows = []
    for k, rel in enumerate(picked):
        img = jpeg(Image.open(os.path.join(args.photos_dir, rel)).convert("RGB"))
        base, box = measure(detector, img)
        for kind in EXPECT:
            var, _ = measure(detector, jpeg(degrade(img, box, kind)))
            album = f"{rel}|{kind}"
            rows.append({"path": f"{album}|original", "album_id": album, "kind": kind,
                         "is_original": True, **base})
            rows.append({"path": f"{album}|degraded", "album_id": album, "kind": kind,
                         "is_original": False, **var})
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(picked)}", flush=True)

    d = features(pd.DataFrame(rows))
    print("\noriginal preferred over its degraded copy (share of pairs):")
    print(f"  {'degradation':12s} {'native sharpness':>17s} {'256px sharpness':>16s}   expected term moved")
    results = {}
    for sharp in ("sharpness", "sharpness_256"):
        results[sharp] = score(d.copy(), WEIGHTS, args.min_det, sharp)
    for kind, term in EXPECT.items():
        line = f"  {kind:12s}"
        for sharp in ("sharpness", "sharpness_256"):
            r = results[sharp][results[sharp].kind == kind]
            o = r[r.is_original].set_index("album_id")
            v = r[~r.is_original].set_index("album_id")
            line += f" {(o.readability > v.readability).mean():17.2f}"
        r = results["sharpness_256"][results["sharpness_256"].kind == kind]
        o = r[r.is_original].set_index("album_id")[term].astype(float)
        v = r[~r.is_original].set_index("album_id")[term].astype(float)
        moved = (v > o).mean() if term == "cut" else (v < o).mean()
        line += f"   {term} {'flagged' if term == 'cut' else 'lower'} in {moved:.2f}"
        print(line)
    for kind in ("far", "blur"):
        r = results["sharpness_256"][results["sharpness_256"].kind == kind]
        o = r[r.is_original].set_index("album_id")
        v = r[~r.is_original].set_index("album_id")
        print(f"  {kind}: native sharpness lower in {(v.sharpness < o.sharpness).mean():.2f} of pairs, "
              f"256px sharpness lower in {(v.sharpness_256 < o.sharpness_256).mean():.2f}")
    if args.out:
        results["sharpness_256"].to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
