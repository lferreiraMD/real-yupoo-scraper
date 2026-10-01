#!/usr/bin/env python3
"""Score how readable each dial photo is and pick the best one per album.

    python3 ml/rank_dials.py --photos photos_dial.parquet --boxes dial_boxes \\
        --out dial_readability.parquet --best-out best_dial.parquet
    python3 ml/rank_dials.py ... --pick-queue 150 --pick-out pick_paths.txt
    python3 ml/rank_dials.py ... --picks pick_labels.csv     # agreement with human picks

"Best" means the photo that allows the most accurate reading of the dial. All
photos in an album show the same watch, so the size of the dial box measures
how close the camera was; the other terms reward a dial that is fully in
frame, centered, seen head-on, sharp and free of glare. Per photo, from the
detect_dials.py box:

    closeness   dial size (sqrt of box area in pixels) / largest in the album
    frontal     box aspect (short/long side) / roundest box in the album;
                an oblique view squeezes a round dial into an ellipse
    centered    1 - distance of the box center from the image center,
                as a fraction of the half-diagonal
    sharp       box Laplacian variance / sharpest in the album (--sharpness:
                sharpness_256 = box resized to 256 px, the default when
                present; or native-resolution sharpness)
    clear       1 - (glare in the box - least glare in the album) / --glare-span,
                clipped to [0, 1]
    cut         box touches the image border: the dial may be cut off

    readability = w_close*closeness + w_front*frontal + w_center*centered
                  + w_sharp*sharp + w_clear*clear - w_cut*cut
                  - w_weak*weak,   weak = clip((min_det - det_score) / min_det, 0, 1)

Photos the detector barely recognized (det_score below --min-det) are
penalized in proportion, since their box may not be the dial; a gradual
penalty avoids a cliff between near-identical photos. --picks takes the label log
written by label_app.py --pick and reports how often each ranking's top
photo matches the human pick, against simple baselines.
"""

import argparse
import glob
import os
import random

import numpy as np
import pandas as pd

WEIGHTS = {"close": 0.40, "front": 0.20, "center": 0.15, "sharp": 0.15, "clear": 0.10,
           "cut": 0.50, "weak": 0.50}


def load_boxes(dirs):
    parts = [p for d in dirs for p in sorted(glob.glob(os.path.join(d, "part_*.parquet")))]
    if not parts:
        raise SystemExit(f"error: no part_*.parquet under {dirs}")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).drop_duplicates("path")


def features(b, edge=0.005):
    f = b.copy()
    px_w = (f.box_x1 - f.box_x0) * f.width
    px_h = (f.box_y1 - f.box_y0) * f.height
    f["dial_px"] = np.sqrt(px_w * px_h)
    f["aspect"] = np.minimum(px_w, px_h) / np.maximum(px_w, px_h)
    cx = (f.box_x0 + f.box_x1) / 2 * f.width - f.width / 2
    cy = (f.box_y0 + f.box_y1) / 2 * f.height - f.height / 2
    f["center_off"] = np.hypot(cx, cy) / (np.hypot(f.width, f.height) / 2)
    f["cut"] = ((f.box_x0 <= edge) | (f.box_y0 <= edge)
                | (f.box_x1 >= 1 - edge) | (f.box_y1 >= 1 - edge))
    return f


def score(f, w, min_det, sharpness="sharpness_256", glare_span=0.05):
    # continuous ratios, not within-album ranks: in a 2-5 photo album a rank
    # turns a negligible difference into a full-weight one
    if sharpness not in f.columns:
        sharpness = "sharpness"
    g = f.groupby("album_id")
    f["closeness"] = f.dial_px / g.dial_px.transform("max")
    f["frontal"] = f.aspect / g.aspect.transform("max")
    f["centered"] = 1 - f.center_off
    f["sharp"] = f[sharpness] / g[sharpness].transform("max")
    f["clear"] = 1 - ((f.glare - g.glare.transform("min")) / glare_span).clip(0, 1)
    f["weak"] = ((min_det - f.det_score) / min_det).clip(0, 1)  # ramps to 1 as det_score -> 0
    f["readability"] = (w["close"] * f.closeness + w["front"] * f.frontal
                        + w["center"] * f.centered + w["sharp"] * f.sharp
                        + w["clear"] * f.clear - w["cut"] * f.cut - w["weak"] * f.weak)
    f["rank"] = f.groupby("album_id").readability.rank(ascending=False, method="first").astype(int)
    return f


def top1(f, col, picks):
    best = f.sort_values(col, ascending=False).drop_duplicates("album_id").set_index("album_id").path
    return (picks.album_id.map(best) == picks.path).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photos", default="photos_dial.parquet",
                        help="photos_dial.parquet from build_dial_set.py (rows with is_dial)")
    parser.add_argument("--boxes", nargs="+", default=["dial_boxes"],
                        help="detect_dials.py output directories")
    parser.add_argument("--out", default="dial_readability.parquet")
    parser.add_argument("--best-out", default="best_dial.parquet")
    parser.add_argument("--min-det", type=float, default=0.10,
                        help="detector score below which the box is not trusted")
    parser.add_argument("--sharpness", choices=("sharpness_256", "sharpness"),
                        default="sharpness_256", help="which sharpness column ranks photos")
    for k, v in WEIGHTS.items():
        parser.add_argument(f"--w-{k}", type=float, default=v)
    parser.add_argument("--pick-queue", type=int, default=0,
                        help="write a label_app.py --pick queue of N random albums")
    parser.add_argument("--pick-out", default="pick_paths.txt")
    parser.add_argument("--pick-min", type=int, default=3,
                        help="only queue albums with at least this many dial photos")
    parser.add_argument("--picks", default=None, help="label_app.py --pick log to evaluate against")
    args = parser.parse_args()
    w = {k: getattr(args, f"w_{k}") for k in WEIGHTS}

    photos = pd.read_parquet(args.photos)
    photos = photos[photos.is_dial][["path", "album_id", "dial_score"]]
    boxes = load_boxes(args.boxes)
    f = photos.merge(boxes.drop(columns="album_id"), on="path")
    missing = len(photos) - len(f)
    # only rank albums whose every dial photo has a box; a partial album could
    # crown a photo only because its better sibling was not processed yet
    complete = photos.groupby("album_id").size() == f.groupby("album_id").size().reindex(
        photos.album_id.unique(), fill_value=0)
    f = f[f.album_id.isin(complete[complete].index)]
    f = score(features(f), w, args.min_det, args.sharpness)
    f.sort_values(["album_id", "rank"]).to_parquet(args.out, index=False)

    print(f"{len(f)} dial photos scored in {complete.sum()} complete albums "
          f"({missing} dial photos without a box yet; {(~complete).sum()} albums held back)")
    print(f"flags: {f.cut.sum()} boxes touch the border, {(f.det_score < args.min_det).sum()} below det_score {args.min_det}")

    best = f[f["rank"] == 1].set_index("album_id")
    second = f[f["rank"] == 2].set_index("album_id")
    out = pd.DataFrame({
        "best_path": best.path, "readability": best.readability,
        "dial_px": best.dial_px, "frontal": best.frontal, "centered": best.centered,
        "cut": best.cut, "det_score": best.det_score,
        "runner_up_path": second.path.reindex(best.index),
        "margin": best.readability - second.readability.reindex(best.index),
        "candidates": f.groupby("album_id").size().reindex(best.index),
    }).reset_index()
    out.to_parquet(args.best_out, index=False)
    print(f"best photo per album -> {args.best_out}; scores -> {args.out}")
    print(f"best photos that are cut at the border: {out.cut.sum()}, weak detections: "
          f"{(out.det_score < args.min_det).sum()}, single-candidate albums: "
          f"{(out.candidates == 1).sum()}")

    if args.pick_queue:
        eligible = sorted(out[out.candidates >= args.pick_min].album_id)
        chosen = random.Random(7).sample(eligible, min(args.pick_queue, len(eligible)))
        lines = []
        for a in chosen:
            cand = sorted(f[f.album_id == a].path)
            random.Random(a).shuffle(cand)  # no position hint from the ranking
            lines += cand
        with open(args.pick_out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"pick queue: {len(chosen)} albums, {len(lines)} photos -> {args.pick_out}")

    if args.picks:
        p = pd.read_csv(args.picks).sort_values("ts", kind="stable")
        p["album_id"] = p.path.str.split("/").str[0]
        p = p.drop_duplicates("album_id", keep="last")
        p = p[(p.label == "best") & p.album_id.isin(f.album_id)]
        n_cand = f.groupby("album_id").size()
        print(f"\nagreement with {len(p)} human picks (top choice == your pick):")
        print(f"  random photo             {(1 / p.album_id.map(n_cand)).mean():.3f}")
        for col, name in [("dial_score", "highest dial-probe score"), ("dial_px", "largest dial box"),
                          ("readability", "readability score")]:
            print(f"  {name:24s} {top1(f, col, p):.3f}")
        r = p.merge(f[["path", "rank"]], on="path", how="left")
        print(f"  your pick is in the readability top 2: {(r['rank'] <= 2).mean():.3f}")


if __name__ == "__main__":
    main()
