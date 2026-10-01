#!/usr/bin/env python3
"""Score how readable each dial photo is and pick the best one per watch.

    python3 ml/rank_dials.py --photos photos_dial.parquet --boxes dial_boxes \\
        --watches watches.parquet --out dial_readability.parquet --best-out best_dial.parquet
    python3 ml/rank_dials.py ... --picks pick_labels.csv     # agreement with human picks

"Best" means the photo that allows the most accurate reading of the dial.
Photos are compared within one watch: a whole album, or one of the watches
split_watches.py found in it (--watches). All of a watch's photos show the
same object, so the size of the dial box measures how close the camera was;
the other terms reward a dial that is fully in frame, centered, seen head-on,
sharp and free of glare. Per photo, from the
detect_dials.py box:

    closeness   dial size (sqrt of box area in pixels) / largest of the watch
    frontal     box aspect (short/long side) / roundest box of the watch;
                an oblique view squeezes a round dial into an ellipse
    centered    1 - distance of the box center from the image center,
                as a fraction of the half-diagonal
    sharp       box Laplacian variance / sharpest of the watch (--sharpness:
                sharpness_256 = box resized to 256 px, the default when
                present; or native-resolution sharpness)
    clear       1 - (glare in the box - least glare of the watch) / --glare-span,
                clipped to [0, 1]
    cut         box touches the image border: the dial may be cut off

    readability = w_close*closeness + w_front*frontal + w_center*centered
                  + w_sharp*sharp + w_clear*clear - w_cut*cut
                  - w_weak*weak,   weak = clip((min_det - det_score) / min_det, 0, 1)

Photos the detector barely recognized (det_score below --min-det) are
penalized in proportion, since their box may not be the dial; a gradual
penalty avoids a cliff between near-identical photos. --picks takes the log
written by label_app.py --pick: for every watch that holds exactly one human
pick, it reports how often each ranking's top photo is that pick, against
simple baselines; --final-out then writes the best photo per watch with the
human answers applied to every reviewed album (picks override the model, and
several picks on one unsplit screen count as separate watches).
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

# Fitted to 467 human picks (album-grouped CV): a model over all terms agreed
# with the pick 91.0% of the time, the largest dial alone 90.6%, the earlier
# hand-set mix (close .40, front .20, center .15, sharp .15, clear .10) 88.0%.
# So the default ranks by closeness, with the cut and weak-box penalties kept
# as safety terms; the other weights remain available as options.
WEIGHTS = {"close": 1.0, "front": 0.0, "center": 0.0, "sharp": 0.0, "clear": 0.0,
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


def score(f, w, min_det, sharpness="sharpness_256", glare_span=0.05, key="album_id"):
    # continuous ratios, not within-group ranks: in a 2-5 photo group a rank
    # turns a negligible difference into a full-weight one
    if sharpness not in f.columns:
        sharpness = "sharpness"
    g = f.groupby(key)
    f["closeness"] = f.dial_px / g.dial_px.transform("max")
    f["frontal"] = f.aspect / g.aspect.transform("max")
    f["centered"] = 1 - f.center_off
    f["sharp"] = f[sharpness] / g[sharpness].transform("max")
    f["clear"] = 1 - ((f.glare - g.glare.transform("min")) / glare_span).clip(0, 1)
    f["weak"] = ((min_det - f.det_score) / min_det).clip(0, 1)  # ramps to 1 as det_score -> 0
    f["readability"] = (w["close"] * f.closeness + w["front"] * f.frontal
                        + w["center"] * f.centered + w["sharp"] * f.sharp
                        + w["clear"] * f.clear - w["cut"] * f.cut - w["weak"] * f.weak)
    f["rank"] = f.groupby(key).readability.rank(ascending=False, method="first").astype(int)
    return f


def read_answers(path):
    """Latest saved answer per screen: one row per picked photo, or one none/unsure row."""
    log = pd.read_csv(path, dtype={"screen": str, "path": str})
    if "screen" not in log:  # first-version log: path,label,ts with one pick per album
        log["screen"] = log.path.str.split("/").str[0]
        log["save"] = range(len(log))
    return log[log.save == log.groupby("screen").save.transform("max")]


def read_picks(path):
    a = read_answers(path)
    return a[(a.label == "best") & a.path.notna()][["screen", "path"]]


def final_table(best, f, answers):
    """Model's best photo per watch, replaced by the human answer wherever an album was reviewed."""
    answers = answers.assign(album_id=answers.screen.str.rstrip("abcdefghij"))
    reviewed = set(answers.album_id)
    rows = [{"album_id": r.album_id, "watch_id": r.watch_id, "best_path": r.best_path,
             "source": "model"} for r in best.itertuples() if r.album_id not in reviewed]
    watch_of = f.set_index("path").watch_id
    num = lambda p: int(p.split("/")[1].split("_")[0])
    for album, a in answers.groupby("album_id"):
        picks = sorted(a[a.label == "best"].path.dropna(), key=num)
        ids = [watch_of.get(p) for p in picks]
        if len(set(ids)) < len(ids):  # several picks inside one unsplit watch: one watch each
            ids = [album + "abcdefghij"[k] for k in range(len(picks))] if len(picks) > 1 else [album]
        rows += [{"album_id": album, "watch_id": i, "best_path": p, "source": "human"}
                 for i, p in zip(ids, picks)]
        rows += [{"album_id": album, "watch_id": s, "best_path": None, "source": f"human: {lab}"}
                 for s, lab in a[a.label != "best"][["screen", "label"]].itertuples(index=False)]
    return pd.DataFrame(rows).sort_values(["album_id", "watch_id"]).reset_index(drop=True)


def evaluate(f, picks):
    p = picks.merge(f[["path", "watch_id", "rank"]], on="path")
    per_watch = p.groupby("watch_id").path.transform("size")
    one = p[per_watch == 1]
    n_cand = f.groupby("watch_id").size()
    print(f"\nagreement with your picks: {len(one)} watches hold exactly one pick "
          f"({(per_watch > 1).sum()} picks share a watch with another pick and are skipped)")
    print(f"  random photo               {(1 / one.watch_id.map(n_cand)).mean():.3f}")
    for col, name in [("dial_score", "highest dial-probe score"), ("sharpness_256", "sharpest dial"),
                      ("dial_px", "largest dial box"), ("readability", "readability score")]:
        if col not in f:
            continue
        top = f.sort_values(col, ascending=False).drop_duplicates("watch_id").set_index("watch_id").path
        print(f"  {name:26s} {(one.watch_id.map(top) == one.path).mean():.3f}")
    print(f"  your pick in readability top 2: {(one['rank'] <= 2).mean():.3f}")
    return one


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photos", default="photos_dial.parquet",
                        help="photos_dial.parquet from build_dial_set.py (rows with is_dial)")
    parser.add_argument("--boxes", nargs="+", default=["dial_boxes"],
                        help="detect_dials.py output directories")
    parser.add_argument("--watches", default=None,
                        help="watches.parquet from split_watches.py; without it, one watch per album")
    parser.add_argument("--out", default="dial_readability.parquet")
    parser.add_argument("--best-out", default="best_dial.parquet")
    parser.add_argument("--min-det", type=float, default=0.10,
                        help="detector score below which the box is not trusted")
    parser.add_argument("--sharpness", choices=("sharpness_256", "sharpness"),
                        default="sharpness_256", help="which sharpness column ranks photos")
    for k, v in WEIGHTS.items():
        parser.add_argument(f"--w-{k}", type=float, default=v)
    parser.add_argument("--picks", default=None, help="label_app.py --pick log to evaluate against")
    parser.add_argument("--final-out", default=None,
                        help="also write the best photo per watch with human answers applied")
    args = parser.parse_args()
    w = {k: getattr(args, f"w_{k}") for k in WEIGHTS}

    photos = pd.read_parquet(args.photos)
    photos = photos[photos.is_dial][["path", "album_id", "dial_score"]]
    photos["album_id"] = photos.album_id.astype(str)
    if args.watches:
        photos = photos.merge(pd.read_parquet(args.watches)[["path", "watch_id"]], on="path")
    else:
        photos["watch_id"] = photos.album_id
    boxes = load_boxes(args.boxes)
    f = photos.merge(boxes.drop(columns="album_id"), on="path")
    missing = len(photos) - len(f)
    # only rank albums whose every dial photo has a box; a partial album could
    # crown a photo only because its better sibling was not processed yet
    complete = photos.groupby("album_id").size() == f.groupby("album_id").size().reindex(
        photos.album_id.unique(), fill_value=0)
    f = f[f.album_id.isin(complete[complete].index)]
    f = score(features(f), w, args.min_det, args.sharpness, key="watch_id")
    f.sort_values(["watch_id", "rank"]).to_parquet(args.out, index=False)

    print(f"{len(f)} dial photos scored: {f.watch_id.nunique()} watches in {complete.sum()} "
          f"complete albums ({missing} dial photos without a box yet; "
          f"{(~complete).sum()} albums held back)")
    print(f"flags: {f.cut.sum()} boxes touch the border, "
          f"{(f.det_score < args.min_det).sum()} below det_score {args.min_det}")

    best = f[f["rank"] == 1].set_index("watch_id")
    second = f[f["rank"] == 2].set_index("watch_id")
    out = pd.DataFrame({
        "album_id": best.album_id, "best_path": best.path, "readability": best.readability,
        "dial_px": best.dial_px, "frontal": best.frontal, "centered": best.centered,
        "cut": best.cut, "det_score": best.det_score,
        "runner_up_path": second.path.reindex(best.index),
        "margin": best.readability - second.readability.reindex(best.index),
        "candidates": f.groupby("watch_id").size().reindex(best.index),
    }).reset_index()
    out.to_parquet(args.best_out, index=False)
    print(f"best photo per watch -> {args.best_out}; scores -> {args.out}")
    print(f"best photos that are cut at the border: {out.cut.sum()}, weak detections: "
          f"{(out.det_score < args.min_det).sum()}, single-candidate watches: "
          f"{(out.candidates == 1).sum()}")

    if args.picks:
        evaluate(f, read_picks(args.picks))
        if args.final_out:
            fin = final_table(out, f, read_answers(args.picks))
            fin.to_parquet(args.final_out, index=False)
            print(f"\nfinal best photos: {len(fin)} watches in {fin.album_id.nunique()} albums "
                  f"({fin.source.value_counts().to_dict()}) -> {args.final_out}")


if __name__ == "__main__":
    main()
