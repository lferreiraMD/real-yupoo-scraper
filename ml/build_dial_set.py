#!/usr/bin/env python3
"""Gate out lume shots, keep photos the dial probe accepts, and pick the border for labeling.

    python3 ml/build_dial_set.py \\
        --dial embeddings/*/svm_scores.parquet --lume embeddings/*/lume_svm_scores.parquet \\
        --out dial_set_svm.parquet \\
        --border 300 --border-out border_paths.txt \\
        --exclude labels.csv audit_accepts.csv holdout_paths.txt

Scores are probe outputs from train_probe.py (SVM margins by default, so the
thresholds default to 0). With several files per role the scores are averaged,
which makes an ensemble of embedding models. A photo is in the dial set when
its lume score is below --lume-threshold and its dial score is at or above
--dial-threshold. Photos with a human label (--labels) take that label instead:
"yes" is a dial photo, "no" and "lume" are not. --photos-out writes every
photo with its scores and final is_dial flag; --albums-out writes dial-photo
counts per album.

--border N writes the N unlabeled photos the pipeline is least sure about, as
a queue for label_app.py --paths: photos closest to the dial boundary (among
those the gate lets through) and photos closest to the lume boundary, split by
--lume-share. At most one photo per album per list, since burst shots from
one album are near-duplicates and teach the probe the same thing twice.
--exclude takes label logs (CSV with a path column) or path lists; those
photos never enter the border queue.
"""

import argparse
import os
import random

import pandas as pd


def mean_scores(files, name):
    out = None
    cols = [f"{name}_{i}" for i in range(len(files))]
    for f, col in zip(files, cols):
        d = pd.read_parquet(f)[["path", "album_id", "score"]].rename(columns={"score": col})
        out = d if out is None else out.merge(d.drop(columns="album_id"), on="path")
    out[name] = out[cols].mean(axis=1)
    out[f"{name}_split"] = (out[cols] >= 0).any(axis=1) & (out[cols] < 0).any(axis=1)
    return out[["path", "album_id", name, f"{name}_split"]]


def read_paths(p):
    if p.endswith(".csv"):
        return set(pd.read_csv(p).path)
    with open(p) as f:
        return {line.strip() for line in f if line.strip()}


def closest(df, col, threshold, n):
    d = df.assign(dist=(df[col] - threshold).abs()).sort_values("dist")
    return d.drop_duplicates("album_id").head(n)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dial", nargs="+", required=True, help="dial score parquet(s)")
    parser.add_argument("--lume", nargs="+", required=True, help="lume score parquet(s)")
    parser.add_argument("--dial-threshold", type=float, default=0.0)
    parser.add_argument("--lume-threshold", type=float, default=0.0)
    parser.add_argument("--out", default="dial_set_svm.parquet")
    parser.add_argument("--border", type=int, default=0, help="border queue size")
    parser.add_argument("--border-out", default="border_paths.txt")
    parser.add_argument("--lume-share", type=float, default=1 / 3,
                        help="fraction of the border queue spent on the lume boundary")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="label logs or path lists to keep out of the border queue")
    parser.add_argument("--labels", nargs="*", default=[],
                        help="human label logs; a human label overrides the model's call")
    parser.add_argument("--photos-out", default=None,
                        help="also write every photo with its scores and is_dial flag")
    parser.add_argument("--albums-out", default=None,
                        help="also write dial-photo counts per album")
    args = parser.parse_args()

    d = mean_scores(args.dial, "dial").merge(
        mean_scores(args.lume, "lume").drop(columns="album_id"), on="path")
    d["album_id"] = d.album_id.astype(str)
    d["is_lume_model"] = d.lume >= args.lume_threshold
    d["is_dial_model"] = ~d.is_lume_model & (d.dial >= args.dial_threshold)
    d["human_label"] = None
    if args.labels:
        logs = pd.concat([pd.read_csv(p) for p in args.labels], ignore_index=True)
        logs = logs.sort_values("ts", kind="stable").drop_duplicates("path", keep="last")
        logs = logs[logs.label.isin(["yes", "no", "lume"])].set_index("path").label
        d["human_label"] = d.path.map(logs)
    human = d.human_label.notna()
    d["is_lume"] = d.is_lume_model.where(~human, d.human_label == "lume")
    d["is_dial"] = d.is_dial_model.where(~human, d.human_label == "yes")
    d["source"] = human.map({True: "human", False: "model"})
    keep = d.is_dial
    out = d[keep][["path", "album_id", "dial", "lume"]].rename(
        columns={"dial": "score", "lume": "lume_score"}).sort_values("path")
    out.to_parquet(args.out, index=False)

    albums = d.album_id.nunique()
    print(f"{len(d)} photos in {albums} albums: {d.is_lume.sum()} lume shots, "
          f"{keep.sum()} in the dial set")
    if args.labels:
        print(f"{human.sum()} photos carry a human label; it overrode the model on "
              f"{(human & (d.is_dial != d.is_dial_model)).sum()} dial calls")
    empty = albums - out.album_id.nunique()
    print(f"albums without a dial photo: {empty}")
    if args.photos_out:
        d.rename(columns={"dial": "dial_score", "lume": "lume_score"})[
            ["path", "album_id", "dial_score", "lume_score", "is_lume", "is_dial",
             "is_dial_model", "human_label", "source"]
        ].sort_values("path").to_parquet(args.photos_out, index=False)
        print(f"all photos -> {args.photos_out}")
    if args.albums_out:
        a = d.assign(nonlume_dial=d.dial.where(~d.is_lume)).groupby("album_id").agg(
            photos=("path", "size"), dial_photos=("is_dial", "sum"),
            lume_photos=("is_lume", "sum"), best_dial_score=("nonlume_dial", "max"))
        a.reset_index().to_parquet(args.albums_out, index=False)
        print(f"per-album counts -> {args.albums_out}")
    print(f"models disagree on the dial call for {(~d.is_lume_model & d.dial_split).sum()} photos, "
          f"on the lume call for {d.lume_split.sum()}")
    print(f"dial set -> {args.out}")

    if args.border:
        excluded = set().union(*(read_paths(p) for p in args.exclude)) if args.exclude else set()
        pool = d[~d.path.isin(excluded)]
        n_lume = round(args.border * args.lume_share)
        on_lume = closest(pool, "lume", args.lume_threshold, n_lume)
        pool = pool[~pool.path.isin(on_lume.path)]
        on_dial = closest(pool[pool.lume < args.lume_threshold], "dial",
                          args.dial_threshold, args.border - len(on_lume))
        queue = list(on_dial.path) + list(on_lume.path)
        random.Random(42).shuffle(queue)
        with open(args.border_out, "w") as f:
            f.write("\n".join(queue) + "\n")
        print(f"border queue: {len(on_dial)} near the dial boundary (|score| <= "
              f"{on_dial.dist.max():.3f}), {len(on_lume)} near the lume boundary "
              f"(|score| <= {on_lume.dist.max():.3f}) -> {args.border_out}")


if __name__ == "__main__":
    main()
