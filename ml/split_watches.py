#!/usr/bin/env python3
"""Split albums that hold more than one watch, and build the pick queue.

    python3 ml/split_watches.py --photos photos_dial.parquet --manifest manifest.parquet \\
        --emb embeddings/dinov3-vitb16-pretrain-lvd1689m \\
        --out watches.parquet --albums-out album_watches.parquet \\
        --pick-queue 150 --pick-out pick_paths.txt --keep pick_labels.csv

Albums come in sizes of about 18, 35 and 50 photos: some albums hold two or
three sets shot one after another. Within each album the dial photos are
grouped by appearance (average-linkage clustering of the embeddings, cut at
cosine similarity --cutoff). An album is split only when it is confident:
at least two groups of two or more dial photos (a lone odd shot joins its
nearest group) AND the album is multi-set sized (--multi-min photos). A
multi-set sized album that the cutoff leaves whole is still split in two
when its two most different groups fall below --multi-cutoff. Split groups
get watch IDs ALBUMa, ALBUMb, ... in shooting order.

Categories in --albums-out:
    split       confident split into watches
    lookalike   multi-set sized but the sets look alike: kept as one screen
                where several best photos can be selected
    single      one watch
and a borderline flag for albums near the cutoff (between-group similarity
within --band of it), or with a split that the album size does not support.

--pick-queue writes a label_app.py --pick queue, one screen per watch, as
"path<TAB>screen" lines with each screen's photos shuffled: all borderline
albums first, then N random albums (plus any album in --keep logs), then the
remaining look-alike albums.
"""

import argparse
import glob
import os
import random
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_probe import load_embeddings  # noqa: E402


def photo_number(path):
    return int(path.split("/")[1].split("_")[0])


def group_album(E, cutoff):
    """Cluster rows of E; merge singletons into their nearest group. Returns labels, between-sim."""
    n = len(E)
    if n < 2:
        return np.zeros(n, int), np.nan
    S = E @ E.T
    two = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average").fit_predict(E)
    between = float(S[np.ix_(two == 0, two == 1)].mean())
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - cutoff,
                                  metric="cosine", linkage="average").fit_predict(E)
    sizes = np.bincount(lab)
    big = np.flatnonzero(sizes >= 2)
    if len(big) == 0:
        return np.zeros(n, int), between
    for i in np.flatnonzero(sizes[lab] < 2):  # lone photo: join the most similar group
        lab[i] = big[np.argmax([S[i, lab == g].mean() for g in big])]
    _, lab = np.unique(lab, return_inverse=True)
    return lab, between


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photos", default="photos_dial.parquet")
    parser.add_argument("--manifest", default="manifest.parquet")
    parser.add_argument("--emb", default="embeddings/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--cutoff", type=float, default=0.80,
                        help="cosine similarity below which groups are different watches")
    parser.add_argument("--multi-cutoff", type=float, default=0.825,
                        help="multi-set sized albums whose two most different groups are less "
                             "similar than this are split in two even if --cutoff does not split "
                             "them (borderline review: 70 of 78 such albums at 0.80-0.825 held "
                             "two watches, and the 2-way split separated the picks in 66 of 70)")
    parser.add_argument("--band", type=float, default=0.025,
                        help="half-width of the borderline zone around --cutoff")
    parser.add_argument("--multi-min", type=int, default=25,
                        help="albums with at least this many photos are multi-set sized")
    parser.add_argument("--out", default="watches.parquet")
    parser.add_argument("--albums-out", default="album_watches.parquet")
    parser.add_argument("--pick-queue", type=int, default=0, help="random albums in the queue")
    parser.add_argument("--pick-out", default="pick_paths.txt")
    parser.add_argument("--keep", nargs="*", default=[],
                        help="pick logs whose albums must stay in the queue")
    parser.add_argument("--no-lookalikes", action="store_true",
                        help="leave the remaining look-alike albums out of the queue")
    args = parser.parse_args()

    p = pd.read_parquet(args.photos)
    dial = p[p.is_dial][["path", "album_id"]].copy()
    dial["album_id"] = dial.album_id.astype(str)
    m = pd.read_parquet(args.manifest)
    m["album_id"] = m.album_id.astype(str)
    count = m.set_index("album_id").photo_count
    idx, emb = load_embeddings(args.emb)
    pos = dict(zip(idx.path, range(len(idx))))

    rows, albums = [], []
    for a, g in dial.groupby("album_id"):
        paths = sorted(g.path, key=photo_number)
        E = emb[[pos[x] for x in paths]]
        lab, between = group_album(E, args.cutoff)
        k = lab.max() + 1
        multi = count.get(a, len(paths)) >= args.multi_min
        if multi and k == 1 and between < args.multi_cutoff and len(paths) >= 4:
            lab = AgglomerativeClustering(n_clusters=2, metric="cosine",
                                          linkage="average").fit_predict(E)
            k = 2 if np.bincount(lab).min() >= 2 else 1
            if k == 1:
                lab = np.zeros(len(paths), int)
        if k >= 2 and multi:
            category = "split"
        elif multi:
            category, lab, k = "lookalike", np.zeros_like(lab), 1
        else:
            category, k = "single", 1
        near = not np.isnan(between) and abs(between - args.cutoff) < args.band
        unsupported = (lab.max() >= 1 and not multi) or (category != "split" and between < args.cutoff
                                                         and multi)
        borderline = bool((near and multi) or unsupported)
        if category != "split":
            lab = np.zeros(len(paths), int)
        # letters in shooting order: the group with the earliest photo is "a"
        first = {g: min(photo_number(x) for x, l in zip(paths, lab) if l == g) for g in set(lab)}
        order = {g: i for i, g in enumerate(sorted(first, key=first.get))}
        for x, l in zip(paths, lab):
            wid = a if k == 1 else a + "abcdefghij"[order[l]]
            rows.append({"path": x, "album_id": a, "watch_id": wid})
        albums.append({"album_id": a, "photo_count": int(count.get(a, -1)), "dial_photos": len(paths),
                       "watches": int(k), "between": between, "category": category,
                       "borderline": borderline})
    w = pd.DataFrame(rows)
    al = pd.DataFrame(albums)
    w.to_parquet(args.out, index=False)
    al.to_parquet(args.albums_out, index=False)
    print(f"{len(al)} albums: {al.category.value_counts().to_dict()}; "
          f"{al.borderline.sum()} borderline")
    print(f"{w.watch_id.nunique()} watches -> {args.out}; per-album summary -> {args.albums_out}")

    if args.pick_queue:
        keep = set()
        for log in args.keep:
            if os.path.exists(log):
                prev = pd.read_csv(log, dtype=str)
                col = prev.path if "screen" not in prev else prev.screen.str.rstrip("abcdefghij")
                keep |= set(col.dropna().str.split("/").str[0])
        border = sorted(al[al.borderline].album_id)
        rest = sorted(set(al.album_id) - set(border) - keep)
        rand = sorted(keep & set(al.album_id)) + random.Random(7).sample(
            rest, max(0, args.pick_queue - len(keep)))
        look = [] if args.no_lookalikes else sorted(
            set(al[al.category == "lookalike"].album_id) - set(border) - set(rand))
        lines = []
        for a in border + rand + look:
            for wid, g in w[w.album_id == a].groupby("watch_id", sort=True):
                cand = sorted(g.path)
                random.Random(wid).shuffle(cand)  # no position hint from any ranking
                lines += [f"{x}\t{wid}" for x in cand]
        with open(args.pick_out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        screens = len({ln.split("\t")[1] for ln in lines})
        print(f"pick queue: {len(border)} borderline + {len(rand)} random + {len(look)} look-alike "
              f"albums = {screens} screens, {len(lines)} photos -> {args.pick_out}")


if __name__ == "__main__":
    main()
