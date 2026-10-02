#!/usr/bin/env python3
"""Cluster photo embeddings without labels and plot them with UMAP and t-SNE.

    python3 ml/cluster_photos.py --photos best_dial_final.csv \\
        --emb embeddings/dinov3-vitb16-pretrain-lvd1689m \\
        --out photo_clusters.csv --sizes-out plots/cluster_sizes.csv \\
        --plot plots/best_photos_umap_tsne.png

Takes the photos listed in --photos (a CSV or parquet with a best_path or path
column, e.g. best_dial_final.csv) and their embeddings from --emb. The
embeddings are L2-normalized and reduced to 50 dimensions with PCA; then:

    clusters  HDBSCAN on a 10-dimensional UMAP (min_dist 0), which finds the
              number of clusters itself and leaves outliers unclustered (-1)
    groups    the clusters merged into --groups groups by cosine similarity of
              their centroids (average linkage), numbered by size; the plot
              can only color a handful of groups legibly
    layouts   a 2-D UMAP (min_dist 0.5, spread for display) and a 2-D t-SNE

Writes one row per photo (cluster, group, both layouts) to --out, cluster
sizes per group to --sizes-out, and both layouts side by side to --plot,
colored by group with each cluster's number at its centroid. Clusters are
not labeled: whether they match brands, models or photo setups needs labels
from elsewhere. Every random step is seeded, so reruns give the same result.
"""

import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_probe import load_embeddings  # noqa: E402

# categorical slots in fixed order (validated for CVD separation on the light surface)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, NOISE = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"


def read_photos(path):
    d = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, dtype=str)
    col = "best_path" if "best_path" in d else "path"
    return d[d[col].notna()].rename(columns={col: "path"}).reset_index(drop=True), col


def plot(d, out):
    groups = sorted(g for g in d.group.unique() if g > 0)
    clusters = sorted(c for c in d.cluster.unique() if c >= 0)
    plt.rcParams.update({"font.family": ["Helvetica Neue", "DejaVu Sans"], "font.size": 10,
                         "text.color": INK, "axes.labelcolor": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=SURFACE)
    for ax, (x, y, title, size) in zip(axes, [("umap_x", "umap_y", "UMAP", 9),
                                              ("tsne_x", "tsne_y", "t-SNE", 5)]):
        ax.set_facecolor(SURFACE)
        n = d[d.group == 0]
        ax.scatter(n[x], n[y], s=size, c=NOISE, linewidths=0, label=f"unclustered ({len(n)})")
        for g in groups:
            p = d[d.group == g]
            k = p.cluster.nunique()
            ax.scatter(p[x], p[y], s=size, c=SERIES[g - 1], linewidths=0, alpha=0.85, zorder=2,
                       label=f"group {g}: {k} cluster{'s' if k > 1 else ''}, {len(p)} photos")
        for c in clusters:  # cluster numbers at their centroids, in ink
            p = d[d.cluster == c]
            ax.text(p[x].median(), p[y].median(), str(c), fontsize=7, color=INK2,
                    ha="center", va="center", zorder=3)
        for g in groups:  # group label above the group's largest cluster
            p = d[d.group == g]
            big = p[p.cluster == p.cluster.value_counts().index[0]]
            ax.text(big[x].median(), big[y].max(), f"G{g}", fontsize=10, fontweight="bold",
                    color=INK, ha="center", va="bottom", zorder=4)
        ax.set_title(f"{title} of {len(d)} photo embeddings", color=INK, fontsize=12, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], loc="lower center", ncol=3,
               frameon=False, markerscale=3, fontsize=9)
    fig.suptitle(f"HDBSCAN found {len(clusters)} clusters (numbers), merged into {len(groups)} "
                 "groups (colors) by centroid similarity. Clusters carry no brand names.",
                 x=0.01, ha="left", fontsize=11, color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photos", default="best_dial_final.csv",
                        help="CSV/parquet with a best_path or path column")
    parser.add_argument("--emb", default="embeddings/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--groups", type=int, default=8,
                        help="groups to merge the clusters into for coloring (at most 8)")
    parser.add_argument("--min-cluster", type=int, default=40, help="HDBSCAN min_cluster_size")
    parser.add_argument("--min-samples", type=int, default=10, help="HDBSCAN min_samples")
    parser.add_argument("--out", default="photo_clusters.csv")
    parser.add_argument("--sizes-out", default="plots/cluster_sizes.csv")
    parser.add_argument("--plot", default="plots/best_photos_umap_tsne.png")
    args = parser.parse_args()
    if not 1 <= args.groups <= len(SERIES):
        parser.error(f"--groups must be 1-{len(SERIES)}: more colors stop being distinguishable")

    d, path_col = read_photos(args.photos)
    idx, emb = load_embeddings(args.emb)
    pos = dict(zip(idx.path, range(len(idx))))
    missing = [p for p in d.path if p not in pos]
    if missing:
        sys.exit(f"error: {len(missing)} photos have no embedding, e.g. {missing[0]}")
    X = PCA(50, random_state=0).fit_transform(emb[[pos[p] for p in d.path]].astype(np.float32))

    u10 = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=10, metric="cosine",
                    random_state=0).fit_transform(X)
    d["cluster"] = HDBSCAN(min_cluster_size=args.min_cluster,
                           min_samples=args.min_samples).fit_predict(u10)
    d[["umap_x", "umap_y"]] = umap.UMAP(n_neighbors=30, min_dist=0.5, metric="cosine",
                                         random_state=0).fit_transform(X)
    d[["tsne_x", "tsne_y"]] = TSNE(perplexity=40, init="pca", random_state=0).fit_transform(X)

    fine = sorted(c for c in d.cluster.unique() if c >= 0)
    cent = np.vstack([X[d.cluster.to_numpy() == c].mean(axis=0) for c in fine])
    merged = AgglomerativeClustering(n_clusters=min(args.groups, len(fine)), metric="cosine",
                                     linkage="average").fit_predict(cent)
    d["group"] = d.cluster.map(dict(zip(fine, merged))).fillna(-1).astype(int)
    order = d[d.group >= 0].group.value_counts().index  # group 1 = largest
    d["group"] = d.group.map({g: i + 1 for i, g in enumerate(order)}).fillna(0).astype(int)

    keep = [c for c in ("album_id", "watch_id") if c in d] + [
        "path", "cluster", "group", "umap_x", "umap_y", "tsne_x", "tsne_y"]
    d[keep].rename(columns={"path": path_col}).to_csv(args.out, index=False)
    os.makedirs(os.path.dirname(args.sizes_out) or ".", exist_ok=True)
    (d[d.cluster >= 0].groupby(["group", "cluster"]).size().rename("photos").reset_index()
     .to_csv(args.sizes_out, index=False))
    plot(d, args.plot)

    noise = (d.cluster < 0).sum()
    print(f"{len(d)} photos: {len(fine)} clusters, {noise} unclustered ({noise / len(d):.1%})")
    for g in sorted(d.group.unique()):
        if g:
            p = d[d.group == g]
            print(f"  group {g}: {len(p):5d} photos in clusters {sorted(p.cluster.unique().tolist())}")
    print(f"-> {args.out}, {args.sizes_out}, {args.plot}")


if __name__ == "__main__":
    main()
