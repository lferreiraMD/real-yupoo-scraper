#!/usr/bin/env python3
"""Train a logistic-regression probe on embeddings + labels; score every photo.

    python3 ml/train_probe.py embeddings/dinov2-base labels.csv
    python3 ml/train_probe.py embeddings/dinov3-vitb16-pretrain-lvd1689m labels.csv

Labels come from label_app.py: "yes"/"no" train the probe, "unsure" is
ignored, and "lume" (dial visible but glowing in the dark) is excluded by
default so it neither pollutes the positives nor distorts the negative
boundary — pass --lume no to fold lume shots into the negative class.
Embeddings are L2-normalized float32 before fitting. Validation is
album-grouped cross-validation — photos of one album never straddle the
train/validation split, which would leak (bursts are near-duplicates).

Outputs <emb_dir>/probe_scores.parquet (path, album_id, score) for all
photos, and prints the triage bands: auto-accept (score >= --accept),
auto-reject (score <= --reject), and the middle band for adjudication.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict


def load_embeddings(emb_dir):
    parquets = sorted(glob.glob(os.path.join(emb_dir, "chunk_*.parquet")))
    if not parquets:
        sys.exit(f"error: no chunk_*.parquet in {emb_dir}")
    idx = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    emb = np.vstack([np.load(p) for p in sorted(glob.glob(os.path.join(emb_dir, "chunk_*.npy")))])
    emb = emb.astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return idx, emb


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("emb_dir", help="embeddings directory (chunk_* files)")
    parser.add_argument("labels_csv", help="labels.csv from label_app.py")
    parser.add_argument("--accept", type=float, default=0.9)
    parser.add_argument("--reject", type=float, default=0.1)
    parser.add_argument("--lume", choices=("exclude", "no"), default="exclude",
                        help="lume labels: 'exclude' from training (default) "
                             "or fold into the 'no' class")
    parser.add_argument("--out", default=None,
                        help="scores output (default: EMB_DIR/probe_scores.parquet)")
    args = parser.parse_args()

    idx, emb = load_embeddings(args.emb_dir)
    labels = pd.read_csv(args.labels_csv)
    n_lume = (labels.label == "lume").sum()
    if args.lume == "no":
        labels.loc[labels.label == "lume", "label"] = "no"
    elif n_lume:
        print(f"{n_lume} lume labels excluded from training (--lume no to include)")
    labels = labels[labels.label.isin(["yes", "no"])]
    merged = idx.reset_index().merge(labels, on="path")
    if merged.label.nunique() < 2:
        sys.exit("error: need both yes and no labels")
    X = emb[merged["index"].to_numpy()]
    y = (merged.label == "yes").to_numpy()
    groups = merged.album_id.to_numpy()
    n_albums = len(set(groups))
    print(f"{len(idx)} embedded photos; {len(merged)} labeled "
          f"({y.sum()} yes / {(~y).sum()} no) across {n_albums} albums")

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    n_folds = min(5, n_albums)
    if n_folds >= 2:
        proba = cross_val_predict(clf, X, y, groups=groups,
                                  cv=GroupKFold(n_splits=n_folds),
                                  method="predict_proba")[:, 1]
        auc = roc_auc_score(y, proba)
        acc = ((proba >= 0.5) == y).mean()
        print(f"album-grouped {n_folds}-fold CV:  AUC {auc:.4f}   "
              f"accuracy@0.5 {acc:.4f}")
    else:
        print("warning: <2 albums with labels — skipping cross-validation")

    clf.fit(X, y)
    scores = clf.predict_proba(emb)[:, 1]
    out = args.out or os.path.join(args.emb_dir, "probe_scores.parquet")
    pd.DataFrame({"path": idx.path, "album_id": idx.album_id,
                  "score": scores.astype(np.float32)}).to_parquet(out, index=False)

    lo, hi = args.reject, args.accept
    n = len(scores)
    print(f"triage at reject<={lo}, accept>={hi}:")
    print(f"  auto-accept: {(scores >= hi).sum():6d}  ({(scores >= hi).mean():5.1%})")
    print(f"  auto-reject: {(scores <= lo).sum():6d}  ({(scores <= lo).mean():5.1%})")
    mid = ((scores > lo) & (scores < hi)).sum()
    print(f"  adjudicate:  {mid:6d}  ({mid / n:5.1%})")
    print(f"scores -> {out}")


if __name__ == "__main__":
    main()
