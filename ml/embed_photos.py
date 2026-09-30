#!/usr/bin/env python3
"""Embed every photo under a directory tree with a DINO-family backbone.

Usage:
    python3 embed_photos.py ../photos_big                    # DINOv2-base, ungated
    python3 embed_photos.py ../photos_big --model facebook/dinov3-vitb16-pretrain-lvd1689m
    python3 embed_photos.py ../photos_big --limit 500        # smoke test

Expects the layout the downloader produces: PHOTOS_DIR/ALBUM_ID/NNN_photoid.jpeg.
For each image, stores the CLS token concatenated with the mean of the patch
tokens (register tokens excluded), as float16. Output is chunked and the run
is resumable: already-embedded paths (per the index files) are skipped, so it
can run while the downloader is still adding albums and be re-run to top up.

Output, under --out (default embeddings/MODELNAME/):
    chunk_NNNNN.npy      float16 array, one row per image
    chunk_NNNNN.parquet  index: path, album_id, chunk, row

Load everything later with:
    import glob, numpy as np, pandas as pd
    idx = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob('OUT/chunk_*.parquet'))])
    emb = np.vstack([np.load(p) for p in sorted(glob.glob('OUT/chunk_*.npy'))])
    # rows of emb align with rows of idx; L2-normalize before cosine use.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

Image.MAX_IMAGE_PIXELS = None  # product photos, trusted sizes


def pick_device(arg):
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_photos(root):
    paths = glob.glob(os.path.join(root, "*", "*.jpeg"))
    paths += glob.glob(os.path.join(root, "*", "*.jpg"))
    return sorted(os.path.relpath(p, root) for p in paths)


def already_done(out_dir):
    done = set()
    for p in glob.glob(os.path.join(out_dir, "chunk_*.parquet")):
        done.update(pd.read_parquet(p, columns=["path"])["path"])
    return done


def flush_chunk(out_dir, seq, vecs, rows):
    stem = os.path.join(out_dir, f"chunk_{seq:05d}")
    np.save(stem + ".npy", np.vstack(vecs).astype(np.float16))
    for r, row in enumerate(rows):
        row["chunk"] = os.path.basename(stem + ".npy")
        row["row"] = r
    pd.DataFrame(rows).to_parquet(stem + ".parquet", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photos_dir", help="directory tree of ALBUM_ID/*.jpeg")
    parser.add_argument("--model", default="facebook/dinov2-base",
                        help="HF model id (default: facebook/dinov2-base; "
                             "DINOv3 ids are license-gated and need HF_TOKEN)")
    parser.add_argument("--out", default=None,
                        help="output directory (default: embeddings/MODELNAME)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--chunk", type=int, default=4096,
                        help="images per output chunk file (default: 4096)")
    parser.add_argument("--limit", type=int, default=None,
                        help="embed at most N new images (smoke tests)")
    parser.add_argument("--device", default="auto",
                        help="cuda | mps | cpu | auto (default: auto)")
    args = parser.parse_args()

    out_dir = args.out or os.path.join("embeddings", args.model.split("/")[-1])
    os.makedirs(out_dir, exist_ok=True)

    device = pick_device(args.device)
    print(f"model {args.model} on {device} "
          f"(torch {torch.__version__})", flush=True)
    try:  # the fast (torchvision) processor helps when CPU preprocessing bottlenecks
        processor = AutoImageProcessor.from_pretrained(args.model, use_fast=True)
    except Exception:
        processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    n_register = getattr(model.config, "num_register_tokens", 0)

    all_paths = list_photos(args.photos_dir)
    done = already_done(out_dir)
    todo = [p for p in all_paths if p not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(all_paths)} photos found, {len(done)} already embedded, "
          f"{len(todo)} to do", flush=True)
    if not todo:
        return

    seq = len(glob.glob(os.path.join(out_dir, "chunk_*.npy")))
    vecs, rows, failed = [], [], 0
    started, embedded = time.monotonic(), 0
    for start in range(0, len(todo), args.batch):
        batch_paths, images = [], []
        for rel in todo[start: start + args.batch]:
            try:
                img = Image.open(os.path.join(args.photos_dir, rel)).convert("RGB")
                images.append(img)
                batch_paths.append(rel)
            except Exception as e:
                failed += 1
                print(f"  unreadable, skipped: {rel} ({e})", flush=True)
        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
        cls = hidden[:, 0]
        patches = hidden[:, 1 + n_register:]
        feats = torch.cat([cls, patches.mean(dim=1)], dim=-1).cpu().numpy()

        for rel, vec in zip(batch_paths, feats):
            vecs.append(vec)
            rows.append({"path": rel, "album_id": rel.split(os.sep)[0]})
            if len(vecs) == args.chunk:
                flush_chunk(out_dir, seq, vecs, rows)
                seq, vecs, rows = seq + 1, [], []

        embedded += len(images)
        if (start // args.batch) % 20 == 0:
            rate = embedded / (time.monotonic() - started)
            print(f"  {embedded}/{len(todo)}  ({rate:.1f} img/s)", flush=True)

    if vecs:
        flush_chunk(out_dir, seq, vecs, rows)
    rate = embedded / max(time.monotonic() - started, 1e-9)
    print(f"done: {embedded} embedded at {rate:.1f} img/s, {failed} unreadable, "
          f"output in {out_dir}/", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
