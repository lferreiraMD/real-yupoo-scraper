#!/usr/bin/env python3
"""Locate the watch dial in each photo with OWLv2 and measure it.

    python3 ml/detect_dials.py photos_big --photos photos_dial.parquet
    python3 ml/detect_dials.py photos_big --paths some_paths.txt --out boxes_test
    python3 ml/detect_dials.py photos_big --restat --out dial_boxes   # redo pixel stats only

Runs the open-vocabulary detector google/owlv2-base-patch16-ensemble with the
text query "a clock face" and keeps the highest-scoring box per photo. That
query boxes the dial itself (square boxes on frontal round dials); queries such
as "a wristwatch" or "the face of a wristwatch" box the whole watch with its
bracelet. Input stays at the model's native 960 px: lower resolutions with
interpolated position embeddings move the boxes. For the box it records
geometry and pixel statistics that rank_dials.py turns into a readability score:

    box_x0..box_y1   box corners as fractions of the image width/height
    det_score        detector confidence for the box
    width, height    image size in pixels
    sharpness        variance of the Laplacian inside the box, at native resolution
    sharpness_256    the same on the box resized to 256x256: detail available at a
                     fixed viewing size, so a close dial is not penalized for
                     spreading over more pixels
    glare            fraction of near-white pixels (all channels >= 245) inside the box
    brightness       mean gray level inside the box (0-255)
    contrast         standard deviation of the gray level inside the box

--photos takes photos_dial.parquet (build_dial_set.py --photos-out) and keeps
the rows flagged is_dial; --paths takes a file with one photo path per line.
Output goes to --out as part_NNNNN.parquet files; the run is resumable, so
paths already in the output are skipped. --restat recomputes the pixel
statistics of existing part files from their stored boxes (CPU only).
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
from transformers import Owlv2ForObjectDetection, Owlv2Processor

MODEL = "google/owlv2-base-patch16-ensemble"
QUERIES = ["a clock face"]
GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_detector(model_name=MODEL, device=None):
    device = device or pick_device()
    processor = Owlv2Processor.from_pretrained(model_name, use_fast=True)
    model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device).eval()
    text = processor.tokenizer(QUERIES, padding="max_length", return_tensors="pt").to(device)
    return processor, model, text, device


def detect(detector, pixels, sizes):
    """Best box per image as (x0, y0, x1, y1) fractions of the unpadded image, plus its score."""
    processor, model, text, device = detector
    pixels = pixels.to(device)
    with torch.inference_mode():
        out = model(pixel_values=pixels,
                    input_ids=text.input_ids.repeat(len(pixels), 1),
                    attention_mask=text.attention_mask.repeat(len(pixels), 1))
    scores = torch.sigmoid(out.logits).amax(dim=-1)
    best = scores.argmax(dim=-1)
    results = []
    for i, (w, h) in enumerate(sizes):
        side = max(h, w)  # OWLv2 pads to a square at the bottom/right
        cx, cy, bw, bh = out.pred_boxes[i, best[i]].tolist()
        box = ((cx - bw / 2) * side / w, (cy - bh / 2) * side / h,
               (cx + bw / 2) * side / w, (cy + bh / 2) * side / h)
        results.append(([min(max(v, 0.0), 1.0) for v in box], float(scores[i, best[i]])))
    return results


def laplacian_var(gray):
    lap = (gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1]
           - 4 * gray[1:-1, 1:-1])
    return float(lap.var())


def box_stats(arr, box):
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = box
    xa, xb = max(int(np.floor(x0 * w)), 0), min(int(np.ceil(x1 * w)), w)
    ya, yb = max(int(np.floor(y0 * h)), 0), min(int(np.ceil(y1 * h)), h)
    crop = arr[ya:yb, xa:xb]
    nan = {k: np.nan for k in ("sharpness", "sharpness_256", "glare", "brightness", "contrast")}
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        return nan
    gray = crop.astype(np.float32) @ GRAY
    fixed = np.asarray(Image.fromarray(crop).convert("L").resize((256, 256), Image.BICUBIC),
                       dtype=np.float32)
    return {"sharpness": laplacian_var(gray), "sharpness_256": laplacian_var(fixed),
            "glare": float((crop >= 245).all(axis=2).mean()),
            "brightness": float(gray.mean()), "contrast": float(gray.std())}


def done_paths(out_dir):
    done = set()
    for p in glob.glob(os.path.join(out_dir, "part_*.parquet")):
        done.update(pd.read_parquet(p, columns=["path"]).path)
    return done


class Photos(torch.utils.data.Dataset):
    def __init__(self, root, paths, processor):
        self.root, self.paths, self.processor = root, paths, processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rel = self.paths[i]
        try:
            img = Image.open(os.path.join(self.root, rel)).convert("RGB")
        except Exception:
            return rel, None, None
        pixels = self.processor.image_processor(images=img, return_tensors="pt").pixel_values[0]
        return rel, pixels, np.asarray(img)


def collate(items):
    return items


def restat(root, out_dir):
    for part in sorted(glob.glob(os.path.join(out_dir, "part_*.parquet"))):
        d = pd.read_parquet(part)
        stats = []
        for r in d.itertuples():
            arr = np.asarray(Image.open(os.path.join(root, r.path)).convert("RGB"))
            stats.append(box_stats(arr, (r.box_x0, r.box_y0, r.box_x1, r.box_y1)))
        s = pd.DataFrame(stats, index=d.index)
        d = d.drop(columns=[c for c in s.columns if c in d.columns]).join(s)
        d.to_parquet(part, index=False)
        print(f"  restat {part}: {len(d)} photos", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photos_dir", help="directory tree of ALBUM_ID/*.jpeg")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--photos", help="photos_dial.parquet; rows with is_dial are processed")
    src.add_argument("--paths", help="file of photo paths, one per line")
    src.add_argument("--restat", action="store_true",
                     help="recompute pixel statistics of the part files in --out")
    parser.add_argument("--out", default="dial_boxes", help="output directory")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch", type=int, default=1,
                        help="photos per forward pass (OWLv2 attends over 3,600 patches; "
                             "1 fits a 4 GB GPU)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--part", type=int, default=1024, help="photos per output part file")
    parser.add_argument("--limit", type=int, default=None, help="process at most N photos")
    args = parser.parse_args()

    if args.restat:
        restat(args.photos_dir, args.out)
        return
    if args.photos:
        p = pd.read_parquet(args.photos)
        paths = sorted(p[p.is_dial].path)
    elif args.paths:
        with open(args.paths) as f:
            paths = [line.strip() for line in f if line.strip()]
    else:
        parser.error("one of --photos, --paths or --restat is required")
    os.makedirs(args.out, exist_ok=True)
    done = done_paths(args.out)
    todo = [p for p in paths if p not in done][: args.limit]
    print(f"{len(paths)} photos listed, {len(done)} already done, {len(todo)} to do", flush=True)
    if not todo:
        return

    detector = load_detector(args.model)
    print(f"model {args.model} on {detector[3]}; queries {QUERIES}", flush=True)
    loader = torch.utils.data.DataLoader(Photos(args.photos_dir, todo, detector[0]),
                                         batch_size=args.batch, num_workers=args.workers,
                                         collate_fn=collate)
    seq = len(glob.glob(os.path.join(args.out, "part_*.parquet")))
    rows, failed, started, n = [], 0, time.monotonic(), 0
    for batch in loader:
        ok = [b for b in batch if b[1] is not None]
        failed += len(batch) - len(ok)
        if not ok:
            continue
        found = detect(detector, torch.stack([b[1] for b in ok]),
                       [(b[2].shape[1], b[2].shape[0]) for b in ok])
        for (rel, _, arr), (box, det_score) in zip(ok, found):
            rows.append({"path": rel, "album_id": rel.split("/")[0],
                         "width": arr.shape[1], "height": arr.shape[0],
                         "box_x0": box[0], "box_y0": box[1], "box_x1": box[2], "box_y1": box[3],
                         "det_score": det_score, **box_stats(arr, box)})
        n += len(ok)
        if len(rows) >= args.part:
            pd.DataFrame(rows).to_parquet(os.path.join(args.out, f"part_{seq:05d}.parquet"), index=False)
            seq, rows = seq + 1, []
            rate = n / (time.monotonic() - started)
            print(f"  {n}/{len(todo)}  ({rate:.1f} img/s)", flush=True)
    if rows:
        pd.DataFrame(rows).to_parquet(os.path.join(args.out, f"part_{seq:05d}.parquet"), index=False)
    rate = n / max(time.monotonic() - started, 1e-9)
    print(f"done: {n} photos at {rate:.1f} img/s, {failed} unreadable, output in {args.out}/", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
