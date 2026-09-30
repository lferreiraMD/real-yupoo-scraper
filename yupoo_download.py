#!/usr/bin/env python3
"""Download the photos of every album in a manifest.parquet, browser-shaped and resumable.

Usage:
    python3 yupoo_download.py manifest.parquet
    python3 yupoo_download.py manifest.parquet --size big -o photos_big --max-albums 50
    python3 yupoo_download.py manifest.parquet --max-hours 8

Traffic is shaped like a person browsing: each album's photos are fetched
with a small concurrent burst (as a browser would), followed by a jittered
pause before the next album. A 429/503 anywhere triggers a long cooldown.

State lives on disk, not in the manifest: an album is done when
OUTDIR/ALBUM_ID/album.json says so. Re-running skips finished albums and
refetches only the missing files of half-finished ones, so the job can be
killed and resumed at any point. Videos are skipped.
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from yupoo_scraper import fetch_file, fetch_text, parse_page

SIZES = ("small", "medium", "big", "origin")
COOLDOWN_SECONDS = 180


def album_done(album_dir):
    try:
        with open(os.path.join(album_dir, "album.json")) as f:
            return json.load(f).get("complete", False)
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def sized_url(origin_src, size):
    return origin_src.rsplit("/", 1)[0] + f"/{size}.jpeg"


def grab_photo(item, dest, referer, size):
    """Fetch one photo at the requested size, falling back to the original file.

    Returns (ok, bytes, rate_limited)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True, os.path.getsize(dest), False
    for url in (sized_url(item["src"], size), item["src"]):
        try:
            return True, fetch_file(url, referer, dest, retries=1), False
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                return False, 0, True
        except Exception:
            pass
    return False, 0, False


def process_album(row, out_dir, size, concurrency):
    """Download one album. Returns (n_ok, n_failed, rate_limited)."""
    referer = row.url
    album_dir = os.path.join(out_dir, str(row.album_id))
    os.makedirs(album_dir, exist_ok=True)

    for attempt in range(3):  # album pages throw the occasional transient 500
        try:
            page_html = fetch_text(f"{referer}?uid=1", referer)
            break
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:
                raise  # 404 and friends are permanent — a delisted album
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    title, items = parse_page(page_html)
    photos = [i for i in items if i["type"] == "photo"]

    jobs = []
    for idx, item in enumerate(photos, 1):
        photo_id = item["src"].split("/")[-2]
        jobs.append((item, os.path.join(album_dir, f"{idx:03d}_{photo_id}.jpeg")))

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(
            lambda j: grab_photo(j[0], j[1], referer, size), jobs
        ))

    n_ok = sum(1 for ok, _, _ in results if ok)
    rate_limited = any(rl for _, _, rl in results)
    manifest = {
        "album_id": int(row.album_id),
        "title": title,
        "url": referer,
        "size": size,
        "badge_count": None if pd.isna(row.photo_count) else int(row.photo_count),
        "items_on_page": len(items),
        "photos_expected": len(photos),
        "photos_ok": n_ok,
        # videos-only albums (photos empty, items not) complete trivially;
        # a page that parsed zero items of any kind stays incomplete
        "complete": n_ok == len(photos) and len(items) > 0,
        "photos": [
            {"file": os.path.basename(dest), "src": item["src"],
             "alt": item.get("alt"), "ok": ok, "bytes": nbytes}
            for (item, dest), (ok, nbytes, _) in zip(jobs, results)
        ],
    }
    with open(os.path.join(album_dir, "album.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return n_ok, len(photos) - n_ok, rate_limited


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", help="manifest.parquet from yupoo_manifest.py")
    parser.add_argument("--size", default="big", choices=SIZES,
                        help="image size to fetch (default: big)")
    parser.add_argument("-o", "--output", default=None,
                        help="output directory (default: photos_SIZE)")
    parser.add_argument("--max-albums", type=int, default=None,
                        help="stop after downloading this many albums this run")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="stop starting new albums after this many hours")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="parallel photo fetches within one album (default: 4)")
    parser.add_argument("--pause", type=float, nargs=2, default=(1.0, 3.0),
                        metavar=("MIN", "MAX"), help="jittered pause between albums (default: 1 3)")
    args = parser.parse_args()

    df = pd.read_parquet(args.manifest)
    out_dir = args.output or f"photos_{args.size}"
    os.makedirs(out_dir, exist_ok=True)

    started = time.monotonic()
    processed = skipped = failures = 0
    for row in df.itertuples():
        if album_done(os.path.join(out_dir, str(row.album_id))):
            skipped += 1
            continue
        if args.max_albums is not None and processed >= args.max_albums:
            print(f"reached --max-albums {args.max_albums}, stopping.")
            break
        if args.max_hours is not None and time.monotonic() - started > args.max_hours * 3600:
            print(f"reached --max-hours {args.max_hours}, stopping.")
            break

        for attempt in (1, 2):
            try:
                n_ok, n_bad, rate_limited = process_album(
                    row, out_dir, args.size, args.concurrency)
            except Exception as e:
                n_ok, n_bad, rate_limited = 0, -1, False
                print(f"  {row.album_id} ({row.title}): album page failed: {e}",
                      flush=True)
                break
            print(f"  {row.album_id} ({row.title}): {n_ok} photos"
                  + (f", {n_bad} FAILED" if n_bad else ""), flush=True)
            if not rate_limited:
                break
            print(f"  rate limited — cooling down {COOLDOWN_SECONDS}s "
                  f"(attempt {attempt}/2)", flush=True)
            time.sleep(COOLDOWN_SECONDS)
        processed += 1
        failures += 1 if n_bad else 0
        time.sleep(random.uniform(*args.pause))

    total_done = skipped + processed - failures
    print(f"\nRun summary: {processed} albums processed, {skipped} already done, "
          f"{failures} with failures. {total_done}/{len(df)} albums complete.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
