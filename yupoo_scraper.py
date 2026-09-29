#!/usr/bin/env python3
"""Download every photo and video from a public Yupoo album, in original quality.

Usage:
    python3 yupoo_scraper.py "https://store.x.yupoo.com/albums/123456789?uid=1"
    python3 yupoo_scraper.py "https://store.x.yupoo.com/albums/123456789" -o ./watches

Yupoo serves full-resolution originals but rejects any request that does not
carry a Referer header from the album's own domain. This script supplies that
header and nothing more exotic. Standard library only; Python 3.8+.
"""

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)
ALBUM_URL_RE = re.compile(r"https?://([\w-]+)\.x\.yupoo\.com/albums/(\d+)")
ITEM_SPLIT = '<div class="showalbum__children image__main"'
MAX_PAGES = 200


def http_open(url, referer, timeout=60):
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Referer": referer}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_text(url, referer):
    with http_open(url, referer) as r:
        return r.read().decode("utf-8", errors="replace")


def fetch_file(url, referer, dest, retries=2):
    """Download url to dest, returning the byte count. Raises on final failure."""
    for attempt in range(retries + 1):
        try:
            with http_open(url, referer) as r, open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            return os.path.getsize(dest)
        except Exception:
            if os.path.exists(dest):
                os.remove(dest)
            if attempt == retries:
                raise
            time.sleep(1 + attempt)


def attr(fragment, name):
    m = re.search(name + r'="([^"]*)"', fragment)
    return html.unescape(m.group(1)) if m else None


def absolute(url):
    return "https:" + url if url.startswith("//") else url


def sanitize(name, fallback):
    name = re.sub(r"[^\w.-]+", "_", name).strip("._")
    return name or fallback


def parse_page(page_html):
    """Return (title, items). Each item is a dict describing one photo/video."""
    m = re.search(r"showalbumheader__gallerytitle[^>]*>([^<]*)<", page_html)
    title = html.unescape(m.group(1)).strip() if m else ""

    items = []
    for fragment in page_html.split(ITEM_SPLIT)[1:]:
        end = fragment.find("</time>")
        fragment = fragment[: end + 7] if end != -1 else fragment[:4000]
        item = {
            "id": attr(fragment, "data-id"),
            "alt": attr(fragment, "<img alt"),
            "type": "video" if 'data-type="video"' in fragment else "photo",
        }
        if item["type"] == "video":
            item["path"] = attr(fragment, "data-path")
            formats = attr(fragment, "data-videoformats") or ""
            item["formats"] = [f for f in formats.split(",") if f]
        else:
            src = attr(fragment, "data-origin-src")
            if not src:
                continue
            item["src"] = absolute(src)
        items.append(item)
    return title, items


def collect_album(album_url):
    """Fetch every page of the album; return (title, items) with duplicates removed."""
    m = ALBUM_URL_RE.match(album_url)
    if not m:
        sys.exit(f"error: not a Yupoo album URL: {album_url}\n"
                 "expected form: https://STORE.x.yupoo.com/albums/ALBUM_ID")
    store, album_id = m.groups()
    referer = f"https://{store}.x.yupoo.com/albums/{album_id}"

    title, items, seen = "", [], set()
    for page in range(1, MAX_PAGES + 1):
        page_url = f"{referer}?uid=1&page={page}"
        try:
            page_html = fetch_text(page_url, referer)
        except urllib.error.HTTPError as e:
            if page == 1:
                sys.exit(f"error: could not fetch album page ({e})")
            break
        page_title, page_items = parse_page(page_html)
        title = title or page_title
        new = [i for i in page_items if i["id"] not in seen]
        if not new:
            if page == 1 and "password" in page_html.lower():
                sys.exit("error: no items found — this album appears to be "
                         "password-protected, which this script does not handle.")
            break
        seen.update(i["id"] for i in new)
        items.extend(new)
    return store, album_id, title, referer, items


def download_item(index, item, referer, out_dir):
    """Download one item; returns (filename, bytes) or raises."""
    if item["type"] == "photo":
        ext = item["src"].rsplit(".", 1)[-1].lower()
        photo_id = item["src"].split("/")[-2]
        dest = os.path.join(out_dir, f"{index:03d}_{photo_id}.{ext}")
        size = fetch_file(item["src"], referer, dest)
        return dest, size

    # Video: try the transcoded formats first, then the original upload.
    base, _, orig_ext = item["path"].rpartition(".")
    candidates = [
        (f"https://uvd.yupoo.com{base}_{fmt}.mp4", "mp4") for fmt in item["formats"]
    ]
    candidates.append((f"https://uvd.yupoo.com{item['path']}", orig_ext.lower() or "mp4"))
    last_error = None
    for url, ext in candidates:
        dest = os.path.join(out_dir, f"{index:03d}_{os.path.basename(base)}.{ext}")
        try:
            size = fetch_file(url, referer, dest, retries=0)
            return dest, size
        except Exception as e:
            last_error = e
    raise last_error


def main():
    parser = argparse.ArgumentParser(
        description="Download all photos and videos from a public Yupoo album."
    )
    parser.add_argument("album_url", help="album URL, e.g. https://store.x.yupoo.com/albums/123456789")
    parser.add_argument("-o", "--output", default=None,
                        help="output directory (default: ./STORE_TITLE_ALBUMID)")
    args = parser.parse_args()

    store, album_id, title, referer, items = collect_album(args.album_url.strip())
    if not items:
        sys.exit("error: no photos or videos found in this album.")

    out_dir = args.output or f"{store}_{sanitize(title, 'album')}_{album_id}"
    os.makedirs(out_dir, exist_ok=True)
    print(f'Album "{title or album_id}" by {store}: {len(items)} items -> {out_dir}/')

    manifest = {"store": store, "album_id": album_id, "title": title,
                "url": referer, "items": []}
    failures = 0
    for index, item in enumerate(items, 1):
        try:
            dest, size = download_item(index, item, referer, out_dir)
            print(f"  [{index:03d}/{len(items)}] {item['type']:5s} "
                  f"{os.path.basename(dest)}  ({size / 1e6:.2f} MB)")
            manifest["items"].append({**item, "file": os.path.basename(dest), "bytes": size})
        except Exception as e:
            failures += 1
            print(f"  [{index:03d}/{len(items)}] {item['type']:5s} FAILED: {e}")
            manifest["items"].append({**item, "error": str(e)})

    with open(os.path.join(out_dir, "album.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Done: {len(items) - failures}/{len(items)} items downloaded.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
