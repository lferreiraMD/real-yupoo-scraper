#!/usr/bin/env python3
"""Build or incrementally update a Parquet manifest of every album in a Yupoo store.

Usage:
    python3 yupoo_manifest.py "https://STORE.x.yupoo.com"
    python3 yupoo_manifest.py "https://STORE.x.yupoo.com" -m manifest.parquet
    python3 yupoo_manifest.py "https://STORE.x.yupoo.com" --full

The store index is sorted newest-first, so an incremental run walks pages from
the top and stops at the first page containing no unseen album id — a daily
update with a handful of new albums costs one or two index requests. Rows are
append-only, keyed on album_id (Yupoo's own stable identifier); delisted
albums keep their rows, distinguishable by a stale last_seen. --full recrawls
every page, refreshing last_seen and photo_count on all rows.

Columns: album_id, url, title, sku, date_label, guessed_date, photo_count,
cover_url, first_seen, last_seen.
"""

import argparse
import datetime as dt
import html
import random
import re
import sys
import time
import urllib.request

import pandas as pd

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)
STORE_RE = re.compile(r"https?://([\w-]+)\.x\.yupoo\.com")
# SKU / month.day, with an optional non-digit suffix some listings carry
# ("9.30ex", "9.28加单", "9.27补发")
TITLE_RE = re.compile(r"^(.*?)/(\d{1,2}\.\d{1,2})(?:\D.*)?$")


def fetch_text(url, referer):
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Referer": referer}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def split_title(title):
    """'AE171547/9.27' -> ('AE171547', '9.27'); unparseable titles keep sku=title."""
    m = TITLE_RE.match(title.strip())
    return (m.group(1), m.group(2)) if m else (title.strip(), None)


def guess_date(date_label, seen):
    """Assign a year to 'M.DD': the most recent such date not after first_seen.

    'Today' is judged in the store's timezone (Yupoo is Chinese), which can be
    a day ahead of the crawler's clock."""
    if not date_label:
        return None
    store_today = seen.tz_convert("Asia/Shanghai")
    try:
        month, day = (int(x) for x in date_label.split("."))
        year = (store_today.year
                if (month, day) <= (store_today.month, store_today.day)
                else store_today.year - 1)
        return dt.date(year, month, day).isoformat()
    except ValueError:
        return None


def parse_index_page(page_html, store):
    cards = []
    for chunk in page_html.split('class="album__main"')[1:]:
        chunk = chunk[:2500]
        aid = re.search(r'href="/albums/(\d+)', chunk)
        if not aid:
            continue
        title = re.search(r'title="([^"]*)"', chunk)
        cover = re.search(r'src="([^"]+)"', chunk)
        count = re.search(r'album__photonumber">(\d+)<', chunk)
        cards.append({
            "album_id": int(aid.group(1)),
            "url": f"https://{store}.x.yupoo.com/albums/{aid.group(1)}",
            "title": html.unescape(title.group(1)) if title else "",
            "photo_count": int(count.group(1)) if count else None,
            "cover_url": html.unescape(cover.group(1)) if cover else None,
        })
    return cards


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("store_url", help="any URL on the store, e.g. https://STORE.x.yupoo.com")
    parser.add_argument("-m", "--manifest", default="manifest.parquet",
                        help="manifest file to create or update (default: manifest.parquet)")
    parser.add_argument("--full", action="store_true",
                        help="recrawl every index page instead of stopping at known albums")
    args = parser.parse_args()

    m = STORE_RE.match(args.store_url.strip())
    if not m:
        sys.exit(f"error: not a Yupoo store URL: {args.store_url}")
    store = m.group(1)
    referer = f"https://{store}.x.yupoo.com/albums"

    try:
        df = pd.read_parquet(args.manifest)
        known = set(df["album_id"])
    except FileNotFoundError:
        df, known = pd.DataFrame(), set()

    now = pd.Timestamp.utcnow().floor("s")
    new_rows, seen_cards = [], {}
    page = 0
    while True:
        page += 1
        cards = parse_index_page(fetch_text(f"{referer}?page={page}", referer), store)
        if not cards:
            break
        fresh = [c for c in cards if c["album_id"] not in known and c["album_id"] not in seen_cards]
        seen_cards.update((c["album_id"], c) for c in cards)
        for c in fresh:
            c["sku"], c["date_label"] = split_title(c["title"])
            c["guessed_date"] = guess_date(c["date_label"], now)
            c["first_seen"] = c["last_seen"] = now
            new_rows.append(c)
        print(f"page {page}: {len(cards)} albums, {len(fresh)} new")
        if not fresh and not args.full:
            break
        time.sleep(random.uniform(0.6, 1.4))

    if not df.empty and seen_cards:
        seen_mask = df["album_id"].isin(seen_cards)
        df.loc[seen_mask, "last_seen"] = now
        if args.full:  # a full recrawl also refreshes the mutable fields
            for col in ("title", "photo_count", "cover_url"):
                df.loc[seen_mask, col] = df.loc[seen_mask, "album_id"].map(
                    lambda a: seen_cards[a][col]
                )
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    if df.empty:
        sys.exit("error: no albums found — wrong URL, or the store markup changed.")
    df = df.sort_values("album_id", ascending=False).reset_index(drop=True)
    df.to_parquet(args.manifest, index=False)
    print(f"{args.manifest}: {len(new_rows)} new, {len(df)} albums total")


if __name__ == "__main__":
    main()
