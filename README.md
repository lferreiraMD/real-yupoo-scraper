# real-yupoo-scraper

Downloads every photo and video from a public album at original quality. One file, standard library only, Python 3.8+. No accounts, no API keys, no cloud services.

Yupoo serves full-resolution originals (the same files the seller uploaded, EXIF intact), but rejects any request without a `Referer` header from the album's own domain. That header is the entire trick; everything else here is parsing and bookkeeping.

## Usage

```
python3 yupoo_scraper.py "https://STORE.x.yupoo.com/albums/ALBUM_ID"
```

Quote the URL — the `?uid=1` query string most Yupoo links carry would otherwise be eaten by your shell. Output goes to `./STORE_TITLE_ALBUMID/` unless you pass `-o some/dir`.

```
$ python3 yupoo_scraper.py "https://andiotwatches.x.yupoo.com/albums/256511469?uid=1"
Album "AE171547/9.27" by andiotwatches: 18 items -> andiotwatches_AE171547_9.27_256511469/
  [001/18] photo 001_0927dae7ed.jpeg  (1.81 MB)
  [002/18] video 002_12084770.mov  (2.51 MB)
  ...
Done: 18/18 items downloaded.
```

Files are numbered in album order. Videos are fetched as the seller's original upload when the transcoded 1080p/720p/480p variants aren't available (they often aren't; the original is better quality anyway). An `album.json` manifest is written alongside the media with the title, source URLs, original filenames, and per-file status. Exit code is 0 only if every item downloaded.

## Limits

- Album URLs only (`/albums/NNNN`). Store fronts and category pages are not crawled; grab the album links yourself and loop.
- Password-protected albums are detected and refused, not unlocked.
- Paginated albums are followed page by page, but I have only tested albums that fit on one page.
- If Yupoo changes its album markup (`data-origin-src`, `data-path`), parsing breaks. It's ~200 lines; fixing it should be quick.

Be sensible: this is for downloading albums you have a legitimate reason to keep. Don't hammer the site — the script downloads sequentially on purpose.

## Store-scale pipeline

For downloading a whole store rather than one album, two more scripts split the job into an index phase and a download phase (these two need `pip install -r requirements.txt`; the single-album scraper stays stdlib-only):

```
python3 yupoo_manifest.py "https://STORE.x.yupoo.com"            # -> manifest.parquet
python3 yupoo_download.py manifest.parquet --size big            # -> photos_big/ALBUM_ID/
```

`yupoo_manifest.py` crawls the store's album index into an append-only Parquet file keyed on `album_id`, with the title split into SKU and date columns. The index is sorted newest-first, so re-running it is cheap: it stops at the first page with no unseen albums, adding only what's new. `--full` recrawls everything and refreshes photo counts.

`yupoo_download.py` walks the manifest and downloads each album's photos (videos are skipped) at the size you pick — `small`, `medium`, `big` (810×1080, ~330 KB, the default) or `origin` (full resolution, ~10× bigger). Traffic is shaped like a person browsing: a small concurrent burst per album, a jittered pause between albums, and a long cooldown if the server ever pushes back. State lives on disk (`ALBUM_ID/album.json`), so the run is idempotent and resumable — kill it any time, re-run, and it refetches only what's missing. `--max-albums` and `--max-hours` let you run it in nightly chunks.

Rough numbers from a 5,600-album store: the full manifest crawl is 47 requests (~1 minute); the download is ~110k requests, an overnight run at default pacing, ~35 GB at `big`.

## Why this exists

The GitHub repo named `Yupoo-Scraper` that ranks well in search contains no code — it's an advertisement for a paid cloud scraper billed per image. This one is the actual scraper.
