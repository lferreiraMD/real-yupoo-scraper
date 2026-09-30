#!/usr/bin/env python3
"""Tiny local web app for yes/no/lume labeling of downloaded photos. Stdlib only.

    python3 ml/label_app.py photos_big --question "Is the dial clearly visible?"

Then open http://127.0.0.1:8765 and use the keyboard:
    Y = yes    N = no    L = lume shot    U = unsure
    Left / Z = previous    Right = forward    F = first unlabeled

Any item can be revisited and relabeled: the CSV is an append-only log
(path,label,ts) and the last row per path wins — train_probe.py reads it
the same way, so fixes are just newer rows. Restart-safe; sampling is
stratified (--per-album per album) and shuffled with a fixed seed, so the
queue order is stable across restarts.
"""

import argparse
import csv
import datetime as dt
import glob
import json
import os
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

PAGE = """<!doctype html><meta charset="utf-8"><title>labeler</title>
<style>
 body{margin:0;background:#111;color:#eee;font:16px system-ui;display:flex;
      flex-direction:column;align-items:center;min-height:100vh}
 #q{margin:12px;font-size:20px} #img{max-width:96vw;max-height:74vh}
 #bar{margin:8px;color:#9a9} #lab{min-height:22px;color:#fc6;font-weight:600}
 kbd{background:#333;border-radius:4px;padding:1px 6px;margin:0 2px}
</style>
<div id="q"></div><img id="img"><div id="bar"></div><div id="lab"></div>
<div><kbd>Y</kbd> yes <kbd>N</kbd> no <kbd>L</kbd> lume <kbd>U</kbd> unsure
 &nbsp; <kbd>&larr;</kbd>/<kbd>Z</kbd> back <kbd>&rarr;</kbd> forward
 <kbd>F</kbd> first unlabeled</div>
<script>
let i=null, frontier=0, cur=null;
async function load(n){
  const r=await (await fetch('/item?i='+n)).json();
  document.getElementById('q').textContent=r.question;
  i=r.i; frontier=r.frontier; cur=r.path;
  const img=document.getElementById('img');
  if(r.done){img.style.display='none';
    document.getElementById('bar').textContent='All '+r.total+' labeled.';
    document.getElementById('lab').textContent='';return;}
  img.style.display='';
  img.src='/img/'+encodeURIComponent(r.path);
  document.getElementById('bar').textContent=
    r.labeled+' / '+r.total+'   #'+(r.i+1)+'   '+r.path;
  document.getElementById('lab').textContent=r.label?'current label: '+r.label:'';
}
async function send(l){ if(!cur)return;
  await fetch('/label',{method:'POST',body:JSON.stringify({path:cur,label:l})});
  load(i<frontier? i+1 : -1); }
document.addEventListener('keydown',e=>{
  const k=e.key.toLowerCase();
  if(k==='y')send('yes'); else if(k==='n')send('no');
  else if(k==='l')send('lume'); else if(k==='u')send('unsure');
  else if(k==='arrowleft'||k==='z')load(Math.max(0,i-1));
  else if(k==='arrowright')load(i+1);
  else if(k==='f')load(-1);});
load(-1);
</script>"""


class State:
    def __init__(self, args):
        self.root = args.photos_dir
        self.labels_path = args.labels
        self.question = args.question
        self.labeled = {}
        if os.path.exists(self.labels_path):
            with open(self.labels_path, newline="") as f:
                for row in csv.DictReader(f):  # append-only log: last row wins
                    self.labeled[row["path"]] = row["label"]
        if args.paths:
            with open(args.paths) as f:
                queue = [line.strip() for line in f if line.strip()]
        else:
            by_album = {}
            for p in glob.glob(os.path.join(self.root, "*", "*.jp*g")):
                rel = os.path.relpath(p, self.root)
                by_album.setdefault(rel.split(os.sep)[0], []).append(rel)
            rng = random.Random(42)
            queue = []
            for album in sorted(by_album):
                photos = sorted(by_album[album])
                rng.shuffle(photos)
                queue.extend(photos[: args.per_album])
            rng.shuffle(queue)
        self.queue = queue
        self.queue_set = set(queue)

    def frontier(self):
        for n, p in enumerate(self.queue):
            if p not in self.labeled:
                return n
        return len(self.queue)

    def label(self, path, label):
        self.labeled[path] = label
        new = not os.path.exists(self.labels_path)
        with open(self.labels_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["path", "label", "ts"])
            w.writerow([path, label, dt.datetime.now().isoformat(timespec="seconds")])


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(PAGE.encode())
            elif self.path.startswith("/item"):
                qs = parse_qs(urlparse(self.path).query)
                want = int(qs.get("i", ["-1"])[0])
                frontier = state.frontier()
                i = frontier if want < 0 or want > frontier else want
                done = i >= len(state.queue)
                path = None if done else state.queue[i]
                self._json({"i": i, "frontier": frontier, "done": done,
                            "path": path, "label": state.labeled.get(path),
                            "question": state.question,
                            "labeled": len(state.labeled),
                            "total": len(state.queue)})
            elif self.path.startswith("/img/"):
                rel = os.path.normpath(self.path[5:].replace("%2F", "/"))
                full = os.path.join(state.root, rel)
                if rel.startswith("..") or not os.path.isfile(full):
                    self.send_response(404)
                    self.end_headers()
                    return
                with open(full, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode() if length else "{}"
            if self.path == "/label":
                req = json.loads(body)
                if (req.get("label") in ("yes", "no", "lume", "unsure")
                        and req.get("path") in state.queue_set):
                    state.label(req["path"], req["label"])
                self._json({"ok": True})
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photos_dir", help="directory tree of ALBUM_ID/*.jpeg")
    parser.add_argument("--labels", default="labels.csv")
    parser.add_argument("--question", default="Is the watch dial clearly visible?")
    parser.add_argument("--per-album", type=int, default=2,
                        help="photos sampled per album (default: 2)")
    parser.add_argument("--paths", default=None,
                        help="file of photo paths (one per line) to use as the "
                             "queue verbatim — audit mode, overrides sampling")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    state = State(args)
    print(f"queue: {len(state.queue)} photos ({len(state.queue) - len(state.labeled)}"
          f" unlabeled) -> http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), make_handler(state)).serve_forever()


if __name__ == "__main__":
    main()
