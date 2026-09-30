#!/usr/bin/env python3
"""Tiny local web app for yes/no labeling of downloaded photos. Stdlib only.

    python3 ml/label_app.py photos_big --question "Is the dial clearly visible?"

Then open http://127.0.0.1:8765 and use the keyboard:
    Y = yes    N = no    L = lume shot    U = unsure    Z = undo last

Labels append to --labels (CSV: path,label,ts). Restart-safe: already-labeled
paths are skipped. Sampling is stratified — up to --per-album photos from each
album, shuffled with a fixed seed so the queue is stable across restarts.
"""

import argparse
import csv
import datetime as dt
import glob
import json
import os
import random
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PAGE = """<!doctype html><meta charset="utf-8"><title>labeler</title>
<style>
 body{margin:0;background:#111;color:#eee;font:16px system-ui;display:flex;
      flex-direction:column;align-items:center;min-height:100vh}
 #q{margin:12px;font-size:20px} #img{max-width:96vw;max-height:78vh}
 #bar{margin:10px;color:#9a9}
 kbd{background:#333;border-radius:4px;padding:1px 6px;margin:0 2px}
</style>
<div id="q"></div><img id="img"><div id="bar"></div>
<div><kbd>Y</kbd> yes <kbd>N</kbd> no <kbd>L</kbd> lume <kbd>U</kbd> unsure <kbd>Z</kbd> undo</div>
<script>
let cur=null;
async function next(){
  const r=await (await fetch('/next')).json();
  document.getElementById('q').textContent=r.question;
  if(r.done){document.getElementById('img').style.display='none';
    document.getElementById('bar').textContent='All '+r.total+' labeled. Done!';cur=null;return;}
  cur=r.path;
  document.getElementById('img').src='/img/'+encodeURIComponent(r.path);
  document.getElementById('bar').textContent=(r.labeled)+' / '+r.total+'   '+r.path;
}
async function send(l){ if(!cur)return;
  await fetch('/label',{method:'POST',body:JSON.stringify({path:cur,label:l})}); next(); }
async function undo(){ await fetch('/undo',{method:'POST'}); next(); }
document.addEventListener('keydown',e=>{
  const k=e.key.toLowerCase();
  if(k==='y')send('yes'); else if(k==='n')send('no');
  else if(k==='l')send('lume');
  else if(k==='u')send('unsure'); else if(k==='z')undo();});
next();
</script>"""


class State:
    def __init__(self, args):
        self.root = args.photos_dir
        self.labels_path = args.labels
        self.question = args.question
        self.labeled = {}
        if os.path.exists(self.labels_path):
            with open(self.labels_path, newline="") as f:
                for row in csv.DictReader(f):
                    self.labeled[row["path"]] = row["label"]
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
        self.history = []

    def next_unlabeled(self):
        for p in self.queue:
            if p not in self.labeled:
                return p
        return None

    def label(self, path, label):
        self.labeled[path] = label
        self.history.append(path)
        new = not os.path.exists(self.labels_path)
        with open(self.labels_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["path", "label", "ts"])
            w.writerow([path, label, dt.datetime.now().isoformat(timespec="seconds")])

    def undo(self):
        if not self.history:
            return
        path = self.history.pop()
        self.labeled.pop(path, None)
        with open(self.labels_path, newline="") as f:
            rows = [r for r in csv.reader(f)]
        rows = [r for i, r in enumerate(rows) if i == 0 or r[0] != path]
        with open(self.labels_path, "w", newline="") as f:
            csv.writer(f).writerows(rows)


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
            elif self.path == "/next":
                p = state.next_unlabeled()
                self._json({"path": p, "done": p is None, "question": state.question,
                            "labeled": len(state.labeled), "total": len(state.queue)})
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
                if req.get("label") in ("yes", "no", "lume", "unsure") and req.get("path"):
                    state.label(req["path"], req["label"])
                self._json({"ok": True})
            elif self.path == "/undo":
                state.undo()
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
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    state = State(args)
    remaining = sum(1 for p in state.queue if p not in state.labeled)
    print(f"queue: {len(state.queue)} photos ({remaining} unlabeled) -> "
          f"http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), make_handler(state)).serve_forever()


if __name__ == "__main__":
    main()
