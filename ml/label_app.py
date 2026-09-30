#!/usr/bin/env python3
"""Tiny local web app for yes/no labeling of downloaded photos. Stdlib only.

    python3 ml/label_app.py photos_big --question "Is the dial clearly visible?"

Then open http://127.0.0.1:8765 and use the keyboard:
    Y = yes    N = no    L = lume shot    U = unsure
    Left/Right = step through earlier photos (Shift = +/-10)  to fix mistakes
    Labeling an already-labeled photo overwrites its stored label.

Labels live in --labels (CSV: path,label,ts), rewritten atomically on every
change. Restart-safe: existing labels are loaded on start. Sampling is
stratified — up to --per-album photos per album, shuffled with a fixed seed
so the queue order is stable across restarts.
"""

import argparse
import csv
import datetime as dt
import glob
import json
import os
import random
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer

PAGE = """<!doctype html><meta charset="utf-8"><title>labeler</title>
<style>
 body{margin:0;background:#111;color:#eee;font:16px system-ui;display:flex;
      flex-direction:column;align-items:center;min-height:100vh}
 #q{margin:12px;font-size:20px} #img{max-width:96vw;max-height:74vh}
 #bar{margin:8px;color:#9a9} #lab{margin:4px;color:#fc6;min-height:20px}
 kbd{background:#333;border-radius:4px;padding:1px 6px;margin:0 2px}
</style>
<div id="q"></div><img id="img"><div id="lab"></div><div id="bar"></div>
<div><kbd>Y</kbd> yes <kbd>N</kbd> no <kbd>L</kbd> lume <kbd>U</kbd> unsure
 <kbd>&larr;</kbd><kbd>&rarr;</kbd> browse (shift=10)</div>
<script>
let i=null;
async function load(idx){
  const r=await (await fetch('/item?i='+(idx===null?'':idx))).json();
  document.getElementById('q').textContent=r.question;
  if(r.done){document.getElementById('img').style.display='none';
    document.getElementById('lab').textContent='';
    document.getElementById('bar').textContent='All '+r.total+' labeled. Done!';i=null;return;}
  i=r.i;
  document.getElementById('img').style.display='';
  document.getElementById('img').src='/img/'+encodeURIComponent(r.path);
  document.getElementById('lab').textContent=r.label?('current label: '+r.label):'';
  document.getElementById('bar').textContent='#'+(r.i+1)+' of '+r.total+'   ('+r.labeled+' labeled)   '+r.path;
}
async function send(l){ if(i===null)return;
  const r=await (await fetch('/label',{method:'POST',
    body:JSON.stringify({i:i,label:l})})).json();
  load(r.next); }
document.addEventListener('keydown',e=>{
  const k=e.key;
  if(k==='y'||k==='Y')send('yes');
  else if(k==='n'||k==='N')send('no');
  else if(k==='l'||k==='L')send('lume');
  else if(k==='u'||k==='U')send('unsure');
  else if(k==='ArrowLeft'&&i!==null)load(Math.max(0,i-(e.shiftKey?10:1)));
  else if(k==='ArrowRight'&&i!==null)load(i+(e.shiftKey?10:1));});
load(null);
</script>"""


class State:
    def __init__(self, args):
        self.root = args.photos_dir
        self.labels_path = args.labels
        self.question = args.question
        self.labels = {}
        if os.path.exists(self.labels_path):
            with open(self.labels_path, newline="") as f:
                for row in csv.DictReader(f):
                    self.labels[row["path"]] = (row["label"], row["ts"])
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

    def first_unlabeled(self, start=0):
        for j in range(start, len(self.queue)):
            if self.queue[j] not in self.labels:
                return j
        for j in range(len(self.queue)):
            if self.queue[j] not in self.labels:
                return j
        return None

    def set_label(self, i, label):
        path = self.queue[i]
        self.labels[path] = (label, dt.datetime.now().isoformat(timespec="seconds"))
        self._write_all()

    def _write_all(self):
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.labels_path) or ".")
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "ts"])
            in_queue = set(self.queue)
            for path in self.queue:
                if path in self.labels:
                    w.writerow([path, *self.labels[path]])
            for path, val in self.labels.items():  # labels from older queues
                if path not in in_queue:
                    w.writerow([path, *val])
        os.replace(tmp, self.labels_path)


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

        def _item(self, i):
            if i is None:
                return self._json({"done": True, "question": state.question,
                                   "total": len(state.queue)})
            i = max(0, min(i, len(state.queue) - 1))
            path = state.queue[i]
            label = state.labels.get(path, (None,))[0]
            self._json({"i": i, "path": path, "label": label, "done": False,
                        "question": state.question, "labeled": len(state.labels),
                        "total": len(state.queue)})

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(PAGE.encode())
            elif self.path.startswith("/item"):
                q = self.path.partition("?i=")[2]
                self._item(int(q) if q else state.first_unlabeled())
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
                i = req.get("i")
                if (req.get("label") in ("yes", "no", "lume", "unsure")
                        and isinstance(i, int) and 0 <= i < len(state.queue)):
                    state.set_label(i, req["label"])
                self._json({"ok": True, "next": state.first_unlabeled(i + 1 if isinstance(i, int) else 0)})
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
    remaining = sum(1 for p in state.queue if p not in state.labels)
    print(f"queue: {len(state.queue)} photos ({remaining} unlabeled) -> "
          f"http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), make_handler(state)).serve_forever()


if __name__ == "__main__":
    main()
