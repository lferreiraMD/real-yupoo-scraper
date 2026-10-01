#!/usr/bin/env python3
"""Tiny local web app for yes/no/lume labeling of downloaded marketing photos. Stdlib only.

    python3 ml/label_app.py photos_big --question "Is the dial clearly visible?"

Then open http://127.0.0.1:8765 and use the keyboard:
    Y = yes    N = no    L = lume shot    U = unsure
    Left / Z = previous    Right = forward    F = first unlabeled

Any item can be revisited and relabeled: the CSV is an append-only log
(path,label,ts) and the last row per path wins — train_probe.py reads it
the same way, so fixes are just newer rows. Restart-safe; sampling is
stratified (--per-album per album) and shuffled with a fixed seed, so the
queue order is stable across restarts.

Pick mode shows all candidate photos of one album side by side and records
the best one (rank_dials.py --pick-queue writes such a queue):

    python3 ml/label_app.py photos_big --pick --paths pick_paths.txt --labels pick_labels.csv

Consecutive paths from the same album form one screen. Keys: 1-9 (and 0 for
the 10th) or a click picks a photo, N = none of them shows the dial readably,
U = unsure; arrows/Z/F navigate as above. Each pick is logged as
(path,best,ts); N and U are logged as (ALBUM_ID/,none|unsure,ts).
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

PICK_PAGE = """<!doctype html><meta charset="utf-8"><title>pick the best</title>
<style>
 body{margin:0;background:#111;color:#eee;font:16px system-ui;text-align:center}
 #q{margin:10px;font-size:20px} #bar{margin:6px;color:#9a9} #lab{min-height:22px;color:#fc6;font-weight:600}
 #grid{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;padding:8px}
 .c{position:relative;cursor:pointer;border:4px solid transparent;border-radius:6px}
 .c.sel{border-color:#fc6} .c img{display:block;max-height:42vh;max-width:30vw}
 .n{position:absolute;top:4px;left:4px;background:#000c;padding:2px 8px;border-radius:4px;font-weight:700}
 kbd{background:#333;border-radius:4px;padding:1px 6px;margin:0 2px}
</style>
<div id="q"></div><div id="bar"></div><div id="lab"></div><div id="grid"></div>
<div><kbd>1</kbd>-<kbd>9</kbd>,<kbd>0</kbd> or click = best <kbd>N</kbd> none readable <kbd>U</kbd> unsure
 &nbsp; <kbd>&larr;</kbd>/<kbd>Z</kbd> back <kbd>&rarr;</kbd> forward <kbd>F</kbd> first unpicked</div>
<script>
let i=null, frontier=0, cur=null;
async function load(n){
  const r=await (await fetch('/album?i='+n)).json();
  document.getElementById('q').textContent=r.question;
  i=r.i; frontier=r.frontier; cur=r;
  const g=document.getElementById('grid'); g.innerHTML='';
  if(r.done){document.getElementById('bar').textContent='All '+r.total+' albums picked.';
    document.getElementById('lab').textContent='';cur=null;return;}
  r.paths.forEach((p,k)=>{
    const c=document.createElement('div'); c.className='c'+(r.pick===p?' sel':'');
    c.innerHTML='<span class="n">'+(k+1)+'</span>';
    const img=document.createElement('img'); img.src='/img/'+encodeURIComponent(p);
    c.appendChild(img); c.onclick=()=>send(p,'best'); g.appendChild(c);});
  document.getElementById('bar').textContent=
    r.labeled+' / '+r.total+' albums   #'+(r.i+1)+'   album '+r.album+'   '+r.paths.length+' photos';
  document.getElementById('lab').textContent=r.pick?'current: '+(r.pick.includes('/')?
    'photo '+(r.paths.indexOf(r.pick)+1):r.pick):'';
}
async function send(path,label){ if(!cur)return;
  await fetch('/pick',{method:'POST',body:JSON.stringify({album:cur.album,path:path,label:label})});
  load(i<frontier? i+1 : -1); }
document.addEventListener('keydown',e=>{
  const k=e.key.toLowerCase();
  if(cur&&/^[0-9]$/.test(k)){const n=k==='0'?9:+k-1; if(n<cur.paths.length)send(cur.paths[n],'best');}
  else if(k==='n')send(null,'none'); else if(k==='u')send(null,'unsure');
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


class PickState(State):
    """One screen per album; the label log holds the picked path per album."""

    def __init__(self, args):
        super().__init__(args)
        self.albums = []
        for p in self.queue:
            album = p.split("/")[0]
            if not self.albums or self.albums[-1][0] != album:
                self.albums.append((album, []))
            self.albums[-1][1].append(p)
        self.picked = {}
        if os.path.exists(self.labels_path):
            with open(self.labels_path, newline="") as f:
                for row in csv.DictReader(f):  # last row per album wins
                    path, label = row["path"], row["label"]
                    self.picked[path.split("/")[0]] = path if label == "best" else label

    def frontier(self):
        for n, (album, _) in enumerate(self.albums):
            if album not in self.picked:
                return n
        return len(self.albums)

    def pick(self, album, path, label):
        self.picked[album] = path if label == "best" else label
        self.label(path if label == "best" else album + "/", label)


def make_handler(state):
    pick_mode = isinstance(state, PickState)

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
                self.wfile.write((PICK_PAGE if pick_mode else PAGE).encode())
            elif pick_mode and self.path.startswith("/album"):
                qs = parse_qs(urlparse(self.path).query)
                want = int(qs.get("i", ["-1"])[0])
                frontier = state.frontier()
                i = frontier if want < 0 or want > frontier else want
                done = i >= len(state.albums)
                album, paths = (None, []) if done else state.albums[i]
                self._json({"i": i, "frontier": frontier, "done": done, "album": album,
                            "paths": paths, "pick": state.picked.get(album),
                            "question": state.question, "labeled": len(state.picked),
                            "total": len(state.albums)})
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
            elif pick_mode and self.path == "/pick":
                req = json.loads(body)
                album, path, label = req.get("album"), req.get("path"), req.get("label")
                paths = dict(state.albums).get(album)
                if paths is not None and (label in ("none", "unsure")
                                          or (label == "best" and path in paths)):
                    state.pick(album, path, label)
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
    parser.add_argument("--pick", action="store_true",
                        help="pick the best photo per album (needs --paths)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.pick and not args.paths:
        parser.error("--pick needs --paths")
    if args.pick and args.question == parser.get_default("question"):
        args.question = "Which photo gives the clearest, most readable view of the dial?"

    if args.pick:
        state = PickState(args)
        print(f"queue: {len(state.albums)} albums ({len(state.albums) - len(state.picked)}"
              f" unpicked) -> http://127.0.0.1:{args.port}", flush=True)
    else:
        state = State(args)
        print(f"queue: {len(state.queue)} photos ({len(state.queue) - len(state.labeled)}"
              f" unlabeled) -> http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), make_handler(state)).serve_forever()


if __name__ == "__main__":
    main()
