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

Pick mode shows the candidate photos of one watch side by side and records
the best one — or several, when the screen turns out to hold more than one
watch (split_watches.py --pick-queue writes such a queue):

    python3 ml/label_app.py photos_big --pick --paths pick_paths.txt --labels pick_labels.csv

Queue lines are "path<TAB>screen" (or bare paths, grouped by album);
consecutive lines with the same screen form one screen. Keys: 1-9 (and 0 for
the 10th) or a click toggle a photo, W (or Enter) writes the selection, N = none of
them shows the dial readably, U = unsure; arrows/Z/F navigate as above. The
log has columns screen,path,label,save,ts: one "best" row per selected photo,
or one "none"/"unsure" row with an empty path. Every save gets a new save
number and replaces the screen's earlier answer.
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
 .c.sel .n{background:#fc6;color:#111}
 kbd{background:#333;border-radius:4px;padding:1px 6px;margin:0 2px}
</style>
<div id="q"></div><div id="bar"></div><div id="lab"></div><div id="grid"></div>
<div><kbd>1</kbd>-<kbd>9</kbd>,<kbd>0</kbd> or click = select/unselect <kbd>W</kbd> write (save) selection
 <kbd>N</kbd> none readable <kbd>U</kbd> unsure &nbsp; <kbd>&larr;</kbd>/<kbd>Z</kbd> back
 <kbd>&rarr;</kbd> forward <kbd>F</kbd> first unpicked</div>
<script>
let i=null, frontier=0, cur=null, sel=new Set();
function draw(){
  document.querySelectorAll('.c').forEach((c,k)=>c.classList.toggle('sel',sel.has(cur.paths[k])));
  const s=[...sel].map(p=>cur.paths.indexOf(p)+1).sort((a,b)=>a-b);
  document.getElementById('lab').textContent=s.length?'selected: '+s.join(', ')+'  (W to save)':
    (cur.label&&cur.label!=='best'?'saved: '+cur.label:'');
}
async function load(n){
  const r=await (await fetch('/album?i='+n)).json();
  document.getElementById('q').textContent=r.question;
  i=r.i; frontier=r.frontier; cur=r;
  const g=document.getElementById('grid'); g.innerHTML='';
  if(r.done){document.getElementById('bar').textContent='All '+r.total+' screens answered.';
    document.getElementById('lab').textContent='';cur=null;return;}
  sel=new Set(r.picks);
  r.paths.forEach((p,k)=>{
    const c=document.createElement('div'); c.className='c';
    c.innerHTML='<span class="n">'+(k+1)+'</span>';
    const img=document.createElement('img'); img.src='/img/'+encodeURIComponent(p);
    c.appendChild(img); c.onclick=()=>toggle(k); g.appendChild(c);});
  document.getElementById('bar').textContent=r.labeled+' / '+r.total+' screens   #'+(r.i+1)+
    '   '+r.screen+(r.note?'  ('+r.note+')':'')+'   '+r.paths.length+' photos';
  draw();
}
function toggle(k){ const p=cur.paths[k]; sel.has(p)?sel.delete(p):sel.add(p); draw(); }
async function send(label){ if(!cur)return;
  const paths=label==='best'?[...sel]:[];
  if(label==='best'&&!paths.length)return;
  await fetch('/pick',{method:'POST',body:JSON.stringify({screen:cur.screen,paths:paths,label:label})});
  load(i<frontier? i+1 : -1); }
document.addEventListener('keydown',e=>{
  const k=e.key.toLowerCase();
  if(cur&&/^[0-9]$/.test(k)){const n=k==='0'?9:+k-1; if(n<cur.paths.length)toggle(n);}
  else if(k==='w'||k==='enter')send('best');
  else if(k==='n')send('none'); else if(k==='u')send('unsure');
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


class PickState:
    """One screen per watch; the log keeps the latest saved selection per screen."""

    def __init__(self, args):
        self.root = args.photos_dir
        self.labels_path = args.labels
        self.question = args.question
        self.screens, self.notes = [], {}
        with open(args.paths) as f:
            for line in f:
                if not line.strip():
                    continue
                path, _, screen = line.rstrip("\n").partition("\t")
                screen = screen or path.split("/")[0]
                if not self.screens or self.screens[-1][0] != screen:
                    self.screens.append((screen, []))
                self.screens[-1][1].append(path)
        self.paths = dict(self.screens)
        album_screens = {}
        for screen, _ in self.screens:
            album_screens.setdefault(screen.rstrip("abcdefghij"), []).append(screen)
        for album, group in album_screens.items():
            for n, screen in enumerate(group):
                if len(group) > 1:
                    self.notes[screen] = f"watch {n + 1} of {len(group)} in album {album}"
        self.answers, self.save = {}, 0
        if os.path.exists(self.labels_path):
            with open(self.labels_path, newline="") as f:
                for row in csv.DictReader(f):
                    save = int(row["save"])
                    self.save = max(self.save, save)
                    prev = self.answers.get(row["screen"])
                    if prev is None or save > prev["save"]:
                        prev = self.answers[row["screen"]] = {"save": save, "label": row["label"],
                                                              "paths": []}
                    if save == prev["save"] and row["path"]:
                        prev["paths"].append(row["path"])

    def frontier(self):
        for n, (screen, _) in enumerate(self.screens):
            if screen not in self.answers:
                return n
        return len(self.screens)

    def pick(self, screen, paths, label):
        self.save += 1
        self.answers[screen] = {"save": self.save, "label": label, "paths": list(paths)}
        new = not os.path.exists(self.labels_path)
        ts = dt.datetime.now().isoformat(timespec="seconds")
        with open(self.labels_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["screen", "path", "label", "save", "ts"])
            for path in (paths or [""]):
                w.writerow([screen, path, label, self.save, ts])


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
                done = i >= len(state.screens)
                screen, paths = (None, []) if done else state.screens[i]
                answer = state.answers.get(screen, {})
                self._json({"i": i, "frontier": frontier, "done": done, "screen": screen,
                            "note": state.notes.get(screen), "paths": paths,
                            "picks": answer.get("paths", []), "label": answer.get("label"),
                            "question": state.question, "labeled": len(state.answers),
                            "total": len(state.screens)})
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
                screen, picks, label = req.get("screen"), req.get("paths") or [], req.get("label")
                allowed = state.paths.get(screen)
                if allowed is not None and (
                        (label in ("none", "unsure") and not picks)
                        or (label == "best" and picks and set(picks) <= set(allowed))):
                    state.pick(screen, sorted(set(picks)), label)
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
        args.question = ("Select the clearest, most readable dial photo — one per distinct "
                         "watch on this screen")

    if args.pick:
        state = PickState(args)
        print(f"queue: {len(state.screens)} screens ({len(state.screens) - len(state.answers)}"
              f" unanswered) -> http://127.0.0.1:{args.port}", flush=True)
    else:
        state = State(args)
        print(f"queue: {len(state.queue)} photos ({len(state.queue) - len(state.labeled)}"
              f" unlabeled) -> http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), make_handler(state)).serve_forever()


if __name__ == "__main__":
    main()
