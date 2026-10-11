"""Deck list viewer and editor with the official card images: python -m zero.ui (run.bat ui).

Opens http://localhost:8765 : every deck of the repository (meta lists, other sources, solo
decks, search results) or your own pasted list, shown as cards (images from TCGdex, French
first, then English), with the checks of the game (20 cards, 2 copies, a Basic Pokémon) and
export as text for the engine. Standard library only; nothing leaves your machine except the
image requests.
"""
import argparse
import glob
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import engine

PAGE = r"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>pocket-zero decks</title>
<style>
:root{--bg:#f6f6f8;--fg:#1d1d22;--card:#fff;--mut:#6b6b76;--ok:#1f8a4c;--ko:#c2352b;--acc:#3557c4}
@media (prefers-color-scheme:dark){:root{--bg:#16161a;--fg:#ececf1;--card:#22222a;--mut:#9a9aa6;--ok:#4cc27e;--ko:#ff7468;--acc:#8ea6ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif}
header{padding:12px 16px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
h1{font-size:18px;margin:0 12px 0 0}select,button,textarea{font:inherit;color:inherit;background:var(--card);border:1px solid #8884;border-radius:8px;padding:6px 10px}
main{display:grid;grid-template-columns:minmax(260px,340px) 1fr;gap:16px;padding:0 16px 16px}
@media (max-width:760px){main{grid-template-columns:1fr}}
textarea{width:100%;box-sizing:border-box;height:420px;font-family:ui-monospace,monospace;font-size:13px}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:10px}
.c{position:relative;background:var(--card);border-radius:10px;overflow:hidden;box-shadow:0 1px 3px #0003}
.c img{width:100%;display:block;aspect-ratio:367/512;object-fit:cover}
.c .t{padding:4px 6px;font-size:12px;color:var(--mut)}.c .n{position:absolute;top:6px;right:6px;background:#000c;color:#fff;border-radius:12px;padding:1px 8px;font-weight:600}
.c .ph{aspect-ratio:367/512;display:flex;align-items:center;justify-content:center;text-align:center;padding:8px;font-size:13px}
#checks span{margin-right:14px}.ok{color:var(--ok)}.ko{color:var(--ko)}#energy{color:var(--acc);font-weight:600}
@media print{header,#left{display:none}main{display:block}}
</style></head><body>
<header><h1>pocket-zero · decklists</h1>
<select id="decks"></select>
<select id="lang"><option value="fr">Images en français</option><option value="en">Images en anglais</option></select>
<button onclick="copyText()">Copier le texte</button><button onclick="window.print()">Imprimer / PDF</button></header>
<main><div id="left"><textarea id="txt" spellcheck="false" placeholder="Colle une decklist (format du moteur : « 2 Oricorio A3 66 »)"></textarea>
<p style="color:var(--mut);font-size:13px">Modifie le texte : la liste se met à jour. Lignes « # » = commentaires (sources).</p></div>
<div><div id="checks"></div><p id="energy"></p><div id="grid"></div></div></main>
<script>
let CARDS={};
const $=id=>document.getElementById(id);
function pad(n){return String(parseInt(n)).padStart(3,'0')}
function imgs(set,num){const l=$('lang').value,o=l=='fr'?'en':'fr',n=pad(num);
 return [`https://assets.tcgdex.net/${l}/tcgp/${set}/${n}/high.webp`,`https://assets.tcgdex.net/${o}/tcgp/${set}/${n}/high.webp`,
 `https://limitlesstcg.nyc3.cdn.digitaloceanspaces.com/pocket/${set}/${set}_${n}_EN.webp`]}
function parse(t){const out=[];let energy='';for(const raw of t.split('\n')){const l=raw.trim();
 if(!l||l.startsWith('#'))continue;if(/^energy:/i.test(l)){energy=l.split(':')[1].trim();continue}
 const m=l.match(/^(\d+)\s+(.*?)\s+([A-Za-z0-9-]+)\s+(\d+)$/);if(m)out.push({n:+m[1],name:m[2],set:m[3],num:m[4],id:`${m[3]} ${pad(m[4])}`})}
 return {cards:out,energy}}
function render(){const {cards,energy}=parse($('txt').value);const g=$('grid');g.innerHTML='';
 let total=0,basic=0,names={},unknown=[];
 for(const c of cards){total+=c.n;const info=CARDS[c.id];if(!info)unknown.push(c.id);
  if(info&&info.basic)basic+=c.n;const k=(info?info.name:c.name).toLowerCase();names[k]=(names[k]||0)+c.n;
  const d=document.createElement('div');d.className='c';const srcs=imgs(c.set,c.num);const im=new Image();let i=0;
  im.alt=c.name;im.loading='lazy';im.src=srcs[0];im.onerror=()=>{i++;if(i<srcs.length)im.src=srcs[i];else{im.replaceWith(Object.assign(document.createElement('div'),{className:'ph',textContent:c.name}))}};
  d.append(im);d.insertAdjacentHTML('beforeend',`<span class="n">×${c.n}</span><div class="t">${c.name} · ${c.id}</div>`);g.append(d)}
 const max=Math.max(0,...Object.values(names));
 const chk=(ok,txt)=>`<span class="${ok?'ok':'ko'}">${ok?'✓':'✗'} ${txt}</span>`;
 $('checks').innerHTML=chk(total==20,`${total}/20 cartes`)+chk(max<=2,'2 exemplaires max')+chk(basic>0,'Pokémon de base')+(unknown.length?chk(false,'inconnues : '+unknown.join(', ')):'');
 $('energy').textContent=energy?'Énergie : '+energy:''}
function copyText(){navigator.clipboard.writeText($('txt').value)}
async function load(p){const r=await fetch('/deck?path='+encodeURIComponent(p));$('txt').value=await r.text();render()}
(async()=>{CARDS=await (await fetch('/cards')).json();const list=await (await fetch('/decks')).json();
 let grp='';for(const p of list){const g=p.split('/').slice(0,-1).join('/');if(g!=grp){grp=g;const o=document.createElement('optgroup');o.label=g;$('decks').append(o)}
  const o=document.createElement('option');o.value=p;o.textContent=p.split('/').pop().replace('.txt','');$('decks').lastChild.append(o)}
 $('decks').onchange=e=>load(e.target.value);$('txt').oninput=render;$('lang').onchange=render;if(list.length)load(list[0])})();
</script></body></html>"""


def deck_files():
    pats = ['lab/decks/meta/*.txt', 'lab/decks/sources/*/*.txt', 'results/**/*.txt', 'lab/decks/solo/*/*.txt']
    out = []
    for p in pats:
        out += sorted(glob.glob(os.path.join(engine.ROOT, p), recursive=True))
    return [os.path.relpath(p, engine.ROOT).replace(os.sep, '/') for p in out]


def cards():
    db = {}
    for e in json.load(open(os.path.join(engine.ROOT, 'database.json'), encoding='utf-8')):
        (kind, c), = e.items()
        db[c['id']] = {'name': c['name'], 'basic': kind == 'Pokemon' and c.get('stage') == 0}
    return db


class Handler(BaseHTTPRequestHandler):
    def send(self, body, ctype='application/json'):
        b = body.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', ctype + '; charset=utf-8')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        u = urlparse(self.path)
        if u.path == '/':
            self.send(PAGE, 'text/html')
        elif u.path == '/cards':
            self.send(json.dumps(CARDS))
        elif u.path == '/decks':
            self.send(json.dumps(deck_files()))
        elif u.path == '/deck':
            p = parse_qs(u.query).get('path', [''])[0]
            if p not in deck_files():  # only the repository's deck files
                self.send_error(404)
                return
            self.send(open(os.path.join(engine.ROOT, p), encoding='utf-8').read(), 'text/plain')
        else:
            self.send_error(404)

    def log_message(self, *a):
        pass


CARDS = {}


def main():
    ap = argparse.ArgumentParser(description='Decklists avec les images officielles')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--no-browser', action='store_true')
    a = ap.parse_args()
    CARDS.update(cards())
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), Handler)
    url = f'http://localhost:{a.port}'
    print(f'Interface : {url}  (Ctrl+C pour arrêter)', flush=True)
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
