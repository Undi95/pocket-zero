"""Deck list viewer and live training page: python -m zero.ui (run.bat ui).

http://localhost:8765 : every deck of the repository (meta lists, other sources, solo decks,
search results) or your own pasted list, shown as cards (French images from Poképédia, then
TCGdex and Limitless), with the checks of the game (20 cards, 2 copies, a Basic Pokémon) and
export as text for the engine.
http://localhost:8765/train : the training run as it goes, refreshed every 20 s: progress and
time left, strength curves, last validation, highlights, and the decks that win the most in the
self-play (runs/<run>/victoires.json).
Standard library only; nothing leaves your machine except the image requests.
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import engine
from . import highlights as highlights_mod
from .decks import KINDS, LABELS_FILE

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
.c img{cursor:zoom-in}
#zoom{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center;z-index:9;cursor:zoom-out}
#zoom img{max-width:94vw;max-height:94vh;border-radius:14px;box-shadow:0 8px 40px #000}
@media print{header,#left,#zoom{display:none}main{display:block}}
</style></head><body>
<header><h1>pocket-zero · decklists</h1>
<select id="decks"></select>
<select id="lang"><option value="fr">Images en français</option><option value="en">Images en anglais</option></select>
<button onclick="copyText()">Copier le texte</button><button onclick="window.print()">Imprimer / PDF</button>
<a href="/train" style="margin-left:auto;color:var(--acc)">Entraînement en direct →</a></header>
<main><div id="left"><textarea id="txt" spellcheck="false" placeholder="Colle une decklist (format du moteur : « 2 Oricorio A3 66 »)"></textarea>
<p style="color:var(--mut);font-size:13px">Modifie le texte : la liste se met à jour. Lignes « # » = commentaires (sources).</p></div>
<div><div id="checks"></div><p id="energy"></p><div id="grid"></div></div></main>
<div id="zoom" onclick="zoomClose()"><img alt=""></div>
<script>
let CARDS={},ZI=-1;
const $=id=>document.getElementById(id);
function pad(n){return String(parseInt(n)).padStart(3,'0')}
// French names of the sets, as Poképédia names its card images ("Carte <extension> <numéro>.png").
const FR_SETS={'A1':'Puissance Génétique','A1a':"L'Île Fabuleuse",'A2':'Choc Spatio-Temporel','A2a':'Lumière Triomphale',
 'A2b':'Réjouissances Rayonnantes','A3':'Gardiens Astraux','A3a':'Crise Interdimensionnelle','A3b':"La Clairière d'Évoli",
 'A4':'Sagesse entre Ciel et Mer','A4a':'Source Secrète','A4b':'Booster de Luxe ex','B1':'Méga-Ascension','B1a':'Embrasement Écarlate',
 'B2':'Parade Onirique','B2a':'Merveilles de Paldea','B2b':'Méga-Rayonnement','B3':'Aura Palpitante','B3a':'Propulsion Paradoxe',
 'B3b':'Jours Heureux','B4':'Domination Céleste','B4a':"L'Ambition de la Team Rocket",'B4b':'Booster de Luxe Méga','P-A':'Promo-A','P-B':'Promo-B'};
function imgs(set,num){const n=pad(num),fr=[],en=[`https://assets.tcgdex.net/en/tcgp/${set}/${n}/high.webp`,
 `https://limitlesstcg.nyc3.cdn.digitaloceanspaces.com/pocket/${set}/${set}_${n}_EN.webp`];
 if(FR_SETS[set])fr.push('https://www.pokepedia.fr/Special:FilePath/'+encodeURIComponent(`Carte_${FR_SETS[set]}_${n}.png`.replace(/ /g,'_'))+'?width=300');
 fr.push(`https://assets.tcgdex.net/fr/tcgp/${set}/${n}/high.webp`);
 return $('lang').value=='fr'?fr.concat(en):en.concat(fr)}
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
  im.onclick=()=>zoomOpen([...$('grid').querySelectorAll('img')].indexOf(im));
  d.append(im);d.insertAdjacentHTML('beforeend',`<span class="n">×${c.n}</span><div class="t">${c.name} · ${c.id}</div>`);g.append(d)}
 const max=Math.max(0,...Object.values(names));
 const chk=(ok,txt)=>`<span class="${ok?'ok':'ko'}">${ok?'✓':'✗'} ${txt}</span>`;
 $('checks').innerHTML=chk(total==20,`${total}/20 cartes`)+chk(max<=2,'2 exemplaires max')+chk(basic>0,'Pokémon de base')+(unknown.length?chk(false,'inconnues : '+unknown.join(', ')):'');
 $('energy').textContent=energy?'Énergie : '+energy:''}
function big(src){return src.replace('?width=300','?width=750')}
function zoomOpen(i){const all=$('grid').querySelectorAll('img');if(i<0||i>=all.length)return;ZI=i;
 const z=$('zoom').querySelector('img');z.onerror=()=>{z.onerror=null;z.src=all[i].currentSrc||all[i].src};z.src=big(all[i].currentSrc||all[i].src);$('zoom').style.display='flex'}
function zoomClose(){$('zoom').style.display='none';ZI=-1}
document.addEventListener('keydown',e=>{if(ZI<0)return;if(e.key=='Escape')zoomClose();
 else if(e.key=='ArrowRight')zoomOpen(ZI+1);else if(e.key=='ArrowLeft')zoomOpen(ZI-1)});
function copyText(){navigator.clipboard.writeText($('txt').value)}
async function load(p){const r=await fetch('/deck?path='+encodeURIComponent(p));$('txt').value=await r.text();render()}
(async()=>{CARDS=await (await fetch('/cards')).json();const list=await (await fetch('/decks')).json();
 let grp='';for(const p of list){const g=p.split('/').slice(0,-1).join('/');if(g!=grp){grp=g;const o=document.createElement('optgroup');o.label=g;$('decks').append(o)}
  const o=document.createElement('option');o.value=p;o.textContent=p.split('/').pop().replace('.txt','');$('decks').lastChild.append(o)}
 $('decks').onchange=e=>load(e.target.value);$('txt').oninput=render;$('lang').onchange=render;
 const want=new URLSearchParams(location.search).get('deck');
 if(want){if(!list.includes(want)){const o=document.createElement('option');o.value=want;o.textContent=want.split('/').pop().replace('.txt','');$('decks').prepend(o)}
  $('decks').value=want;load(want)}else if(list.length)load(list[0])})();
</script></body></html>"""

TRAIN = r"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>pocket-zero entraînement</title>
<style>
:root{--bg:#f6f6f8;--fg:#1d1d22;--card:#fff;--mut:#6b6b76;--ok:#1f8a4c;--ko:#c2352b;--warn:#b7791f;--acc:#3557c4;--acc2:#c2357a;--grid:#8882}
@media (prefers-color-scheme:dark){:root{--bg:#16161a;--fg:#ececf1;--card:#22222a;--mut:#9a9aa6;--ok:#4cc27e;--ko:#ff7468;--warn:#f0b450;--acc:#8ea6ff;--acc2:#ff8ec4}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif}
header{padding:12px 16px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
h1{font-size:18px;margin:0 12px 0 0}h2{font-size:15px;margin:0 0 8px}a{color:var(--acc)}
select{font:inherit;color:inherit;background:var(--card);border:1px solid #8884;border-radius:8px;padding:6px 10px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:14px;padding:0 16px 16px}
.box{background:var(--card);border-radius:12px;padding:12px 14px;box-shadow:0 1px 3px #0002;min-width:0}
.wide{grid-column:1/-1}.mut{color:var(--mut);font-size:13px}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.kpi div b{display:block;font-size:20px}.bar{height:8px;background:var(--grid);border-radius:4px;overflow:hidden;margin:8px 0}
.bar i{display:block;height:100%;background:var(--acc)}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:5px 6px;text-align:left;border-bottom:1px solid var(--grid)}
th{cursor:pointer;color:var(--mut);font-weight:600;white-space:nowrap}td.n,th.n{text-align:right;white-space:nowrap}
.ok{color:var(--ok)}.ko{color:var(--ko)}.warn{color:var(--warn)}
svg{width:100%;height:auto;display:block}svg text{fill:var(--mut);font-size:11px}
pre{white-space:pre-wrap;font:12px ui-monospace,monospace;margin:0;color:var(--mut)}
.leg span{margin-right:12px;font-size:13px}.leg i{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:4px}
.scroll{overflow-x:auto}
</style></head><body>
<header><h1>pocket-zero · entraînement</h1><select id="run"></select>
<span class="mut" id="upd"></span><a href="/" style="margin-left:auto">← Decklists</a></header>
<main>
<div class="box wide" id="status"></div>
<div class="box"><h2>Force du réseau</h2><div id="c_force"></div>
<div class="leg"><span><i style="background:var(--acc)"></i>contre e3 (moyenne 5)</span><span><i style="background:var(--acc2)"></i>contre e2</span><span><i style="background:var(--mut)"></i>contre e3 (brut)</span></div></div>
<div class="box"><h2>Écart aux vrais matchups (RMSE, plus bas = mieux)</h2><div id="c_rmse"></div>
<div class="leg"><span><i style="background:var(--acc)"></i>RMSE</span><span><i style="background:var(--warn)"></i>référence e3 (11,5)</span></div></div>
<div class="box"><h2>Pertes sur les parties de contrôle</h2><div id="c_loss"></div>
<div class="leg"><span><i style="background:var(--acc)"></i>politique</span><span><i style="background:var(--acc2)"></i>valeur ×5</span></div></div>
<div class="box"><h2 id="val_t">Dernière validation</h2><div id="val"></div></div>
<div class="box wide"><h2>Decks qui gagnent le plus</h2>
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">
<select id="win"><option value="1">dernière itération</option><option value="10" selected>10 dernières itérations</option><option value="0">tout le run</option></select>
<select id="kind"><option value="all">tous les decks</option><option value="home">maison (créés par le bot)</option><option value="real">vrais decks (méta, sources, forgés, devs)</option></select>
<select id="min"><option value="5">≥ 5 parties</option><option value="10" selected>≥ 10 parties</option><option value="30">≥ 30 parties</option><option value="100">≥ 100 parties</option></select></div>
<p class="mut" id="dk_info"></p>
<div class="scroll"><table><thead><tr><th>#</th><th data-s="label">Deck</th><th data-s="kind">Type</th><th class="n" data-s="wins">Victoires</th><th class="n" data-s="games">Parties</th><th class="n" data-s="rate">Taux</th></tr></thead><tbody id="decks"></tbody></table></div>
<p class="mut">Résultats de l'auto-jeu contre tout le pool d'entraînement (decks faibles compris, avec un peu d'exploration) : un repérage, pas une preuve. Clique un deck pour voir ses cartes ; à confirmer avec la forge (<code>--deck</code>) ou le ranked.</p></div>
<div class="box"><h2 id="hl_t">Moments forts</h2><div id="hl"></div></div>
<div class="box"><h2>Journal en direct</h2><pre id="tail"></pre></div>
</main>
<script>
const $=id=>document.getElementById(id);let RUN='',STATSV=null,SORT='score',ROWS=[],DK={};
const esc=t=>String(t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function dur(s){s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.round(s%3600/60);return h?`${h} h ${String(m).padStart(2,'0')}`:`${m} min`}
function chart(series,{lo,hi,pct,ref}={}){const W=640,H=200,L=38,R=8,T=8,B=22;const pts=series.flatMap(s=>s.pts);if(!pts.length)return '<p class="mut">Pas encore de données.</p>';
 const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]).concat(ref!=null?[ref]:[]);const x0=Math.min(...xs),x1=Math.max(...xs,x0+1);
 let y0=lo??Math.min(...ys),y1=hi??Math.max(...ys);if(y1-y0<1e-9){y0-=1;y1+=1}const pad=(y1-y0)*.08;if(lo==null)y0-=pad;if(hi==null)y1+=pad;
 const X=x=>L+(x-x0)/(x1-x0)*(W-L-R),Y=y=>T+(1-(y-y0)/(y1-y0))*(H-T-B);let g='';
 for(let k=0;k<=4;k++){const v=y0+(y1-y0)*k/4,y=Y(v);g+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="var(--grid)"/><text x="${L-4}" y="${y+4}" text-anchor="end">${pct?Math.round(v*100)+'%':v.toFixed(v<2?2:1)}</text>`}
 g+=`<text x="${L}" y="${H-6}">${x0}</text><text x="${W-R}" y="${H-6}" text-anchor="end">itération ${x1}</text>`;
 if(ref!=null)g+=`<line x1="${L}" x2="${W-R}" y1="${Y(ref)}" y2="${Y(ref)}" stroke="var(--warn)" stroke-dasharray="5 4"/>`;
 for(const s of series){if(!s.pts.length)continue;const d=s.pts.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1)).join('');
  g+=`<path d="${d}" fill="none" stroke="${s.c}" stroke-width="${s.w||2}" opacity="${s.o||1}"/>`;
  if(s.dots)g+=s.pts.map(p=>`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="2.5" fill="${s.c}"/>`).join('')}
 return `<svg viewBox="0 0 ${W} ${H}">${g}</svg>`}
function avg(pts,k){return pts.map((p,i)=>{const w=pts.slice(Math.max(0,i-k+1),i+1);return [p[0],w.reduce((a,q)=>a+q[1],0)/w.length]})}
function render(st){const m=st.metrics,it=m.map(r=>r.iteration);const col=k=>m.filter(r=>r[k]!=null).map(r=>[r.iteration,r[k]]);
 const e3=col('vs_e3');
 $('c_force').innerHTML=chart([{pts:e3,c:'var(--mut)',w:1,o:.6},{pts:col('vs_e2'),c:'var(--acc2)',w:1.5},{pts:avg(e3,5),c:'var(--acc)',w:2.5}],{pct:1});
 $('c_rmse').innerHTML=chart([{pts:col('rmse_limitless'),c:'var(--acc)',dots:1}],{ref:11.5,lo:0});
 $('c_loss').innerHTML=chart([{pts:col('val_policy_loss'),c:'var(--acc)'},{pts:col('val_value_loss').map(p=>[p[0],p[1]*5]),c:'var(--acc2)'}]);
 const s=st.status;let h='';
 if(s){const best=e3.length?e3.reduce((a,b)=>b[1]>a[1]?b:a):null,last5=e3.slice(-5),m5=last5.length?last5.reduce((a,p)=>a+p[1],0)/last5.length:null;
  const state=s.finished?'<span class="ok">terminé</span>':s.alive?'<span class="ok">● en cours</span>':`<span class="ko">● à l'arrêt</span> (dernier message il y a ${dur(s.quiet)})`;
  const eta=s.eta!=null&&!s.finished&&s.alive?`<b>${new Date(Date.now()+s.eta*1000).toLocaleString('fr-BE',{weekday:'short',hour:'2-digit',minute:'2-digit'})}</b><span class="mut">dans ${dur(s.eta)}</span>`:'<b>-</b>';
  h=`<div class="kpi"><div><span class="mut">Itération</span><b>${s.iteration} / ${s.total}</b>${state}</div>
  <div><span class="mut">Étape</span><b style="font-size:16px">${esc(s.phase)}</b><span class="mut">${s.alive&&!s.finished?"depuis "+dur(s.elapsed):"(interrompu)"}</span></div>
  <div><span class="mut">Durée d'une itération</span><b>${s.avg?dur(s.avg):'-'}</b><span class="mut">moyenne des 5 dernières</span></div>
  <div><span class="mut">Fin estimée</span>${eta}</div>
  <div><span class="mut">Contre e3 (moy. 5)</span><b>${m5!=null?(m5*100).toFixed(1)+' %':'-'}</b><span class="mut">record ${best?(best[1]*100).toFixed(1)+' % (it. '+best[0]+')':'-'}</span></div></div>
  <div class="bar"><i style="width:${100*Math.min(1,(s.iteration-1)/s.total)}%"></i></div>`}
 $('status').innerHTML=h||'<p class="mut">Pas de journal pour ce run.</p>';
 const v=st.validation;$('val_t').textContent=v&&v.iteration?`Validation de l'itération ${v.iteration}`:'Dernière validation';
 $('val').innerHTML=v&&v.rows.length?`<table><tr><th>Matchup</th><th class="n">Bot</th><th class="n">Réel</th><th class="n">Écart</th></tr>`+v.rows.map(r=>{const a=Math.abs(r.gap),c=a>10?'ko':a>5?'warn':'ok';
  return `<tr><td>${esc(r.a)} <span class="mut">vs</span> ${esc(r.b)}</td><td class="n">${r.sim.toFixed(1)}</td><td class="n">${r.real.toFixed(1)}</td><td class="n ${c}">${r.gap>0?'+':''}${r.gap.toFixed(1)}</td></tr>`}).join('')+
  `</table>${v.rmse!=null?`<p class="mut">RMSE ${v.rmse.toFixed(1)} points (e3 : 11-12). Écart = bot − réel, en points de % pour le premier deck.</p>`:''}`:'<p class="mut">Pas encore de validation.</p>';
 $('hl_t').textContent=st.highlights.title||'Moments forts';
 $('hl').innerHTML=st.highlights.lines.length?'<table>'+st.highlights.lines.map(l=>`<tr><td class="mut">${esc(l[0])}</td><td>${esc(l[1])}</td></tr>`).join('')+'</table>':'<p class="mut">-</p>';
 $('tail').textContent=st.tail.join('\n');
 $('upd').textContent='mis à jour à '+new Date().toLocaleTimeString('fr-BE')+' · rafraîchi toutes les 20 s'}
function drawDecks(){const rows=[...ROWS].sort((a,b)=>SORT=='label'||SORT=='kind'?a[SORT].localeCompare(b[SORT]):b[SORT]-a[SORT]||b.games-a.games);
 $('decks').innerHTML=rows.length?rows.map((r,i)=>`<tr><td class="mut">${i+1}</td><td>${r.path?`<a href="/?deck=${encodeURIComponent(r.path)}" target="_blank">${esc(r.label)}</a>`:esc(r.label)}</td><td class="mut">${esc(r.kind)}</td><td class="n">${r.wins}</td><td class="n">${r.games}</td><td class="n"><b>${(100*r.rate).toFixed(0)} %</b></td></tr>`).join(''):
 `<tr><td colspan="6" class="mut">${DK.iterations?`Aucun deck avec au moins ${$('min').value} parties (${DK.hidden} en dessous) : baisse le minimum ou attends quelques itérations.`:'Pas encore de résultats : ils arrivent à la fin de l\'auto-jeu de chaque itération.'}</td></tr>`}
async function loadDecks(){const q=new URLSearchParams({run:RUN,window:$('win').value,kind:$('kind').value,min:$('min').value});
 DK=await (await fetch('/train/decks?'+q)).json();ROWS=DK.rows;
 $('dk_info').textContent=DK.iterations?`${DK.iterations} itération${DK.iterations>1?'s':''} enregistrée${DK.iterations>1?'s':''} (${DK.first} à ${DK.last}). Un deck maison joue ~8 parties par itération, une liste méta ~400 (avec ses copies). Classement par taux corrigé : 5 victoires et 5 défaites fictives en plus, pour qu'un 3 sur 3 ne passe pas devant un 60 sur 100.`:'';drawDecks()}
async function tick(){if(!RUN)return;try{const st=await (await fetch('/train/state?run='+encodeURIComponent(RUN))).json();render(st);
 if(st.stats_version!==STATSV){STATSV=st.stats_version;loadDecks()}}catch(e){$('upd').textContent='serveur injoignable, nouvel essai dans 20 s'}}
document.querySelectorAll('th[data-s]').forEach(th=>th.onclick=()=>{SORT=th.dataset.s;drawDecks()});
['win','kind','min'].forEach(id=>$(id).onchange=loadDecks);
(async()=>{const runs=await (await fetch('/train/runs')).json();for(const r of runs)$('run').append(new Option(r,r));
 const want=new URLSearchParams(location.search).get('run');RUN=want&&runs.includes(want)?want:runs[0]||'';$('run').value=RUN;
 $('run').onchange=e=>{RUN=e.target.value;STATSV=null;history.replaceState(null,'','?run='+encodeURIComponent(RUN));tick()};
 tick();setInterval(tick,20000)})();
</script></body></html>"""

HEADER = re.compile(r'^(\d\d):(\d\d):(\d\d) === itération (\d+)/(\d+)')
VALID = re.compile(r'(\S+)\s+vs\s+(\S+)\s+sim\s+([\d.]+)\s+réel\s+([\d.]+)\s+écart\s+([+-]?[\d.]+)')
TAG = re.compile(r'^\d\d:\d\d:\d\d\s+\[([A-Z]+)\]\s*(.*)$')
REAL_KINDS = ('meta', 'src', 'forge', 'solo')


def runs():
    """Training runs (folders of runs/ with a log), the most recently active first."""
    base = os.path.join(engine.ROOT, 'runs')
    if not os.path.isdir(base):
        return []
    found = [d for d in os.listdir(base) if os.path.exists(os.path.join(base, d, 'log.txt'))]
    return sorted(found, key=lambda d: -os.path.getmtime(os.path.join(base, d, 'log.txt')))


def run_state(name):
    """What the training page shows: progress, timing, curves, last validation, highlights."""
    d = os.path.join(engine.ROOT, 'runs', name)
    log_f = os.path.join(d, 'log.txt')
    lines = open(log_f, encoding='utf-8', errors='replace').read().splitlines()
    heads, restarts = [], []  # (line, second of the day, iteration, total)
    for i, l in enumerate(lines):
        m = HEADER.match(l)
        if m:
            heads.append((i, int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]), int(m[4]), int(m[5])))
        elif "reprise à l'itération" in l:
            restarts.append(i)
    durations = [(t1 - t0) % 86400 for (i0, t0, n0, _), (i1, t1, n1, _) in zip(heads, heads[1:])
                 if n1 == n0 + 1 and not any(i0 < r < i1 for r in restarts)]
    status = None
    if heads:
        i0, t0, it, total = heads[-1]
        block = lines[i0 + 1:]
        has = lambda txt: any(txt in l for l in block)
        if has('terminé. Réseau final'):
            phase = 'Terminé'
        elif not has('self-play :'):
            phase = 'Auto-jeu sur les GPU'
        elif not has('nouveau réseau'):
            phase = 'Entraînement puis évaluation'
        elif has('validation contre') and not has('écart moyen'):
            phase = 'Validation (vrais matchups)'
        else:
            phase = "Fin de l'itération"
        now = time.localtime()
        elapsed = (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec - t0) % 86400
        avg = sum(durations[-5:]) / len(durations[-5:]) if durations else None
        quiet = time.time() - os.path.getmtime(log_f)
        status = {'iteration': it, 'total': total, 'phase': phase, 'elapsed': elapsed, 'avg': avg,
                  'eta': max(0, avg * (total - it + 1) - elapsed) if avg else None, 'quiet': quiet,
                  'alive': quiet < max(1800, 1.5 * (avg or 0)), 'finished': phase == 'Terminé',
                  'selfplay_done': has('self-play :')}
    validation = {'iteration': None, 'rows': [], 'rmse': None}
    start = max((i for i, l in enumerate(lines) if 'validation contre les vrais matchups' in l), default=None)
    if start is not None:
        validation['iteration'] = max((n for i, _, n, _ in heads if i < start), default=None)
        for l in lines[start + 1:]:
            m = VALID.search(l)
            if m:
                validation['rows'].append({'a': m[1], 'b': m[2], 'sim': float(m[3]), 'real': float(m[4]), 'gap': float(m[5])})
            elif 'écart moyen' in l:
                m = re.search(r'RMSE\) : ([\d.]+)', l)
                validation['rmse'] = float(m[1]) if m else None
                break
    highlights = {'title': '', 'lines': []}
    start = max((i for i, l in enumerate(lines) if '---- Moments forts' in l), default=None)
    if start is not None:
        highlights['title'] = lines[start].split('----')[1].strip()
        for l in lines[start + 1:]:
            m = TAG.match(l)
            if not m:
                break
            highlights['lines'].append([m[1].capitalize(), m[2]])
    metrics = []
    try:
        with open(os.path.join(d, 'metrics.csv'), newline='') as f:
            for r in csv.DictReader(f):
                metrics.append({k: (float(v) if v not in ('', None) else None) for k, v in r.items()})
    except OSError:
        pass
    if status:
        catch_up(d, status['iteration'], status['selfplay_done'])
    stats_f = os.path.join(d, highlights_mod.STATS_FILE)
    return {'status': status, 'validation': validation, 'highlights': highlights, 'metrics': metrics,
            'tail': [l[:200] for l in lines[-14:]],
            'stats_version': os.path.getmtime(stats_f) if os.path.exists(stats_f) else None}


def catch_up(run_dir, current, selfplay_done):
    """Records the per-deck results of the iterations whose game logs are still there (the loop
    keeps the last ones): runs started before this page existed get their 'top decks' too."""
    games = os.path.join(run_dir, 'games')
    if not os.path.isdir(games):
        return
    by_it = collections.defaultdict(list)
    for f in os.listdir(games):
        m = re.match(r'iter_(\d+)_.*\.csv$', f)
        if m:
            by_it[int(m[1]) + 1].append(os.path.join(games, f))
    for it, paths in by_it.items():
        if it < current or (it == current and selfplay_done):  # complete logs only
            try:
                highlights_mod.record(run_dir, it, paths)
            except (OSError, ValueError):
                pass


_CACHE = {}


def cached(key, path, load):
    """load(path), recomputed only when the file changes."""
    m = os.path.getmtime(path)
    hit = _CACHE.get(key)
    if hit and hit[0] == m:
        return hit[1]
    val = load(path)
    _CACHE[key] = (m, val)
    return val


def deck_content(path):
    """A deck's cards, whatever its file name: the copies of a deck in the pool count as one."""
    return cached(('deck', path), path, lambda p: '\n'.join(sorted(
        l.strip() for l in open(p, encoding='utf-8') if l.strip() and not l.startswith('#'))))


def top_decks(name, window, kind, min_games):
    d = os.path.join(engine.ROOT, 'runs', name)
    stats_f = os.path.join(d, highlights_mod.STATS_FILE)
    if not os.path.exists(stats_f):
        return {'rows': [], 'iterations': 0, 'hidden': 0}
    stats = cached(('stats', stats_f), stats_f, lambda p: json.load(open(p, encoding='utf-8')))
    labels_f = os.path.join(d, 'decks', LABELS_FILE)
    labels = cached(('labels', labels_f), labels_f, lambda p: json.load(open(p, encoding='utf-8'))) \
        if os.path.exists(labels_f) else {}
    its = sorted(stats, key=int)
    groups = {}
    for it in its[-window:] if window else its:
        for stem, (w, n) in stats[it].items():
            p = os.path.join(d, 'decks', stem + '.txt')
            g = groups.setdefault(deck_content(p) if os.path.exists(p) else stem, [0, 0, set()])
            g[0] += w
            g[1] += n
            g[2].add(stem)
    rows, hidden = [], 0
    for w, n, stems in groups.values():
        stem = min(stems, key=lambda s: (s.startswith('focus-'), s))  # the original, not a focus copy
        label, _, k = labels.get(stem) or [stem, stem, stem.split('-', 1)[0]]
        if (kind == 'real' and k not in REAL_KINDS) or (kind == 'home' and k in REAL_KINDS):
            continue
        if n < min_games:
            hidden += 1
            continue
        path = os.path.join(d, 'decks', stem + '.txt')
        rows.append({'label': label, 'kind': KINDS.get(k, k), 'wins': w, 'games': n, 'rate': w / n,
                     'score': (w + 5) / (n + 10),  # a few games say little: pulled towards 50 %
                     'path': f'runs/{name}/decks/{stem}.txt' if os.path.exists(path) else None})
    rows.sort(key=lambda r: -r['score'])
    used = its[-window:] if window else its
    return {'rows': rows[:150], 'hidden': hidden, 'iterations': len(used),
            'first': int(used[0]) if used else None, 'last': int(used[-1]) if used else None}



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
        q = parse_qs(u.query)
        if u.path == '/':
            self.send(PAGE, 'text/html')
        elif u.path == '/cards':
            self.send(json.dumps(CARDS))
        elif u.path == '/decks':
            self.send(json.dumps(deck_files()))
        elif u.path == '/deck':
            p = q.get('path', [''])[0]
            pool = re.fullmatch(r'runs/[\w.-]+/decks/[\w.-]+\.txt', p) and '..' not in p
            if not (pool and os.path.exists(os.path.join(engine.ROOT, p))) and p not in deck_files():
                self.send_error(404)  # only the repository's deck files and the training pools
                return
            self.send(open(os.path.join(engine.ROOT, p), encoding='utf-8').read(), 'text/plain')
        elif u.path == '/train':
            self.send(TRAIN, 'text/html')
        elif u.path == '/train/runs':
            self.send(json.dumps(runs()))
        elif u.path in ('/train/state', '/train/decks'):
            name = q.get('run', [''])[0]
            if name not in runs():
                self.send_error(404)
                return
            if u.path == '/train/state':
                self.send(json.dumps(run_state(name)))
            else:
                num = lambda k, dflt: int(q.get(k, [dflt])[0])
                self.send(json.dumps(top_decks(name, num('window', 10), q.get('kind', ['all'])[0], num('min', 30))))
        else:
            self.send_error(404)

    def log_message(self, *a):
        pass


def watch_runs():
    """Records the top decks of the active runs every minute, even with no page open: the loop
    keeps only the latest iteration's game logs."""
    while True:
        for name in runs():
            try:
                if time.time() - os.path.getmtime(os.path.join(engine.ROOT, 'runs', name, 'log.txt')) < 7200:
                    run_state(name)
            except Exception:
                pass
        time.sleep(60)


def lan_ip():
    """This PC's address on the local network (no packet is sent)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        s.close()


class Server(ThreadingHTTPServer):
    allow_reuse_address = os.name != 'nt'  # on Windows it would let two servers share the port


CARDS = {}


def main():
    ap = argparse.ArgumentParser(description='Decklists avec les images officielles')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--host', default='0.0.0.0',
                    help='0.0.0.0 (défaut) : ouverte aux autres appareils du réseau local ; 127.0.0.1 : ce PC seulement')
    ap.add_argument('--train', action='store_true', help="ouvre directement la page de l'entraînement")
    ap.add_argument('--run', default='', help='run affiché par la page de l\'entraînement')
    a = ap.parse_args()
    url = f'http://localhost:{a.port}' + ('/train' if a.train or a.run else '') + (f'?run={a.run}' if a.run else '')
    try:
        srv = Server((a.host, a.port), Handler)
    except OSError:  # already open (another run.bat ui, or a training run that opened it)
        print(f'Interface déjà ouverte : {url}', flush=True)
        if not a.no_browser:
            webbrowser.open(url)
        return
    CARDS.update(cards())
    print(f'Interface : {url}  (entraînement en direct : http://localhost:{a.port}/train ; Ctrl+C pour arrêter)', flush=True)
    if a.host != '127.0.0.1':
        print(f'Depuis un autre PC ou un téléphone du réseau local : http://{lan_ip()}:{a.port}/train', flush=True)
    threading.Thread(target=watch_runs, daemon=True).start()
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
