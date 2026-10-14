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
from .decks import LABELS_FILE

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
<select id="lang"><option value="fr" data-i18n="img_fr"></option><option value="en" data-i18n="img_en"></option></select>
<button onclick="copyText()" data-i18n="copy"></button><button onclick="window.print()" data-i18n="print"></button>
<a href="/train" style="margin-left:auto;color:var(--acc)" data-i18n="to_train"></a>
<a href="/live" style="color:var(--acc)" data-i18n="to_live"></a>
<button id="langbtn" onclick="toggleLang()" title="Langue / Language"></button></header>
<main><div id="left"><textarea id="txt" spellcheck="false" data-i18n-ph="paste_ph"></textarea>
<p style="color:var(--mut);font-size:13px" data-i18n="edit_hint"></p></div>
<div><div id="checks"></div><p id="energy"></p><div id="grid"></div></div></main>
<div id="zoom" onclick="zoomClose()"><img alt=""></div>
<script>
/*I18N*/
const TITLE_KEY='t_decks';
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
 $('checks').innerHTML=chk(total==20,t('cards_n',total))+chk(max<=2,t('max2'))+chk(basic>0,t('basic'))+(unknown.length?chk(false,t('unknown')+unknown.join(', ')):'');
 $('energy').textContent=energy?t('energy')+energy:''}
function imgPref(){try{return localStorage.getItem('pz_img')}catch(e){return null}}
window.onLang=()=>{if(!imgPref())$('lang').value=LANG;render()};
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
 applyI18n();$('lang').value=imgPref()||LANG;
 $('decks').onchange=e=>load(e.target.value);$('txt').oninput=render;
 $('lang').onchange=()=>{try{localStorage.setItem('pz_img',$('lang').value)}catch(e){}render()};
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
<header><h1 data-i18n="h_train"></h1><select id="run"></select>
<span class="mut" id="upd"></span><a href="/" style="margin-left:auto" data-i18n="back"></a>
<a href="/live" data-i18n="to_live"></a>
<button id="langbtn" onclick="toggleLang()" title="Langue / Language"></button></header>
<main>
<div class="box wide" id="status"></div>
<div class="box"><h2 data-i18n="force"></h2><div id="c_force"></div>
<div class="leg"><span><i style="background:var(--acc)"></i><em data-i18n="leg_e3avg"></em></span><span><i style="background:var(--acc2)"></i><em data-i18n="leg_e2"></em></span><span><i style="background:var(--mut)"></i><em data-i18n="leg_e3raw"></em></span></div></div>
<div class="box"><h2 data-i18n="rmse_t"></h2><div id="c_rmse"></div>
<div class="leg"><span><i style="background:var(--acc)"></i>RMSE</span><span><i style="background:var(--warn)"></i><em data-i18n="leg_ref"></em></span></div></div>
<div class="box"><h2 data-i18n="loss_t"></h2><div id="c_loss"></div>
<div class="leg"><span><i style="background:var(--acc)"></i><em data-i18n="leg_pol"></em></span><span><i style="background:var(--acc2)"></i><em data-i18n="leg_val"></em></span></div></div>
<div class="box"><h2 id="val_t"></h2><div id="val"></div></div>
<div class="box wide"><h2 data-i18n="top_t"></h2>
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">
<select id="win"><option value="1" data-i18n="win1"></option><option value="10" selected data-i18n="win10"></option><option value="0" data-i18n="win0"></option></select>
<select id="kind"><option value="all" data-i18n="k_all"></option><option value="home" data-i18n="k_home"></option><option value="real" data-i18n="k_real"></option></select>
<select id="min"><option value="5"></option><option value="10" selected></option><option value="30"></option><option value="100"></option></select></div>
<p class="mut" id="dk_info"></p>
<div class="scroll"><table><thead><tr><th>#</th><th data-s="label">Deck</th><th data-s="kind">Type</th><th class="n" data-s="wins" data-i18n="th_wins"></th><th class="n" data-s="games" data-i18n="th_games"></th><th class="n" data-s="rate" data-i18n="th_rate"></th></tr></thead><tbody id="decks"></tbody></table></div>
<p class="mut" data-i18n-html="top_note"></p></div>
<div class="box"><h2 id="hl_t"></h2><div id="hl"></div></div>
<div class="box"><h2 data-i18n="log_t"></h2><pre id="tail"></pre></div>
</main>
<script>
/*I18N*/
const TITLE_KEY='t_train';
const $=id=>document.getElementById(id);let RUN='',STATSV=null,SORT='score',ROWS=[],DK={},LAST=null;
const esc=t=>String(t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const loc=()=>LANG=='fr'?'fr-BE':'en-GB';
function dur(s){s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.round(s%3600/60);return h?`${h} h ${String(m).padStart(2,'0')}`:`${m} min`}
function chart(series,{lo,hi,pct,ref}={}){const W=640,H=200,L=38,R=8,T=8,B=22;const pts=series.flatMap(s=>s.pts);if(!pts.length)return `<p class="mut">${t('no_data')}</p>`;
 const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]).concat(ref!=null?[ref]:[]);const x0=Math.min(...xs),x1=Math.max(...xs,x0+1);
 let y0=lo??Math.min(...ys),y1=hi??Math.max(...ys);if(y1-y0<1e-9){y0-=1;y1+=1}const pad=(y1-y0)*.08;if(lo==null)y0-=pad;if(hi==null)y1+=pad;
 const X=x=>L+(x-x0)/(x1-x0)*(W-L-R),Y=y=>T+(1-(y-y0)/(y1-y0))*(H-T-B);let g='';
 for(let k=0;k<=4;k++){const v=y0+(y1-y0)*k/4,y=Y(v);g+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="var(--grid)"/><text x="${L-4}" y="${y+4}" text-anchor="end">${pct?Math.round(v*100)+'%':v.toFixed(v<2?2:1)}</text>`}
 g+=`<text x="${L}" y="${H-6}">${x0}</text><text x="${W-R}" y="${H-6}" text-anchor="end">${t('it_axis',x1)}</text>`;
 if(ref!=null)g+=`<line x1="${L}" x2="${W-R}" y1="${Y(ref)}" y2="${Y(ref)}" stroke="var(--warn)" stroke-dasharray="5 4"/>`;
 for(const s of series){if(!s.pts.length)continue;const d=s.pts.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1)).join('');
  g+=`<path d="${d}" fill="none" stroke="${s.c}" stroke-width="${s.w||2}" opacity="${s.o||1}"/>`;
  if(s.dots)g+=s.pts.map(p=>`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="2.5" fill="${s.c}"/>`).join('')}
 return `<svg viewBox="0 0 ${W} ${H}">${g}</svg>`}
function avg(pts,k){return pts.map((p,i)=>{const w=pts.slice(Math.max(0,i-k+1),i+1);return [p[0],w.reduce((a,q)=>a+q[1],0)/w.length]})}
// Deck labels and log tags come from the training log (in French): kinds and tags are translated.
function kindLabel(label){return LANG=='fr'?label:label.replace(/\((méta|lignée|paire|mutant|découverte|autre source|forgé|solo)\)$/,(m,k)=>'('+t('kind_'+({'méta':'meta','lignée':'line','paire':'pair','découverte':'cover','autre source':'src','forgé':'forge'}[k]||k))+')')}
function render(st){LAST=st;const m=st.metrics;const col=k=>m.filter(r=>r[k]!=null).map(r=>[r.iteration,r[k]]);
 const e3=col('vs_e3');
 $('c_force').innerHTML=chart([{pts:e3,c:'var(--mut)',w:1,o:.6},{pts:col('vs_e2'),c:'var(--acc2)',w:1.5},{pts:avg(e3,5),c:'var(--acc)',w:2.5}],{pct:1});
 $('c_rmse').innerHTML=chart([{pts:col('rmse_limitless'),c:'var(--acc)',dots:1}],{ref:11.5,lo:0});
 $('c_loss').innerHTML=chart([{pts:col('val_policy_loss'),c:'var(--acc)'},{pts:col('val_value_loss').map(p=>[p[0],p[1]*5]),c:'var(--acc2)'}]);
 const s=st.status;let h='';
 if(s){const best=e3.length?e3.reduce((a,b)=>b[1]>a[1]?b:a):null,last5=e3.slice(-5),m5=last5.length?last5.reduce((a,p)=>a+p[1],0)/last5.length:null;
  const state=s.finished?`<span class="ok">${t('st_done')}</span>`:s.alive?`<span class="ok">${t('st_run')}</span>`:`<span class="ko">${t('st_stop')}</span> ${t('st_quiet',dur(s.quiet))}`;
  const eta=s.eta!=null&&!s.finished&&s.alive?`<b>${new Date(Date.now()+s.eta*1000).toLocaleString(loc(),{weekday:'short',hour:'2-digit',minute:'2-digit'})}</b><span class="mut">${t('in_t',dur(s.eta))}</span>`:'<b>-</b>';
  h=`<div class="kpi"><div><span class="mut">${t('k_it')}</span><b>${s.iteration} / ${s.total}</b>${state}</div>
  <div><span class="mut">${t('k_step')}</span><b style="font-size:16px">${t('ph_'+s.phase)}</b><span class="mut">${s.alive&&!s.finished?t('since',dur(s.elapsed)):t('interrupted')}</span></div>
  <div><span class="mut">${t('k_dur')}</span><b>${s.avg?dur(s.avg):'-'}</b><span class="mut">${t('k_dur2')}</span></div>
  <div><span class="mut">${t('k_eta')}</span>${eta}</div>
  <div><span class="mut">${t('k_e3')}</span><b>${m5!=null?(m5*100).toFixed(1)+' %':'-'}</b><span class="mut">${t('k_best')} ${best?(best[1]*100).toFixed(1)+' % (it. '+best[0]+')':'-'}</span></div></div>
  <div class="bar"><i style="width:${100*Math.min(1,(s.iteration-1)/s.total)}%"></i></div>`}
 $('status').innerHTML=h||`<p class="mut">${t('no_log')}</p>`;
 const v=st.validation;$('val_t').textContent=v&&v.iteration?t('val_it',v.iteration):t('val_t');
 $('val').innerHTML=v&&v.rows.length?`<table><tr><th>Matchup</th><th class="n">Bot</th><th class="n">${t('real')}</th><th class="n">${t('gap')}</th></tr>`+v.rows.map(r=>{const a=Math.abs(r.gap),c=a>10?'ko':a>5?'warn':'ok';
  return `<tr><td>${esc(r.a)} <span class="mut">vs</span> ${esc(r.b)}</td><td class="n">${r.sim.toFixed(1)}</td><td class="n">${r.real.toFixed(1)}</td><td class="n ${c}">${r.gap>0?'+':''}${r.gap.toFixed(1)}</td></tr>`}).join('')+
  `</table>${v.rmse!=null?`<p class="mut">${t('rmse_note',v.rmse.toFixed(1))}</p>`:''}`:`<p class="mut">${t('no_val')}</p>`;
 const hl=st.highlights;$('hl_t').textContent=hl.iteration?t('hl_it',hl.iteration):t('hl_t');
 $('hl').innerHTML=hl.lines.length?(LANG=='fr'?'':`<p class="mut">${t('hl_note')}</p>`)+'<table>'+hl.lines.map(l=>`<tr><td class="mut">${esc(T[LANG]['tag_'+l[0].toLowerCase()]||l[0])}</td><td>${esc(l[1])}</td></tr>`).join('')+'</table>':'<p class="mut">-</p>';
 $('tail').textContent=st.tail.join('\n');
 $('upd').textContent=t('upd',new Date().toLocaleTimeString(loc()))}
function drawDecks(){const rows=[...ROWS].sort((a,b)=>SORT=='label'||SORT=='kind'?a[SORT].localeCompare(b[SORT]):b[SORT]-a[SORT]||b.games-a.games);
 $('decks').innerHTML=rows.length?rows.map((r,i)=>`<tr><td class="mut">${i+1}</td><td>${r.path?`<a href="/?deck=${encodeURIComponent(r.path)}" target="_blank">${esc(kindLabel(r.label))}</a>`:esc(kindLabel(r.label))}</td><td class="mut">${esc(T[LANG]['kind_'+r.kind]||r.kind)}</td><td class="n">${r.wins}</td><td class="n">${r.games}</td><td class="n"><b>${(100*r.rate).toFixed(0)} %</b></td></tr>`).join(''):
 `<tr><td colspan="6" class="mut">${DK.iterations?t('dk_empty_min',$('min').value,DK.hidden):t('dk_empty')}</td></tr>`;
 $('dk_info').textContent=DK.iterations?t('dk_info',DK.iterations,DK.first,DK.last):''}
async function loadDecks(){const q=new URLSearchParams({run:RUN,window:$('win').value,kind:$('kind').value,min:$('min').value});
 DK=await (await fetch('/train/decks?'+q)).json();ROWS=DK.rows;drawDecks()}
async function tick(){if(!RUN)return;try{const st=await (await fetch('/train/state?run='+encodeURIComponent(RUN))).json();render(st);
 if(st.stats_version!==STATSV){STATSV=st.stats_version;loadDecks()}}catch(e){$('upd').textContent=t('offline')}}
function minLabels(){for(const o of $('min').options)o.textContent=t('min_n',o.value)}
window.onLang=()=>{minLabels();if(LAST)render(LAST);drawDecks()};
document.querySelectorAll('th[data-s]').forEach(th=>th.onclick=()=>{SORT=th.dataset.s;drawDecks()});
['win','kind','min'].forEach(id=>$(id).onchange=loadDecks);
applyI18n();minLabels();
(async()=>{const runs=await (await fetch('/train/runs')).json();for(const r of runs)$('run').append(new Option(r,r));
 const want=new URLSearchParams(location.search).get('run');RUN=want&&runs.includes(want)?want:runs[0]||'';$('run').value=RUN;
 $('run').onchange=e=>{RUN=e.target.value;STATSV=null;history.replaceState(null,'','?run='+encodeURIComponent(RUN));tick()};
 tick();setInterval(tick,20000)})();
</script></body></html>"""

LIVE_PAGE = r"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>pocket-zero</title>
<style>
:root{--bg:#f6f6f8;--fg:#1d1d22;--card:#fff;--mut:#6b6b76;--ok:#1f8a4c;--ko:#c2352b;--warn:#b7791f;--acc:#3557c4;--grid:#8882}
@media (prefers-color-scheme:dark){:root{--bg:#16161a;--fg:#ececf1;--card:#22222a;--mut:#9a9aa6;--ok:#4cc27e;--ko:#ff7468;--warn:#f0b450;--acc:#8ea6ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif}
header{padding:12px 16px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
h1{font-size:18px;margin:0 12px 0 0}h2{font-size:15px;margin:0 0 8px}a{color:var(--acc)}
select,button{font:inherit;color:inherit;background:var(--card);border:1px solid #8884;border-radius:8px;padding:6px 10px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:14px;padding:0 16px 16px}
.box{background:var(--card);border-radius:12px;padding:12px 14px;box-shadow:0 1px 3px #0002;min-width:0}
.wide{grid-column:1/-1}.mut{color:var(--mut);font-size:13px}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.kpi div b{display:block;font-size:18px}
.bar{height:10px;background:var(--grid);border-radius:5px;overflow:hidden;margin:10px 0 4px}.bar i{display:block;height:100%;background:var(--acc);transition:width .5s}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:5px 6px;text-align:left;border-bottom:1px solid var(--grid)}
th{color:var(--mut);font-weight:600;white-space:nowrap}td.n,th.n{text-align:right;white-space:nowrap}
.ok{color:var(--ok)}.ko{color:var(--ko)}.warn{color:var(--warn)}.scroll{overflow-x:auto}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;background:var(--grid);font-size:13px;margin-right:6px}
#events div{padding:3px 0;border-bottom:1px solid var(--grid);font-size:13px}#events span{color:var(--mut);margin-right:8px;font-family:ui-monospace,monospace}
dl{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;margin:0}dt{color:var(--mut)}dd{margin:0;font-weight:600}
</style></head><body>
<header><h1 data-i18n="h_live"></h1><select id="job"></select><span class="mut" id="upd"></span>
<a href="/" style="margin-left:auto" data-i18n="back"></a><a href="/train" data-i18n="to_train"></a>
<button id="langbtn" onclick="toggleLang()" title="Langue / Language"></button></header>
<main>
<div class="box wide" id="head"></div>
<div class="box"><h2 data-i18n="now_t"></h2><div id="now"></div></div>
<div class="box"><h2 data-i18n="info_t"></h2><div id="info"></div></div>
<div class="box wide"><h2 data-i18n="rows_t"></h2><div class="scroll" id="rows"></div></div>
<div class="box wide"><h2 data-i18n="ev_t"></h2><div id="events"></div></div>
</main>
<script>
/*I18N*/
const TITLE_KEY='t_live';
const $=id=>document.getElementById(id);let JOB='',LAST=null,JOBS=[];
const esc=t=>String(t??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const loc=()=>LANG=='fr'?'fr-BE':'en-GB';
const tt=(k,f)=>T[LANG][k]??T.en[k]??f;
function dur(s){s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.round(s%3600/60);return h?`${h} h ${String(m).padStart(2,'0')}`:`${m} min`}
const pct=v=>v==null?'-':(100*v).toFixed(1)+' %';
const pts=v=>v==null?'-':(v>=0?'+':'')+(100*v).toFixed(1)+' pts';
// Results columns per tool: [field, label key, format]
const COLS={forge:[['deck','col_deck','link'],['status','col_status','status'],['changes','col_changes','n'],['last','col_last'],['score','col_score','pct'],['gain','col_gain','gain']],
 search:[['species','col_species','link'],['total','col_total','pct'],['meta','col_meta','pct'],['field','col_field','pct'],['games','col_games','n']],
 ranked:[['deck','col_deck','link'],['archetype','col_arch'],['players','col_players','n'],['win','col_win','pct'],['master','col_master','pct'],['games_master','col_gm','n']]};
function cell(r,[f,,fmt]){const v=r[f];
 if(fmt=='link')return r.path?`<a href="/?deck=${encodeURIComponent(r.path)}" target="_blank">${esc(v)}</a>`:esc(v);
 if(fmt=='pct')return pct(v);if(fmt=='n')return v??'-';
 if(fmt=='gain')return v==null?'-':`${r.proven?'<span class="ok">✓</span> ':''}${pts(v)}`;
 if(fmt=='status')return `<span class="${v=='running'?'warn':v=='waiting'?'mut':'ok'}">${esc(tt('rs_'+v,v))}</span>`;
 return esc(v)}
function stateOf(j){return j.state=='running'&&j.stale?'stale':j.state}
function render(j){LAST=j;if(!j){$('head').innerHTML=`<p class="mut">${t('no_jobs')}</p>`;['now','info','rows','events'].forEach(id=>$(id).innerHTML='');return}
 const st=stateOf(j),cls={running:'ok',done:'ok',failed:'ko',stopped:'ko',stale:'warn'}[st];
 const p=j.progress,frac=p&&p.total?Math.min(1,p.done/p.total):null,el=(j.updated||Date.now()/1000)-j.started;
 const eta=st=='running'&&frac&&frac<1&&frac>0.02?t('eta',dur(el*(1-frac)/frac)):'';
 $('head').innerHTML=`<div class="kpi"><div><span class="badge">${esc(tt('tool_'+j.tool,j.tool))}</span><b>${esc(j.name)}</b><span class="${cls}">${t('js_'+st)}</span></div>
  <div><span class="mut">${t('k_step')}</span><b>${esc(tt('phase_'+j.phase,j.phase||'-'))}</b></div>
  <div><span class="mut">${t('started_t')}</span><b>${new Date(j.started*1000).toLocaleString(loc(),{weekday:'short',hour:'2-digit',minute:'2-digit'})}</b><span class="mut">${st=='running'?t('since',dur(el)):dur(el)}</span></div>
  <div><span class="mut">${t('progress_t')}</span><b>${p?`${p.done} / ${p.total}`:'-'}</b><span class="mut">${eta}</span></div></div>
  ${frac!=null?`<div class="bar"><i style="width:${100*frac}%"></i></div>`:''}`;
 const kv=(o,pre)=>{const e=Object.entries(o||{}).filter(([,v])=>v!=null&&v!=='');return e.length?'<dl>'+e.map(([k,v])=>`<dt>${esc(tt(pre+k,k))}</dt><dd>${typeof v=='number'&&v>0&&v<1&&String(v).includes('.')?pct(v):esc(v)}</dd>`).join('')+'</dl>':'<p class="mut">-</p>'};
 $('now').innerHTML=kv(j.current,'cur_');$('info').innerHTML=kv(j.info,'info_');
 const cols=COLS[j.tool]||Object.keys((j.rows||[])[0]||{}).filter(k=>k!='key'&&k!='path').map(k=>[k,k]);
 $('rows').innerHTML=(j.rows||[]).length?`<table><tr>${cols.map(c=>`<th class="${['pct','n','gain'].includes(c[2])?'n':''}">${esc(tt(c[1],c[0]))}</th>`).join('')}</tr>`+
  j.rows.map(r=>`<tr>${cols.map(c=>`<td class="${['pct','n','gain'].includes(c[2])?'n':''}">${cell(r,c)}</td>`).join('')}</tr>`).join('')+'</table>':`<p class="mut">${t('no_rows')}</p>`;
 $('events').innerHTML=(j.events||[]).length?(LANG=='fr'?'':`<p class="mut">${t('hl_note')}</p>`)+[...j.events].reverse().map(e=>`<div><span>${esc(e[0])}</span>${esc(e[1])}</div>`).join(''):'<p class="mut">-</p>';
 $('upd').textContent=t('upd5',new Date().toLocaleTimeString(loc()))}
async function tick(){try{JOBS=await (await fetch('/live/jobs')).json();
 const sel=$('job'),cur=JOB;sel.innerHTML='';for(const j of JOBS){const o=new Option(`${tt('tool_'+j.tool,j.tool)} · ${j.name} (${t('js_'+stateOf(j))})`,j.id);sel.append(o)}
 if(!JOBS.length){JOB='';render(null);return}
 const want=new URLSearchParams(location.search).get('job');JOB=JOBS.some(j=>j.id==cur)?cur:JOBS.some(j=>j.id==want)?want:JOBS[0].id;sel.value=JOB;
 render(await (await fetch('/live/job?id='+encodeURIComponent(JOB))).json())}catch(e){$('upd').textContent=t('offline')}}
$('job').onchange=e=>{JOB=e.target.value;history.replaceState(null,'','?job='+encodeURIComponent(JOB));tick()};
window.onLang=()=>{if(JOBS.length)tick();else render(null)};
applyI18n();tick();setInterval(tick,5000);
</script></body></html>"""

# Both pages: French or English (browser language by default, the FR/EN button switches and
# remembers). Texts from the training log (highlights, journal) stay in French.
I18N = r"""const T={
fr:{t_live:'pocket-zero outils en direct',h_live:'pocket-zero · outils en direct',to_live:'Outils en direct →',
tool_forge:'Forge',tool_search:'Recherche de decks',tool_ranked:'Ranked simulé',
js_running:'● en cours',js_done:'terminé',js_failed:'● échec',js_stopped:'● arrêté',js_stale:'● plus de nouvelles',
phase_climb:'Amélioration des decks',phase_final:'Mesure finale',phase_screen:'Criblage des lignées',phase_pairs:"Paires d'attaquants",
phase_genetic:'Algorithme génétique',phase_refs:'Références méta',phase_matrix:'Matrice des matchups',phase_season:'Saison',phase_done:'Terminé',
now_t:'En ce moment',info_t:'Réglages',rows_t:'Résultats (se remplissent en direct)',ev_t:'Événements',started_t:'Lancé',progress_t:'Avancement',
eta:'fin estimée dans {0}',upd5:'mis à jour à {0} · rafraîchi toutes les 5 s',
no_jobs:"Aucun outil lancé pour l'instant. Lance run.bat forge, run.bat search ou run.bat ranked : leur avancement s'affiche ici en direct.",
no_rows:'Pas encore de résultats.',
cur_deck:'Deck',cur_step:'Étape',cur_seed:'Graine',cur_alive:'Variantes en course',cur_score:'Score actuel',cur_lines:'Lignées',cur_best:'Meilleur score',
cur_gen:'Génération',cur_meta:'Méta',cur_field:'Terrain',cur_archetypes:'Archétypes vivants',cur_games:'Parties',cur_pairs:'Paires',cur_round:'Tour de saison',cur_master:'Joueurs Master Ball',
info_climb:'Bot (amélioration)',info_final:'Bot (mesure finale)',info_opponents:'Adversaires',info_engine:'Moteur',info_decks:'Decks',info_player:'Bot',
info_players:'Joueurs',info_matches:'Parties par joueur',info_games:'Parties par paire',info_screen:'Criblage',info_search:'Génétique',info_gens:'Générations',info_lock:'Cartes imposées',
col_deck:'Deck',col_status:'État',col_changes:'Changements',col_last:'Dernier changement',col_score:'Score méta',col_gain:'Gain prouvé',col_species:'Archétype',
col_total:'Score',col_meta:'Méta',col_field:'Terrain',col_games:'Parties',col_arch:'Archétype',col_players:'Joueurs',col_win:'Victoires',col_master:'Master Ball',col_gm:'Parties pour Master',
rs_waiting:'en attente',rs_running:'en cours',rs_done:'amélioré',rs_final:'mesuré',
t_decks:'pocket-zero decks',t_train:'pocket-zero entraînement',img_fr:'Images en français',img_en:'Images en anglais',copy:'Copier le texte',print:'Imprimer / PDF',
to_train:'Entraînement en direct →',paste_ph:'Colle une decklist (format du moteur : « 2 Oricorio A3 66 »)',
edit_hint:'Modifie le texte : la liste se met à jour. Lignes « # » = commentaires (sources). Clic sur une carte : zoom (flèches, Échap).',
cards_n:'{0}/20 cartes',max2:'2 exemplaires max',basic:'Pokémon de base',unknown:'inconnues : ',energy:'Énergie : ',
h_train:'pocket-zero · entraînement',back:'← Decklists',force:'Force du réseau',leg_e3avg:'contre e3 (moyenne 5)',leg_e2:'contre e2',leg_e3raw:'contre e3 (brut)',
rmse_t:'Écart aux vrais matchups (RMSE, plus bas = mieux)',leg_ref:'référence e3 (11,5)',loss_t:'Pertes sur les parties de contrôle',leg_pol:'politique',leg_val:'valeur ×5',
val_t:'Dernière validation',val_it:"Validation de l'itération {0}",top_t:'Decks qui gagnent le plus',win1:'dernière itération',win10:'10 dernières itérations',win0:'tout le run',
k_all:'tous les decks',k_home:'maison (créés par le bot)',k_real:'vrais decks (méta, sources, forgés, devs)',min_n:'≥ {0} parties',th_wins:'Victoires',th_games:'Parties',th_rate:'Taux',
top_note:"Résultats de l'auto-jeu contre tout le pool d'entraînement (decks faibles compris, avec un peu d'exploration) : un repérage, pas une preuve. Clique un deck pour voir ses cartes ; à confirmer avec la forge (<code>--deck</code>) ou le ranked.",
hl_t:'Moments forts',hl_it:"Moments forts de l'itération {0}",hl_note:'',log_t:'Journal en direct',
k_it:'Itération',k_step:'Étape',k_dur:"Durée d'une itération",k_dur2:'moyenne des 5 dernières',k_eta:'Fin estimée',k_e3:'Contre e3 (moy. 5)',k_best:'record',
st_done:'terminé',st_run:'● en cours',st_stop:"● à l'arrêt",st_quiet:'(dernier message il y a {0})',in_t:'dans {0}',since:'depuis {0}',interrupted:'(interrompu)',
ph_selfplay:'Auto-jeu sur les GPU',ph_train:'Entraînement puis évaluation',ph_valid:'Validation (vrais matchups)',ph_end:"Fin de l'itération",ph_done:'Terminé',
no_log:'Pas de journal pour ce run.',no_data:'Pas encore de données.',no_val:'Pas encore de validation.',real:'Réel',gap:'Écart',
rmse_note:'RMSE {0} points (e3 : 11-12). Écart = bot − réel, en points de % pour le premier deck.',upd:'mis à jour à {0} · rafraîchi toutes les 20 s',
offline:'serveur injoignable, nouvel essai dans 20 s',it_axis:'itération {0}',
dk_info:"{0} itération(s) enregistrée(s) ({1} à {2}). Un deck maison joue ~8 parties par itération, une liste méta ~400 (avec ses copies). Classement par taux corrigé : 5 victoires et 5 défaites fictives en plus, pour qu'un 3 sur 3 ne passe pas devant un 60 sur 100.",
dk_empty_min:'Aucun deck avec au moins {0} parties ({1} en dessous) : baisse le minimum ou attends quelques itérations.',
dk_empty:"Pas encore de résultats : ils arrivent à la fin de l'auto-jeu de chaque itération.",
kind_meta:'méta',kind_src:'autre source',kind_forge:'forgé',kind_line:'lignée',kind_pair:'paire',kind_mutant:'mutant',kind_cover:'découverte',kind_solo:'solo',
tag_top:'Top',tag_meta:'Méta',tag_devs:'Devs',tag_pepites:'Pépites',tag_serie:'Série',tag_eclair:'Éclair',tag_exploit:'Exploit',tag_marathon:'Marathon',tag_depart:'Départ',tag_record:'Record'},
en:{t_live:'pocket-zero live tools',h_live:'pocket-zero · live tools',to_live:'Live tools →',
tool_forge:'Forge',tool_search:'Deck search',tool_ranked:'Simulated ranked',
js_running:'● running',js_done:'finished',js_failed:'● failed',js_stopped:'● stopped',js_stale:'● no news',
phase_climb:'Improving the decks',phase_final:'Final measurement',phase_screen:'Screening the lines',phase_pairs:'Attacker pairs',
phase_genetic:'Genetic algorithm',phase_refs:'Meta references',phase_matrix:'Matchup matrix',phase_season:'Season',phase_done:'Finished',
now_t:'Right now',info_t:'Settings',rows_t:'Results (filling in live)',ev_t:'Events',started_t:'Started',progress_t:'Progress',
eta:'estimated end in {0}',upd5:'updated at {0} · refreshed every 5 s',
no_jobs:'No tool running yet. Start run.bat forge, run.bat search or run.bat ranked: their progress shows here live.',
no_rows:'No results yet.',
cur_deck:'Deck',cur_step:'Step',cur_seed:'Seed',cur_alive:'Variants still in',cur_score:'Current score',cur_lines:'Lines',cur_best:'Best score',
cur_gen:'Generation',cur_meta:'Meta',cur_field:'Field',cur_archetypes:'Archetypes alive',cur_games:'Games',cur_pairs:'Pairs',cur_round:'Season round',cur_master:'Master Ball players',
info_climb:'Bot (improving)',info_final:'Bot (final measurement)',info_opponents:'Opponents',info_engine:'Engine',info_decks:'Decks',info_player:'Bot',
info_players:'Players',info_matches:'Games per player',info_games:'Games per pair',info_screen:'Screening',info_search:'Genetic',info_gens:'Generations',info_lock:'Required cards',
col_deck:'Deck',col_status:'Status',col_changes:'Changes',col_last:'Last change',col_score:'Meta score',col_gain:'Proven gain',col_species:'Archetype',
col_total:'Score',col_meta:'Meta',col_field:'Field',col_games:'Games',col_arch:'Archetype',col_players:'Players',col_win:'Wins',col_master:'Master Ball',col_gm:'Games to Master',
rs_waiting:'waiting',rs_running:'running',rs_done:'improved',rs_final:'measured',
t_decks:'pocket-zero decks',t_train:'pocket-zero training',img_fr:'French card images',img_en:'English card images',copy:'Copy text',print:'Print / PDF',
to_train:'Live training →',paste_ph:'Paste a decklist (engine format: "2 Oricorio A3 66")',
edit_hint:'Edit the text: the list updates. "#" lines are comments (sources). Click a card to zoom (arrows, Esc).',
cards_n:'{0}/20 cards',max2:'2 copies max',basic:'Basic Pokémon',unknown:'unknown: ',energy:'Energy: ',
h_train:'pocket-zero · training',back:'← Decklists',force:'Network strength',leg_e3avg:'vs e3 (5-iteration mean)',leg_e2:'vs e2',leg_e3raw:'vs e3 (raw)',
rmse_t:'Gap to real matchups (RMSE, lower is better)',leg_ref:'e3 reference (11.5)',loss_t:'Losses on held-out games',leg_pol:'policy',leg_val:'value ×5',
val_t:'Latest validation',val_it:'Validation at iteration {0}',top_t:'Decks that win the most',win1:'last iteration',win10:'last 10 iterations',win0:'whole run',
k_all:'all decks',k_home:'homemade (built by the bot)',k_real:'real decks (meta, sources, forged, devs)',min_n:'≥ {0} games',th_wins:'Wins',th_games:'Games',th_rate:'Rate',
top_note:'Self-play results against the whole training pool (weak decks included, with a little exploration): a scouting tool, not a proof. Click a deck to see its cards; confirm with the forge (<code>--deck</code>) or the ranked simulation.',
hl_t:'Highlights',hl_it:'Highlights of iteration {0}',hl_note:'Messages from the tools (in French).',log_t:'Live log',
k_it:'Iteration',k_step:'Step',k_dur:'Iteration time',k_dur2:'mean of the last 5',k_eta:'Estimated end',k_e3:'vs e3 (5-it. mean)',k_best:'best',
st_done:'finished',st_run:'● running',st_stop:'● stopped',st_quiet:'(last message {0} ago)',in_t:'in {0}',since:'for {0}',interrupted:'(interrupted)',
ph_selfplay:'Self-play on the GPUs',ph_train:'Training, then evaluation',ph_valid:'Validation (real matchups)',ph_end:'End of iteration',ph_done:'Finished',
no_log:'No log for this run.',no_data:'No data yet.',no_val:'No validation yet.',real:'Real',gap:'Gap',
rmse_note:'RMSE {0} points (e3: 11-12). Gap = bot − real, in percentage points for the first deck.',upd:'updated at {0} · refreshed every 20 s',
offline:'server unreachable, retrying in 20 s',it_axis:'iteration {0}',
dk_info:'{0} iteration(s) recorded ({1} to {2}). A homemade deck plays ~8 games per iteration, a meta list ~400 (with its copies). Ranked by a corrected rate: 5 wins and 5 losses added, so that a 3 out of 3 does not beat a 60 out of 100.',
dk_empty_min:'No deck with at least {0} games ({1} below): lower the minimum or wait a few iterations.',
dk_empty:"No results yet: they arrive at the end of each iteration's self-play.",
kind_meta:'meta',kind_src:'other source',kind_forge:'forged',kind_line:'line',kind_pair:'pair',kind_mutant:'mutant',kind_cover:'discovery',kind_solo:'solo',
tag_top:'Top',tag_meta:'Meta',tag_devs:'Devs',tag_pepites:'Gems',tag_serie:'Streak',tag_eclair:'Fastest',tag_exploit:'Upset',tag_marathon:'Marathon',tag_depart:'First player',tag_record:'Record'}};
let LANG=(()=>{try{const l=localStorage.getItem('pz_lang');if(l=='fr'||l=='en')return l}catch(e){}return (navigator.language||'en').toLowerCase().startsWith('fr')?'fr':'en'})();
function t(k,...a){let s=T[LANG][k]??T.en[k]??k;a.forEach((v,i)=>{s=s.replace('{'+i+'}',v)});return s}
function applyI18n(){document.documentElement.lang=LANG;document.title=t(TITLE_KEY);
 document.querySelectorAll('[data-i18n]').forEach(e=>{e.textContent=t(e.dataset.i18n)});
 document.querySelectorAll('[data-i18n-ph]').forEach(e=>{e.placeholder=t(e.dataset.i18nPh)});
 document.querySelectorAll('[data-i18n-html]').forEach(e=>{e.innerHTML=t(e.dataset.i18nHtml)});
 const b=document.getElementById('langbtn');if(b)b.textContent=LANG=='fr'?'🌐 English':'🌐 Français'}
function toggleLang(){LANG=LANG=='fr'?'en':'fr';try{localStorage.setItem('pz_lang',LANG)}catch(e){}applyI18n();if(window.onLang)window.onLang()}"""
PAGE = PAGE.replace('/*I18N*/', I18N)
TRAIN = TRAIN.replace('/*I18N*/', I18N)
LIVE_PAGE = LIVE_PAGE.replace('/*I18N*/', I18N)

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
            phase = 'done'
        elif not has('self-play :'):
            phase = 'selfplay'
        elif not has('nouveau réseau'):
            phase = 'train'
        elif has('validation contre') and not has('écart moyen'):
            phase = 'valid'
        else:
            phase = 'end'
        now = time.localtime()
        elapsed = (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec - t0) % 86400
        avg = sum(durations[-5:]) / len(durations[-5:]) if durations else None
        quiet = time.time() - os.path.getmtime(log_f)
        status = {'iteration': it, 'total': total, 'phase': phase, 'elapsed': elapsed, 'avg': avg,
                  'eta': max(0, avg * (total - it + 1) - elapsed) if avg else None, 'quiet': quiet,
                  'alive': quiet < max(1800, 1.5 * (avg or 0)), 'finished': phase == 'done',
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
    highlights = {'iteration': None, 'lines': []}
    start = max((i for i, l in enumerate(lines) if '---- Moments forts' in l), default=None)
    if start is not None:
        m = re.search(r"itération (\d+)", lines[start])
        highlights['iteration'] = int(m[1]) if m else None
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
        rows.append({'label': label, 'kind': k, 'wins': w, 'games': n, 'rate': w / n,
                     'score': (w + 5) / (n + 10),  # a few games say little: pulled towards 50 %
                     'path': f'runs/{name}/decks/{stem}.txt' if os.path.exists(path) else None})
    rows.sort(key=lambda r: -r['score'])
    used = its[-window:] if window else its
    return {'rows': rows[:150], 'hidden': hidden, 'iterations': len(used),
            'first': int(used[0]) if used else None, 'last': int(used[-1]) if used else None}



LIVE_DIR = os.path.join(engine.ROOT, 'results', 'live')


def live_jobs():
    """Every live record (results/live/*.json), the most recently updated first; a running one
    that has not written for 15 minutes is marked stale (crashed or closed window)."""
    out = []
    for f in glob.glob(os.path.join(LIVE_DIR, '*.json')):
        try:
            with open(f, encoding='utf-8') as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        d['id'] = os.path.basename(f)[:-5]
        d['stale'] = d.get('state') == 'running' and time.time() - d.get('updated', 0) > 900
        out.append(d)
    return sorted(out, key=lambda d: -d.get('updated', 0))


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
        elif u.path == '/live':
            self.send(LIVE_PAGE, 'text/html')
        elif u.path == '/live/jobs':
            self.send(json.dumps([{k: d.get(k) for k in ('id', 'tool', 'name', 'state', 'stale', 'phase', 'progress', 'updated')}
                                  for d in live_jobs()]))
        elif u.path == '/live/job':
            want = q.get('id', [''])[0]
            job = next((d for d in live_jobs() if d['id'] == want), None)
            if job is None:
                self.send_error(404)
                return
            self.send(json.dumps(job))
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
