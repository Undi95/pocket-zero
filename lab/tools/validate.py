"""Compare simulated matchups vs real Limitless winrates for reference decks."""
import subprocess, re, sys, itertools, json
from concurrent.futures import ThreadPoolExecutor
import os
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN=os.path.join(ROOT,'target/release/deckgym'); D=os.path.join(ROOT,'lab/decks/meta/')
REAL={('butterfree','lucario'):62.4,('butterfree','altaria'):48.0,('butterfree','dedenne'):55.7,('butterfree','hoopa'):60.6,
      ('lucario','altaria'):27.2,('lucario','dedenne'):64.4,('lucario','hoopa'):57.7,
      ('altaria','dedenne'):48.4,('altaria','hoopa'):25.8,('dedenne','hoopa'):67.3}
player=sys.argv[1]; n=int(sys.argv[2])
def run(pair):
    a,b=pair
    out=subprocess.run([BIN,'simulate',D+a+'.txt',D+b+'.txt','--num',str(n),'--players',f'{player},{player}'],capture_output=True,text=True,cwd=ROOT); out=out.stdout+out.stderr
    if 'Draws' not in out: print('ECHEC',pair,out[-600:]); return pair,float('nan')
    w=[int(x) for x in re.findall(r'won: (\d+)',out)]; dr=int(re.search(r'Draws: (\d+)',out).group(1))
    return pair,(w[0]+dr/2)/n*100
with ThreadPoolExecutor(2) as ex: res=list(ex.map(run,REAL))
err=[]
for (a,b),wr in res:
    e=wr-REAL[(a,b)]; err.append(e); print(f'{a:10} vs {b:8} sim {wr:5.1f}  réel {REAL[(a,b)]:5.1f}  écart {e:+5.1f}')
print(f'[{player}] RMSE {(sum(x*x for x in err)/len(err))**.5:.1f}  biais {sum(err)/len(err):+.1f}')
