"""Crash fuzz: random legal decks from the whole pool against each other.

    python3 lab/tools/fuzz.py /tmp/fuzz 400 r,r 1 30   # out_dir, pairs, players, seed, games per pair

Failing pairs are kept in out_dir with the seed to replay them."""
import importlib.util, os, random, subprocess, sys, tempfile, collections, re, time
spec=importlib.util.spec_from_file_location('p', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'search', 'pipeline.py')); p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
OUT=sys.argv[1]; N=int(sys.argv[2]); PLAYERS=sys.argv[3]; SEED=int(sys.argv[4]); GAMES=int(sys.argv[5]) if len(sys.argv)>5 else 20
rng=random.Random(SEED)
def rand_deck():
    for _ in range(1000):
        c=[]
        while len(c)<20:
            if rng.random()<0.45:
                for x in p.line_variant(rng.choice(p.POKEMON), rng): c.append(x)
            else: c.append(rng.choice(p.TRAINERS))
        c=c[:20]
        if not any(p.is_basic(p.CARD[x]) for x in c): c[rng.randrange(20)]=rng.choice([x for x in p.POKEMON if p.is_basic(p.CARD[x])])
        e=p.auto_energy(c) if rng.random()<0.7 else rng.sample(p.E, rng.randint(1,3))
        d={'e':sorted(set(e)),'c':c}
        if p.valid(d): return d
os.makedirs(OUT,exist_ok=True)
fails=0
for i in range(N):
    a,b=rand_deck(),rand_deck()
    pa,pb=f'{OUT}/a_{SEED}_{i}.txt',f'{OUT}/b_{SEED}_{i}.txt'
    p.write_deck(a,pa); p.write_deck(b,pb)
    s=rng.randrange(1<<30)
    try:
        r=subprocess.run([p.BIN,'simulate',pa,pb,'--num',str(GAMES),'--players',PLAYERS,'--seed',str(s),'-v'],capture_output=True,text=True,timeout=300)
        txt=r.stdout+r.stderr; ok=(r.returncode==0 and 'panicked' not in txt)
    except subprocess.TimeoutExpired:
        txt='TIMEOUT'; ok=False
    if ok:
        os.remove(pa); os.remove(pb)
    else:
        fails+=1
        m=re.search(r"panicked at (.*?)\n(.*?)\n",txt)
        print(f'FAIL {i} seed={s} {pa} {pb} :: {m.group(1) if m else txt[-300:]} :: {m.group(2) if m else ""}',flush=True)
print('done',N,'fails',fails)
