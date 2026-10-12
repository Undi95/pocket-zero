"""Deck forge: improves decks change by change, each change proven on paired games.

    python -m zero.forge --name forge1                 # improve the meta lists (Limitless + other sources)
    python -m zero.forge --name forge1 --deck my.txt   # improve your deck(s)
    python -m zero.forge --name maison1 --new 12       # homemade: 12 archetypes from the whole card pool

Why it works where the genetic search did not:
- Paired games. A change is judged by playing the deck and its variant on the SAME seeded games
  (same shuffles, same coin flips) against the real meta, weighted by real shares (DotGG). One
  card is worth ~1-3 %, invisible in independent games (±5 %), clear in paired ones. A change
  is kept only when its gain is significant (2 standard errors), then confirmed on new seeds.
- Bot biases cancel out. The network pilots some archetypes better than others (it overrates
  Butterfree, underrates Dedenne); a deck and its variant share the archetype, so a gain
  measured between them is mostly free of that bias. Absolute scores (homemade decks against
  the meta) keep it: retrain with the forged decks (run.bat train --add-decks results\\forge\\<name>\\decks)
  so the network learns to pilot them, then --reeval.
- Smart changes. Trainers are drawn by how often real lists play them (synergy cards included),
  plus whole evolution lines, copies and energy changes.

Results: results/forge/<name>/FORGE.md and decks/*.txt (for run.bat ui and run.bat ranked).
"""
import argparse
import collections
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from . import engine


def setup(res, weights, climb, final):
    os.environ.update({'PZ_RESULTS': res, 'PZ_SCREEN': 'z0', 'PZ_SEARCH': climb, 'PZ_FINAL': final})
    if weights and os.path.exists(weights):
        os.environ['POCKET_ZERO_WEIGHTS'] = os.path.abspath(weights)
    sys.path.insert(0, os.path.join(engine.ROOT, 'lab', 'search'))
    import pipeline
    return pipeline


def population(P, folder):
    """The real meta: Limitless lists and other sources, weighted by DotGG archetype shares."""
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    files = {os.path.join('lab', 'decks', 'meta', f): f for f in os.listdir(os.path.join(engine.ROOT, 'lab', 'decks', 'meta'))
             if f.endswith('.txt')}
    src = os.path.join(engine.ROOT, 'lab', 'decks', 'sources')
    for s in sorted(os.listdir(src)):
        if os.path.isdir(os.path.join(src, s)):
            for f in sorted(os.listdir(os.path.join(src, s))):
                if f.endswith('.txt'):
                    files[os.path.join('lab', 'decks', 'sources', s, f)] = f'{s}-{f}'
    share = json.load(open(os.path.join(src, 'meta_share.json'), encoding='utf-8'))['archetypes']
    w = collections.Counter()
    for a in share.values():
        for d in a['decks']:
            w[os.path.normpath(d)] += a['share'] / len(a['decks'])
    floor = min(w.values()) / 2 if w else 1.0
    weights = {}
    for rel, name in files.items():
        shutil.copy(os.path.join(engine.ROOT, rel), os.path.join(folder, name))
        weights[name[:-4]] = w.get(os.path.normpath(rel), floor)
    return files, weights


class Judge:
    """Weighted meta scores of decks on seeded games. Every deck of a call plays the same games
    (same opponent, same seed per game): differences between them are the cards'. GPU backend
    (batched engine, all decks of a call in one job) when available, else CPU processes."""

    def __init__(self, P, folder, weights, per_opp, gpu=None):
        self.P, self.folder, self.w, self.per_opp, self.gpu = P, folder, weights, per_opp, gpu
        self.opps = sorted(f[:-4] for f in os.listdir(folder) if f.endswith('.txt'))

    def _cpu(self, d, seed, player):
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            path = f.name
        self.P.write_deck(d, path)
        jobs = min(self.per_opp, os.cpu_count() or 2)

        def one(opp):
            cmd = [self.P.BIN, 'simulate', path, os.path.join(self.folder, opp + '.txt'), '--num', str(self.per_opp),
                   '--players', f'{player},{player}', '-p', '-j', str(jobs), '--seed', str(seed)]
            out = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=engine.ROOT)
            txt = out.stdout + out.stderr
            w, dr = re.search(r'Player 0 won: (\d+)', txt), re.search(r'Draws: (\d+)', txt)
            if not w:
                raise SystemExit(f'Le moteur n\'a rendu aucun résultat.\nCommande : {" ".join(cmd)}\nSortie :\n{txt[-2000:]}')
            return int(w.group(1)) + (int(dr.group(1)) if dr else 0) / 2, self.per_opp
        try:
            with ThreadPoolExecutor(max(1, (os.cpu_count() or 2) // jobs)) as ex:
                return dict(zip(self.opps, ex.map(one, self.opps)))
        finally:
            os.unlink(path)

    def _gpu(self, decks, seed, player):
        tmp = tempfile.mkdtemp(prefix='pz_forge_')
        paths = []
        for i, d in enumerate(decks):
            paths.append(os.path.join(tmp, f'd{i}.txt'))
            self.P.write_deck(d, paths[-1])
        paths += [os.path.join(self.folder, o + '.txt') for o in self.opps]
        nd = len(decks)
        game_seed = lambda j, r: (seed * 1_000_003 + j * 10_007 + r) % (1 << 62)
        pairs = [(i, nd + j, game_seed(j, r)) for i in range(nd) for j in range(len(self.opps)) for r in range(self.per_opp)]
        res = self.gpu.play(paths, pairs, int(player.split(':')[0][1:]))
        out, k = [], 0
        for i in range(nd):
            per = {}
            for o in self.opps:
                per[o] = (sum(res[k:k + self.per_opp]), self.per_opp)
                k += self.per_opp
            out.append(per)
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)
        return out

    def weighted(self, per):
        tot = sum(self.w.get(o, 0) for o in per)
        return sum(self.w.get(o, 0) * wn / n for o, (wn, n) in per.items() if n) / tot if tot else 0.0

    def scores(self, decks, seed, player):
        """[(score, {opponent: (wins, games)})] for decks, all on the same games."""
        pers = self._gpu(decks, seed, player) if self.gpu else [self._cpu(d, seed, player) for d in decks]
        return [(self.weighted(p), p) for p in pers]


def mutations(P, d, rng, usage, lock, k):
    """k distinct variants: meta-weighted trainer swaps, copy tweaks, structural changes."""
    trainers, tw = zip(*[(t, usage.get(P.CARD[t]['name'], 0) + 0.3) for t in P.TRAINERS])
    played = [(c, usage[P.CARD[c]['name']]) for c in P.POKEMON if usage.get(P.CARD[c]['name'])]
    pokes, pw = (list(x) for x in zip(*played)) if played else ([], [])
    out, seen = [], {P.key(d)}
    for _ in range(k * 30):
        if len(out) == k:
            break
        r, c = rng.random(), list(d['c'])
        free = [i for i, x in enumerate(c) if x not in lock or c.count(x) > 1]
        if r < 0.40:
            tr = [i for i in free if P.CARD[c[i]]['kind'] == 'Trainer'] or free
            c[rng.choice(tr)] = rng.choices(trainers, tw)[0]
            v = {'e': d['e'], 'c': c}
        elif r < 0.55:
            c[rng.choice(free)] = rng.choice(c)
            v = {'e': d['e'], 'c': c}
        elif r < 0.80 and pokes:  # a Pokémon line the real lists play, over trainers first
            line = P.line_variant(rng.choices(pokes, pw)[0], rng)
            slots = sorted(free, key=lambda i: (P.CARD[c[i]]['kind'] != 'Trainer', rng.random()))
            for i, x in zip(slots, line):
                c[i] = x
            v = {'e': P.auto_energy(c) if rng.random() < 0.5 else d['e'], 'c': c}
        else:
            v = P.mutate(d, rng, lock)
        if v and P.valid(v, lock) and P.key(v) not in seen:
            seen.add(P.key(v))
            out.append(v)
    return out


def diff(P, a, b):
    ca, cb = collections.Counter(a['c']), collections.Counter(b['c'])
    out = [f"-{n} {P.CARD[c]['name']}" for c, n in (ca - cb).items()] + [f"+{n} {P.CARD[c]['name']}" for c, n in (cb - ca).items()]
    if a['e'] != b['e']:
        out.append(f"énergie {', '.join(a['e'])} -> {', '.join(b['e'])}")
    return ', '.join(out) or 'aucun changement'


def stats(ds):
    n = len(ds)
    m = sum(ds) / n
    se = (sum((x - m) ** 2 for x in ds) / (n - 1)) ** 0.5 / n ** 0.5 if n > 1 else 1.0
    return m, se, n


def climb(P, J, d, rng, usage, lock, player, steps, cands, chunks, log):
    """Each step: all variants play the same seeded games as the current deck, chunk after chunk
    (one batched job per chunk); clearly worse ones drop out, a significantly better one (2
    standard errors, at least 4 chunks) is confirmed on new seeds and becomes the deck."""
    best, changes = d, []
    for step in range(steps):
        variants = mutations(P, best, rng, usage, lock, cands)
        alive, deltas, winner = list(range(len(variants))), {i: [] for i in range(len(variants))}, None
        t0 = time.time()
        for c in range(chunks):
            if not alive:
                break
            seed = rng.randrange(1 << 30)
            sc = J.scores([best] + [variants[i] for i in alive], seed, player)
            for i, (v, _) in zip(alive, sc[1:]):
                deltas[i].append(v - sc[0][0])
            keep = []
            for i in alive:
                m, se, n = stats(deltas[i])
                if not ((n == 1 and m < -0.06) or (n >= 2 and m + 1.5 * se < 0)):
                    keep.append(i)
            alive = keep
            log(f'    étape {step + 1}, graine {c + 1}/{chunks} : {len(alive)} variantes encore en course '
                f'({(time.time() - t0) / 60:.0f} min)')
            good = [i for i in alive if stats(deltas[i])[2] >= 4 and stats(deltas[i])[0] - 2 * stats(deltas[i])[1] > 0]
            if good:
                winner = max(good, key=lambda i: stats(deltas[i])[0])
                break
        if winner is None:
            log(f'    plus aucun changement gagnant après {step + 1} étapes')
            break
        v = variants[winner]
        fresh = [J.scores([best, v], rng.randrange(1 << 30), player) for _ in range(max(2, chunks // 2))]
        m2 = sum(b[0] - a[0] for a, b in fresh) / len(fresh)
        m = stats(deltas[winner])[0]
        if m2 > 0:
            changes.append((diff(P, best, v), m, m2))
            log(f'    étape {step + 1} : {changes[-1][0]}  (+{100 * m:.1f} pts, confirmé +{100 * m2:.1f})')
            best = v
        else:
            log(f'    étape {step + 1} : {diff(P, best, v)} non confirmé ({100 * m2:+.1f}), on continue')
    return best, changes


def main():
    ap = argparse.ArgumentParser(description='Forge de decks : améliorations prouvées sur parties appariées')
    ap.add_argument('--name', default='forge')
    ap.add_argument('--weights', default=os.path.join(engine.ROOT, 'runs', 'principal', 'weights', 'latest.bin'))
    ap.add_argument('--deck', action='append', default=[], help='deck(s) à améliorer (sinon : les listes méta)')
    ap.add_argument('--new', type=int, default=0, help='mode maison : nombre d\'archétypes tirés de tout le pool')
    ap.add_argument('--top', type=int, default=10, help='listes méta à améliorer (par part de méta)')
    ap.add_argument('--climb', default='z32', help='bot pendant l\'amélioration')
    ap.add_argument('--final', default='z128', help='bot de la mesure finale')
    ap.add_argument('--per-opp', type=int, default=4, help='parties par adversaire et par graine')
    ap.add_argument('--steps', type=int, default=12, help='changements maximum par deck')
    ap.add_argument('--cands', type=int, default=30, help='variantes essayées par étape')
    ap.add_argument('--chunks', type=int, default=8, help='graines maximum par comparaison')
    ap.add_argument('--lock', default='', help='cartes imposées, ex. "B1a 42,B1a 62"')
    ap.add_argument('--reeval', action='store_true', help='refait seulement la mesure finale (après réentraînement)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--engine', default='auto', choices=['auto', 'gpu', 'cpu'], help='où jouer les parties (auto : GPU si disponible)')
    a = ap.parse_args()

    res = os.path.join(engine.ROOT, 'results', 'forge', a.name)
    os.makedirs(os.path.join(res, 'decks'), exist_ok=True)
    climb_p = engine.zspec(a.climb[1:], None) if a.climb.startswith('z') else a.climb
    final_p = engine.zspec(a.final[1:], None) if a.final.startswith('z') else a.final
    P = setup(res, a.weights, climb_p, final_p)
    log = P.log
    files, weights = population(P, os.path.join(res, 'opponents'))
    gpu = None
    if a.engine != 'cpu':
        from . import gpu_eval
        if gpu_eval.available():
            gpu = gpu_eval.GpuJudge(a.weights, engine.info())
        elif a.engine == 'gpu':
            raise SystemExit('GPU ou module moteur indisponible : relance avec --engine cpu')
    log_engine = f'{len(gpu.devices)} GPU' if gpu else 'processeur'
    J = Judge(P, os.path.join(res, 'opponents'), weights, a.per_opp, gpu)
    usage = collections.Counter()
    for rel in files:
        for c in set(P.parse_deck(os.path.join(engine.ROOT, rel))['c']):
            usage[P.CARD[c]['name']] += 1
    lock = tuple(f'{x.split()[0]} {int(x.split()[1]):03d}' for x in a.lock.split(',') if x.strip())

    if a.deck:
        starts = [(os.path.splitext(os.path.basename(f))[0], P.parse_deck(f)) for f in a.deck]
    elif a.new:
        A = P.stage_A()  # every attacker line in a generic shell, screened against the meta
        starts, seen = [], set()
        for r in A:
            d = r['deck']  # lean shells are allowed to start: the climb adds Pokémon first
            if P.species(d) in seen or not P.valid(d, lock, strict=False):
                continue
            seen.add(P.species(d))
            starts.append((P.species(d).replace(' ', '-').replace('+', ''), d))
            if len(starts) == a.new:
                break
    else:
        ranked = sorted(weights, key=lambda k: -weights[k])[:a.top]
        starts = [(k, P.parse_deck(os.path.join(res, 'opponents', k + '.txt'))) for k in ranked]

    state_f = os.path.join(res, 'state.json')
    state = json.load(open(state_f)) if os.path.exists(state_f) else {}
    rng = random.Random(a.seed)
    log(f'Forge "{a.name}" : {len(starts)} decks, amélioration {a.climb}, mesure finale {a.final}, '
        f'{len(weights)} adversaires pondérés par la méta réelle, parties sur {log_engine}')
    for name, d in starts:
        if name in state and 'best' in state[name]:
            continue
        log(f'== {name}')
        t0 = time.time()
        best, changes = climb(P, J, d, rng, usage, lock, climb_p, a.steps, a.cands, a.chunks, log)
        state[name] = {'start': d, 'best': best, 'changes': changes, 'minutes': round((time.time() - t0) / 60)}
        json.dump(state, open(state_f, 'w'))

    log('== mesure finale')
    lines = [f'# Forge « {a.name} »', '',
             f'Adversaires : {len(weights)} listes réelles (Limitless et autres sources), pondérées par la part de méta '
             f'(DotGG). Mesure finale : {a.final}, parties appariées (deck de départ et deck forgé sur les mêmes parties).',
             '', 'Le gain (« forgé - départ ») est fiable même avec les biais du bot (même archétype des deux côtés). '
             'Le score absolu garde ces biais : confirme avec `run.bat ranked --deck` puis en vrai.', '']
    rows = []
    for name, s in state.items():
        if 'best' not in s:
            continue
        if a.reeval or 'final' not in s:
            runs = [J.scores([s['start'], s['best']], rng.randrange(1 << 30), final_p) for _ in range(6)]
            m, se, _ = stats([b[0] - a_[0] for a_, b in runs])
            k = 1
            if m < 0:  # the stronger bot disagrees with the climbing bot: keep the original
                s['best'], s['changes'], k = s['start'], ([('annulés : ' + '; '.join(c[0] for c in s['changes']), 0, 0)]
                                                          if s['changes'] else []), 0
            per = collections.defaultdict(lambda: [0, 0])
            for r in runs:
                for o, (w, n) in r[k][1].items():
                    per[o][0] += w
                    per[o][1] += n
            s['final'] = {'gain': max(m, 0.0) if k else 0.0, 'se': se, 'score': sum(r[k][0] for r in runs) / len(runs),
                          'per': {o: w / n for o, (w, n) in per.items() if n}}
            json.dump(state, open(state_f, 'w'))
        P.write_deck(s['best'], os.path.join(res, 'decks', name + '.txt'))
        rows.append((s['final']['score'], name, s))
    rows.sort(key=lambda r: -r[0])
    lines += ['| Deck | Score méta (forgé) | Gain prouvé vs départ | Changements |', '|---|---|---|---|']
    for sc, name, s in rows:
        f = s['final']
        sig = '✓' if f['gain'] - 2 * f['se'] > 0 else '≈'
        lines.append(f"| {name} | {100 * sc:.1f} % | {sig} {100 * f['gain']:+.1f} ± {200 * f['se']:.1f} pts | {len(s['changes'])} |")
    for sc, name, s in rows:
        po, tr = P.describe(s['best'])
        worst = sorted(s['final']['per'].items(), key=lambda kv: kv[1])[:3]
        best3 = sorted(s['final']['per'].items(), key=lambda kv: -kv[1])[:3]
        lines += ['', f'## {name} : {100 * sc:.1f} %', '', f"Énergie : {', '.join(s['best']['e'])}", '',
                  'Changements : ' + ('; '.join(c[0] for c in s['changes']) or 'aucun (déjà optimal pour le bot)'), '',
                  'Meilleurs matchups : ' + ', '.join(f'{o} {100 * v:.0f} %' for o, v in best3),
                  'Pires matchups : ' + ', '.join(f'{o} {100 * v:.0f} %' for o, v in worst), '',
                  '**Pokémon**', ''] + [f'- {x}' for x in po] + ['', '**Dresseurs**', ''] + [f'- {x}' for x in tr]
    open(os.path.join(res, 'FORGE.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    for sc, name, s in rows:
        log(f"  {name:40} {100 * sc:5.1f} %   gain {100 * s['final']['gain']:+.1f} pts")
    log(f'Résultats : {os.path.join(res, "FORGE.md")} et {os.path.join(res, "decks")}')
    if gpu:
        gpu.close()


if __name__ == '__main__':
    main()
