"""Pocket Deck Lab — autonomous deck search over the whole card pool.

Goal: find decks that are strong against the real meta AND strong everywhere (against a field of
off-meta decks the search discovers itself), including combinations nobody would think of.

Deck legality = the game's rules only, nothing more:
  20 cards, at most 2 cards with the same name, at least 1 Basic Pokémon, 1 to 3 Energy types.
Evolution lines do not have to be complete (Torchic + Rare Candy + Blaziken is legal); the simulator
decides whether something works, not a hand-written filter.

Stages (resumable, each writes lab/results/stage_<X>.json):
  A  every attacker line in a generic shell, vs the meta          (screening, e2)
  B  pairs of the best lines from distinct archetypes, vs the meta (screening, e2)
  F  "field": the best distinct off-meta archetypes from A/B, added to the gauntlet
  C  genetic search over all cards, vs meta + field                (e3, niching, immigrants)
  D  final validation: best deck of each archetype + every meta deck as reference (e3)

Usage:
  python3 lab/search/pipeline.py                          # full run
  python3 lab/search/pipeline.py --lock "B1a 42,B1a 62"   # build around given cards
"""
import argparse, collections, json, os, random, re, shutil, subprocess, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.path.join(ROOT, 'target/release/deckgym')
META = os.path.join(ROOT, 'lab/decks/meta')
RES = os.path.join(ROOT, 'lab/results')
GAUNTLET = os.path.join(RES, 'gauntlet')  # meta decks + field decks, one folder for the simulator
os.makedirs(RES, exist_ok=True)
LOG = open(os.path.join(RES, 'pipeline.log'), 'a')

def log(*a):
    s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

# Meta share (Limitless B4a, adjusted for B4b trends). Filenames in lab/decks/meta.
WEIGHTS = {'altaria': 7.7, 'lucario': 7.7, 'butterfree': 5.2, 'vespiquen': 4.5, 'suicune': 4.2, 'rayquaza': 3.1,
           'hydreigon': 3.1, 'hoopa': 2.9, 'blaziken': 2.9, 'magnezone': 2.7, 'altaria-greninja': 2.6, 'weezing': 2.5,
           'charizard': 2.4, 'dedenne': 4.0, 'manectric': 2.1, 'sceptile-greninja': 1.8, 'flygon': 1.5}
FIELD_SHARE = 0.5  # final score = (1 - FIELD_SHARE) * meta (weighted) + FIELD_SHARE * field (uniform)

# ------------------------------------------------------------------ cards
DB = json.load(open(os.path.join(ROOT, 'database.json')))
CARD = {}
for e in DB:
    kind, c = next(iter(e.items())); c['kind'] = kind; CARD[c['id']] = c
def is_poke(c): return c['kind'] == 'Pokemon'
def is_fossil(c): return c['kind'] == 'Trainer' and c.get('trainer_card_type') == 'Fossil'
def is_basic(c): return is_poke(c) and c['stage'] == 0  # the engine needs a real Basic Pokémon (fossils don't count)
def sig(c):
    if is_poke(c):
        return ('P', c['name'], c['hp'], json.dumps(c['attacks'], sort_keys=True), json.dumps(c['ability'], sort_keys=True))
    return ('T', c['name'])
CANON = {}
for cid in sorted(CARD):
    s = sig(CARD[cid]); CANON.setdefault(s, cid)
UNIQUE = sorted(set(CANON.values()))
BY_NAME = collections.defaultdict(list)
for cid in UNIQUE: BY_NAME[CARD[cid]['name']].append(cid)
POKEMON = [cid for cid in UNIQUE if is_poke(CARD[cid])]
TRAINERS = [cid for cid in UNIQUE if CARD[cid]['kind'] == 'Trainer' and not is_fossil(CARD[cid])]
TRAINERS = list({CARD[c]['name']: c for c in TRAINERS}.values())
TID = {CARD[c]['name']: c for c in TRAINERS}
RARE_CANDY = TID.get('Rare Candy')
E = ['Grass', 'Fire', 'Water', 'Lightning', 'Psychic', 'Fighting', 'Darkness', 'Metal']  # selectable in the Energy Zone
LETTER = {'G': 'Grass', 'R': 'Fire', 'W': 'Water', 'L': 'Lightning', 'P': 'Psychic', 'F': 'Fighting', 'D': 'Darkness', 'M': 'Metal'}

def atk_value(c):
    best = 0
    for a in c.get('attacks') or []:
        d = a.get('fixed_damage') or 0
        e = a.get('effect') or ''
        m = re.search(r'(\d+) more damage', e)
        if m: d += int(m.group(1)) // 2
        if re.search(r'Asleep|Paralyzed|Confused', e): d += 30
        best = max(best, d)
    return best

def pre_evo(c):
    if not c.get('evolves_from'): return None
    cands = [x for x in BY_NAME.get(c['evolves_from'], []) if is_poke(CARD[x]) or is_fossil(CARD[x])]
    return max(cands, key=lambda x: (atk_value(CARD[x]), CARD[x].get('hp') or 0)) if cands else None

def chain(cid):
    """Basic (or fossil) -> ... -> cid, or None if the line can't be traced."""
    out = [cid]
    while CARD[out[0]].get('evolves_from'):
        p = pre_evo(CARD[out[0]])
        if p is None: return None
        out.insert(0, p)
    return out

def family(cid):
    """Names of every card in cid's evolution line (used to remove a whole line at once)."""
    return {CARD[x]['name'] for x in (chain(cid) or [cid])}

def auto_energy(cards):
    """Energy the deck's attackers actually pay for: main type, plus a 2nd one only if it matters."""
    cnt = collections.Counter()
    for cid in cards:
        c = CARD[cid]
        if not is_poke(c): continue
        w = 1 + atk_value(c) / 50
        for a in c.get('attacks') or []:
            for t in a['energy_required']:
                if t in E: cnt[t] += w
        m = re.search(r'\[(\w)\] Energy from your Energy Zone', (c.get('ability') or {}).get('effect', '') or '')
        if m and m.group(1) in LETTER: cnt[LETTER[m.group(1)]] += 3
    if not cnt:
        types = collections.Counter(CARD[c]['energy_type'] for c in cards if is_poke(CARD[c]) and CARD[c]['energy_type'] in E)
        return [types.most_common(1)[0][0]] if types else ['Psychic']
    top = cnt.most_common(2)
    return sorted([t for t, w in top if w >= 0.4 * top[0][1]])

# A deck is {'e': [energy types], 'c': [20 card ids]}.
def mk(cards, energy=None): return {'e': sorted(set(energy or auto_energy(cards))), 'c': list(cards)}

def valid(d, lock=()):
    c = d['c']
    if len(c) != 20: return False
    if not 1 <= len(d['e']) <= 3 or any(t not in E for t in d['e']): return False
    names = collections.Counter(CARD[x]['name'] for x in c)
    if any(v > 2 for v in names.values()): return False
    if not any(is_basic(CARD[x]) for x in c): return False
    return all(x in c for x in lock)

def key(d): return ','.join(d['e']) + '#' + '|'.join(sorted(d['c']))

def species(d):
    """Archetype = the (up to) two strongest attackers, by damage x copies. Used for diversity."""
    cnt = collections.Counter(d['c'])
    pre = {CARD[x].get('evolves_from') for x in cnt}  # skip Riolu when Mega Lucario ex is there
    att = collections.Counter()
    for cid, k in cnt.items():
        c = CARD[cid]
        if is_poke(c) and c['name'] not in pre: att[c['name']] = max(att[c['name']], atk_value(c) * k + 1)
    top = [n for n, _ in att.most_common(2)]
    return ' + '.join(sorted(top)) or '?'

def write_deck(d, path):
    lines = ['Energy: ' + ', '.join(d['e'])]
    for cid, k in sorted(collections.Counter(d['c']).items()):
        lines.append(f"{k} {CARD[cid]['name']} {cid}")
    open(path, 'w').write('\n'.join(lines) + '\n')

def parse_deck(path):
    cards, energy = [], None
    for l in open(path):
        l = l.strip()
        if l.lower().startswith('energy:'):
            energy = [t.strip() for t in l.split(':', 1)[1].split(',') if t.strip()]
            continue
        m = re.match(r'(\d+) .* ([A-Z][\w-]* \d+)$', l)
        if m:
            # deck files may use unpadded numbers ("B3a 20"), database.json pads them ("B3a 020")
            st, num = m.group(2).split()
            cid = f'{st} {int(num):03d}'
            cards += [CANON.get(sig(CARD[cid]), cid)] * int(m.group(1))
    return mk(cards, energy)

def describe(d):
    cnt = collections.Counter(d['c'])
    po = [f"{k} {CARD[c]['name']} ({c})" for c, k in sorted(cnt.items()) if is_poke(CARD[c]) or is_fossil(CARD[c])]
    tr = [f"{k} {CARD[c]['name']} ({c})" for c, k in sorted(cnt.items()) if not (is_poke(CARD[c]) or is_fossil(CARD[c]))]
    return po, tr

def short(d): return ', '.join(f"{k}x{CARD[c]['name']}" for c, k in collections.Counter(d['c']).most_common())

# ------------------------------------------------------------------ evaluation
CACHE_F = os.path.join(RES, 'cache.json')
CACHE = json.load(open(CACHE_F)) if os.path.exists(CACHE_F) else {}
_last_save = [time.time()]
def save_cache(force=False):
    if force or time.time() - _last_save[0] > 60:
        json.dump(CACHE, open(CACHE_F + '.tmp', 'w')); os.replace(CACHE_F + '.tmp', CACHE_F); _last_save[0] = time.time()

def opponents(folder): return sorted(f[:-4] for f in os.listdir(folder) if f.endswith('.txt'))

def run_games(d, per_opp, player, folder):
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f: path = f.name
    write_deck(d, path)
    n_opp = len(opponents(folder))
    cmd = [BIN, 'simulate', path, folder, '--num', str(per_opp * n_opp), '--players', f'{player},{player}', '-p',
           '-j', os.environ.get('THREADS', '2'), '--seed', str(random.randrange(1 << 30))]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=ROOT)
        txt = out.stdout + out.stderr
    except subprocess.TimeoutExpired:
        txt = ''
    os.unlink(path)
    per = {}
    for blk in txt.split('Simulating against deck')[1:]:
        m = re.search(r': (\S+)\.txt', blk); w = re.search(r'Player 0 won: (\d+)', blk); dr = re.search(r'Draws: (\d+)', blk)
        g = re.search(r'Running (\d+) games', blk)
        if m and w and g:
            per[m.group(1)] = (int(w.group(1)) + (int(dr.group(1)) if dr else 0) / 2, int(g.group(1)))
    return per

def evaluate(d, per_opp, player='e2', folder=None):
    """Accumulates games in the cache; returns score(d)."""
    rec = CACHE.setdefault(f'{player}:{key(d)}', {})
    for opp, (w, n) in run_games(d, per_opp, player, folder or META).items():
        a = rec.get(opp, [0, 0]); rec[opp] = [a[0] + w, a[1] + n]
    save_cache()
    return score(d, player)

def winrates(d, player):
    rec = CACHE.get(f'{player}:{key(d)}', {})
    return {o: v[0] / v[1] for o, v in rec.items() if v[1]}, sum(v[1] for v in rec.values())

def score(d, player='e2'):
    """(total, games). total mixes the weighted meta winrate and the uniform field winrate."""
    s = full_score(d, player); return s['total'], s['games']

def full_score(d, player='e2'):
    wr, n = winrates(d, player)
    meta = [(WEIGHTS[o], v) for o, v in wr.items() if o in WEIGHTS]
    field = [v for o, v in wr.items() if o.startswith('field-')]
    m = sum(w * v for w, v in meta) / sum(w for w, _ in meta) if meta else 0.0
    f = sum(field) / len(field) if field else None
    tot = m if f is None else (1 - FIELD_SHARE) * m + FIELD_SHARE * f
    worst = sorted(wr.items(), key=lambda x: x[1])[:3]
    return {'total': tot, 'meta': m, 'field': f, 'games': n, 'worst': worst, 'per_opp': wr}

# ------------------------------------------------------------------ deck builders
# Generic consistency shell for the screening stages only. Stage C replaces any of it freely.
FILL = ["Professor's Research", "Professor's Research", 'Poké Ball', 'Poké Ball', 'Copycat', 'Cyrus', 'Sabrina',
        'Giant Cape', 'X Speed', 'Pokémon Center Lady', 'Red', 'Copycat', 'Giant Cape', 'Potion', 'Giovanni', 'Leaf',
        'Lucky Ice Pop', 'Lucky Ice Pop', 'Rocky Helmet', 'Iono']

def line_cards(ch, tight=False):
    if len(ch) == 1: return [ch[0]] * 2
    if len(ch) == 2: return [ch[0]] * 2 + [ch[1]] * 2
    return [ch[0]] * 2 + [ch[1]] * (1 if tight else 2) + [ch[2]] * 2

def shell(poke):
    cards = list(poke)
    has_s2 = any(is_poke(CARD[c]) and CARD[c]['stage'] == 2 for c in cards)
    for t in (['Rare Candy', 'Rare Candy'] if has_s2 else []) + FILL:
        if len(cards) >= 20: break
        cid = TID.get(t)
        if cid and cards.count(cid) < 2: cards.append(cid)
    return mk(cards) if len(cards) == 20 else None

def attacker_finals():
    out, seen = [], set()
    for cid in POKEMON:
        c = CARD[cid]
        if atk_value(c) < 50: continue
        s = (c['name'], json.dumps(c['attacks'], sort_keys=True))
        if s in seen: continue
        seen.add(s); out.append(cid)
    return out

def line_variant(cid, rng):
    """A random way to play cid's line: full line, Rare Candy skip (no Stage 1), or a single tech copy."""
    ch = chain(cid) or [cid]
    if not is_basic(CARD[ch[0]]) and not is_fossil(CARD[ch[0]]): return []
    r = rng.random()
    if len(ch) == 3 and r < 0.35 and RARE_CANDY:
        return [ch[0]] * 2 + [ch[2]] * rng.choice([1, 2]) + [RARE_CANDY] * rng.choice([1, 2])
    if len(ch) == 1 and r < 0.3:
        return [ch[0]]
    return [x for i, x in enumerate(ch) for _ in range(rng.choice([1, 2]) if 0 < i < len(ch) - 1 else 2)]

# ------------------------------------------------------------------ genetic operators
def mutate(d, rng, lock=()):
    for _ in range(200):
        c, e = list(d['c']), list(d['e'])
        free = lambda: [i for i, x in enumerate(c) if x not in lock or c.count(x) > 1]
        for _ in range(rng.choice([1, 1, 2, 2, 3])):
            r = rng.random()
            if r < 0.40:    # swap any card for any trainer
                c[rng.choice(free())] = rng.choice(TRAINERS)
            elif r < 0.60:  # count tweak: duplicate a card already in the deck
                c[rng.choice(free())] = rng.choice(c)
            elif r < 0.80:  # bring in a whole line, from the deck's types or from anywhere
                pool = [p for p in POKEMON if CARD[p]['energy_type'] in e + ['Colorless']] if rng.random() < 0.5 else POKEMON
                for x in line_variant(rng.choice(pool), rng):
                    c[rng.choice(free())] = x
            elif r < 0.90:  # drop a whole line, refill with trainers / copies
                pk = [x for x in c if (is_poke(CARD[x]) or is_fossil(CARD[x])) and x not in lock]
                if pk:
                    fam = family(rng.choice(pk))
                    for i in [i for i, x in enumerate(c) if CARD[x]['name'] in fam and x not in lock]:
                        c[i] = rng.choice(TRAINERS) if rng.random() < 0.7 else rng.choice(c)
            else:           # energy
                op = rng.random()
                if op < 0.4: e = auto_energy(c)
                elif op < 0.6 and len(e) < 3: e = e + [rng.choice(E)]
                elif op < 0.8 and len(e) > 1: e.remove(rng.choice(e))
                else: e[rng.randrange(len(e))] = rng.choice(E)
        if rng.random() < 0.3: e = auto_energy(c)
        nd = {'e': sorted(set(e)), 'c': c}
        if valid(nd, lock) and key(nd) != key(d): return nd
    return None

def crossover(a, b, rng, lock=()):
    """Pokémon of a + trainers of b (then of a) — combines a good core with a good engine."""
    pk = [x for x in a['c'] if is_poke(CARD[x]) or is_fossil(CARD[x])]
    tr = [x for x in b['c'] if not (is_poke(CARD[x]) or is_fossil(CARD[x]))]
    tr2 = [x for x in a['c'] if not (is_poke(CARD[x]) or is_fossil(CARD[x]))]
    rng.shuffle(tr); rng.shuffle(tr2)
    c = pk + [x for x in lock if x not in pk]
    for x in tr + tr2:
        if len(c) >= 20: break
        if sum(CARD[y]['name'] == CARD[x]['name'] for y in c) < 2: c.append(x)
    nd = {'e': a['e'], 'c': c[:20]}
    return nd if valid(nd, lock) else None

# ------------------------------------------------------------------ stages
def stage_file(s): return os.path.join(RES, f'stage_{s}.json')

def stage_A(per_opp=4):
    if os.path.exists(stage_file('A')): return json.load(open(stage_file('A')))
    finals = attacker_finals(); log(f'A: {len(finals)} lignées')
    res = []
    for i, f in enumerate(finals):
        ch = chain(f)
        d = shell(line_cards(ch)) if ch else None
        if d and valid(d):
            s, n = score(d)
            if n < per_opp * len(WEIGHTS): s, n = evaluate(d, per_opp)
            res.append({'final': f, 'deck': d, 'score': s})
        if i % 50 == 0 and res: log(f'A {i}/{len(finals)} best={max(r["score"] for r in res):.3f}')
    res.sort(key=lambda r: -r['score'])
    json.dump(res, open(stage_file('A'), 'w')); save_cache(True)
    return res

def stage_B(A, top=30, per_opp=6):
    if os.path.exists(stage_file('B')): return json.load(open(stage_file('B')))
    finals, seen = [], set()
    for r in A:  # top lines from distinct Pokémon names
        n = CARD[r['final']]['name']
        if n not in seen: seen.add(n); finals.append(r['final'])
        if len(finals) == top: break
    res = []
    for i in range(len(finals)):
        for j in range(i + 1, len(finals)):
            c1, c2 = chain(finals[i]), chain(finals[j])
            d = shell(line_cards(c1, True) + line_cards(c2, True))
            if not d or not valid(d): continue
            s, n = score(d)
            if n < per_opp * len(WEIGHTS): s, n = evaluate(d, per_opp)
            res.append({'finals': [finals[i], finals[j]], 'deck': d, 'score': s})
        log(f'B {i+1}/{len(finals)} best={max([r["score"] for r in res] or [0]):.3f}')
    res.sort(key=lambda r: -r['score'])
    json.dump(res, open(stage_file('B'), 'w')); save_cache(True)
    return res

def stage_F(A, B, size=10):
    """Field = best off-meta archetypes found so far. Decks must also beat these to score well."""
    if os.path.exists(stage_file('F')):
        field = json.load(open(stage_file('F')))
    else:
        meta_species = {species(parse_deck(os.path.join(META, f))) for f in os.listdir(META) if f.endswith('.txt')}
        field, seen = [], set(meta_species)
        for r in sorted(A + B, key=lambda r: -r['score']):
            sp = species(r['deck'])
            if sp in seen: continue
            seen.add(sp); field.append({'species': sp, 'deck': r['deck'], 'score_e2': r['score']})
            if len(field) == size: break
        json.dump(field, open(stage_file('F'), 'w'))
    shutil.rmtree(GAUNTLET, ignore_errors=True); os.makedirs(GAUNTLET)
    for f in os.listdir(META):
        if f.endswith('.txt'): shutil.copy(os.path.join(META, f), GAUNTLET)
    for i, r in enumerate(field):
        name = re.sub(r'[^a-z0-9]+', '-', r['species'].lower()).strip('-')
        write_deck(r['deck'], os.path.join(GAUNTLET, f'field-{i:02d}-{name}.txt'))
    log('F: terrain = ' + ' | '.join(r['species'] for r in field))
    return field

def select(cands, pop, per_species=2):
    """Best decks first, but at most per_species of each archetype, so the search keeps several ideas alive."""
    out, cnt = [], collections.Counter()
    for d in sorted(cands, key=lambda x: -score(x, 'e3')[0]):
        sp = species(d)
        if cnt[sp] < per_species: out.append(d); cnt[sp] += 1
        if len(out) == pop: break
    return out

def stage_C(seeds, immigrants_pool, gens=25, pop=16, kids=3, per_opp=4, lock=(), tag='C'):
    f = stage_file(tag)
    state = json.load(open(f)) if os.path.exists(f) else {'gen': 0, 'pop': seeds}
    P = [p for p in state['pop'] if valid(p, lock)]
    if state['gen'] == 0:
        for p in P: evaluate(p, per_opp, 'e3', GAUNTLET)
        P = select(P, pop)
    for g in range(state['gen'], gens):
        rng = random.Random(1234 + g)
        children = []
        for p in P:
            for _ in range(kids):
                c = crossover(p, rng.choice(P), rng, lock) if rng.random() < 0.2 else mutate(p, rng, lock)
                if c: children.append(c)
        for c in rng.sample(immigrants_pool, min(2, len(immigrants_pool))):  # fresh blood from screening
            if valid(c, lock): children.append(c)
        for c in children: evaluate(c, per_opp, 'e3', GAUNTLET)
        for p in P: evaluate(p, max(1, per_opp // 2), 'e3', GAUNTLET)  # elites accumulate games -> less noise
        P = select({key(x): x for x in P + children}.values(), pop)
        b = P[0]; s = full_score(b, 'e3')
        log(f'{tag} gen {g+1}/{gens} best={s["total"]:.3f} (méta {s["meta"]:.3f}, terrain {s["field"] or 0:.3f}, '
            f'{s["games"]} parties) [{species(b)}] {",".join(b["e"])} | {short(b)}')
        log(f'{tag}   archétypes vivants: ' + ' | '.join(sorted({species(x) for x in P})))
        json.dump({'gen': g + 1, 'pop': P}, open(f, 'w')); save_cache(True)
    return P

def stage_D(P, top=6, per_opp=30, tag='D'):
    out, seen = [], set()
    for d in P:  # best deck of each archetype
        if species(d) in seen: continue
        seen.add(species(d))
        evaluate(d, per_opp, 'e3', GAUNTLET)
        out.append({'deck': d, **full_score(d, 'e3')})
        log(f'{tag}: {species(d)} -> {out[-1]["total"]:.3f} ({out[-1]["games"]} parties)')
        if len(out) == top: break
    out.sort(key=lambda r: -r['total'])
    refs = {}
    for name in sorted(WEIGHTS, key=lambda n: -WEIGHTS[n]):  # every meta deck, same bot, same gauntlet
        ref = parse_deck(os.path.join(META, name + '.txt'))
        evaluate(ref, per_opp // 3, 'e3', GAUNTLET)
        refs[name] = {k: v for k, v in full_score(ref, 'e3').items() if k != 'per_opp'}
        log(f'{tag}: réf {name} -> {refs[name]["total"]:.3f}')
    json.dump({'top': out, 'refs': refs}, open(stage_file(tag), 'w'))
    report(out, refs, tag)
    return out

def pct(x): return '—' if x is None else f'{x*100:.1f} %'

def report(out, refs, tag):
    L = [f'# Résultats ({tag}) — {time.strftime("%Y-%m-%d %H:%M")}', '',
         'Bot expectiminimax profondeur 3 des deux côtés. Score = '
         f'{int((1-FIELD_SHARE)*100)} % méta (17 listes réelles, pondérées par leur part de tournoi) + '
         f'{int(FIELD_SHARE*100)} % terrain (archétypes hors méta trouvés par la recherche, poids égal).', '',
         '## Références : les decks du méta, même bot, même adversaires', '',
         '| Deck | Score | Méta | Terrain |', '|---|---|---|---|'] + \
        [f'| {k} | {pct(v["total"])} | {pct(v["meta"])} | {pct(v["field"])} |' for k, v in sorted(refs.items(), key=lambda x: -x[1]['total'])] + ['']
    for i, r in enumerate(out, 1):
        po, tr = describe(r['deck'])
        L += [f'## #{i} — {species(r["deck"])} : {pct(r["total"])} (méta {pct(r["meta"])}, terrain {pct(r["field"])}, {r["games"]} parties)', '',
              'Énergie : ' + ', '.join(r['deck']['e']), '',
              'Pires matchups : ' + ', '.join(f'{o} {v*100:.0f} %' for o, v in r['worst']), '',
              '**Pokémon**', ''] + [f'- {x}' for x in po] + ['', '**Dresseurs**', ''] + [f'- {x}' for x in tr] + \
             ['', '| Adversaire | Victoires |', '|---|---|'] + \
             [f'| {o} | {v*100:.0f} % |' for o, v in sorted(r['per_opp'].items(), key=lambda x: (-WEIGHTS.get(x[0], 0), x[0]))] + ['']
    open(os.path.join(RES, f'RESULTATS_{tag}.md'), 'w').write('\n'.join(L))
    log(f'rapport écrit: RESULTATS_{tag}.md')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lock', default='', help='cartes imposées, ex "B1a 42,B1a 62"')
    ap.add_argument('--gens', type=int, default=25)
    a = ap.parse_args()
    lock = tuple(f'{x.split()[0]} {int(x.split()[1]):03d}' for x in a.lock.split(',') if x.strip())
    lock = tuple(CANON.get(sig(CARD[c]), c) for c in lock)
    t0 = time.time()
    A = stage_A(); log(f'A terminé ({(time.time()-t0)/60:.0f} min) top: ' + ', '.join(CARD[r['final']]['name'] for r in A[:8]))
    B = stage_B(A); log(f'B terminé ({(time.time()-t0)/60:.0f} min) top: ' + ' | '.join(' + '.join(CARD[f]['name'] for f in r['finals']) for r in B[:5]))
    stage_F(A, B)
    pool = [r['deck'] for r in B + A]
    seeds, seen = [], collections.Counter()
    for d in sorted(pool, key=lambda d: -score(d)[0]):  # 24 seeds, at most 2 per archetype
        if seen[species(d)] < 2: seeds.append(d); seen[species(d)] += 1
        if len(seeds) == 24: break
    immigrants = [r['deck'] for r in A[:150]]
    tag = 'C' if not lock else 'C_lock'
    if lock:
        seeds = [d for d in pool if all(c in d['c'] for c in lock)][:24]
        immigrants = [d for d in immigrants if all(c in d['c'] for c in lock)]
        if not seeds:
            ch = [c for l in lock for c in (chain(l) or [l])]
            base = shell(list(dict.fromkeys(ch)) * 2)
            seeds = [base] if base and valid(base, lock) else []
    P = stage_C(seeds, immigrants, gens=a.gens, lock=lock, tag=tag)
    stage_D(P, tag='D' if not lock else 'D_lock')
    log(f'FINI en {(time.time()-t0)/3600:.1f} h')
