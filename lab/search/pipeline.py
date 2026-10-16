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
import argparse, atexit, collections, json, os, random, re, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from zero.live import Live, Off, console_lang  # live progress for run.bat ui (/live); Off when used as a library (forge)
LIVE = Off()
_EXE = 'deckgym.exe' if os.name == 'nt' else 'deckgym'
BIN = os.environ.get('PZ_DECKGYM') or next(
    (p for p in (os.path.join(ROOT, 'bin', _EXE), os.path.join(ROOT, 'target', 'release', _EXE)) if os.path.exists(p)),
    os.path.join(ROOT, 'target', 'release', _EXE))
META = os.path.normpath(os.path.join(ROOT, 'lab', 'decks', 'meta'))
RES = os.environ.get('PZ_RESULTS') or os.path.join(ROOT, 'lab/results')
# Players (engine player codes). The defaults are the expectiminimax bots; pocket-zero sets
# PZ_SCREEN / PZ_SEARCH to its trained network, e.g. z0 (network policy) and z16 (16-sim MCTS).
SCREEN = os.environ.get('PZ_SCREEN', 'e2')
SEARCH = os.environ.get('PZ_SEARCH', 'e3')
FINAL = os.environ.get('PZ_FINAL', SEARCH)  # stage D (final validation)
GAUNTLET = os.path.normpath(os.path.join(RES, 'gauntlet'))  # meta decks + field decks, one folder for the simulator
_LOG = []

LANG = console_lang()
_FR = []  # (regex, {English name: French name}), built on first use


def fr_names(text):
    """`text` with the English card names in French (lab/data/names_fr.json)."""
    if not _FR:
        try:
            names = json.load(open(os.path.join(ROOT, 'lab', 'data', 'names_fr.json'), encoding='utf-8'))
        except OSError:
            names = {}
        names = {k: v for k, v in names.items() if k != v}
        alt = '|'.join(re.escape(k) for k in sorted(names, key=len, reverse=True))
        _FR.append((re.compile(rf"(?<![\w'’-])(?:{alt})(?![\w'’-])") if names else None, names))
    rx, names = _FR[0]
    return rx.sub(lambda m: names[m[0]], text) if rx else text


def log(fr, en=None):
    """One line in both logs: pipeline.log in French (French card names), pipeline.en.log in
    English; the console shows the system's language. `en` defaults to `fr` (numbers only)."""
    os.makedirs(RES, exist_ok=True)
    t = time.strftime('%H:%M:%S ')
    line = {'fr': t + fr_names(str(fr)), 'en': t + str(fr if en is None else en)}
    if not _LOG:  # opened on first use only
        _LOG.append({k: open(os.path.join(RES, f), 'a', encoding='utf-8')
                     for k, f in (('fr', 'pipeline.log'), ('en', 'pipeline.en.log'))})
    print(line[LANG], flush=True)
    for k, f in _LOG[0].items():
        f.write(line[k] + '\n'); f.flush()

# Meta share of each reference list (file names of lab/decks/meta): the current shares of
# lab/decks/meta/reference.json (lab/tools/meta_reference.py), else this fallback (B4a).
WEIGHTS = {'altaria': 7.7, 'lucario': 7.7, 'butterfree': 5.2, 'vespiquen': 4.5, 'suicune': 4.2, 'rayquaza': 3.1,
           'hydreigon': 3.1, 'hoopa': 2.9, 'blaziken': 2.9, 'magnezone': 2.7, 'altaria-greninja': 2.6, 'weezing': 2.5,
           'charizard': 2.4, 'dedenne': 4.0, 'manectric': 2.1, 'sceptile-greninja': 1.8, 'flygon': 1.5}
try:
    _REF = json.load(open(os.path.join(META, 'reference.json'), encoding='utf-8'))['archetypes']
    if _REF:
        WEIGHTS = {k: round(100 * v['share'], 2) for k, v in _REF.items() if os.path.exists(os.path.join(META, k + '.txt'))}
except (OSError, ValueError, KeyError):
    pass
POLISH = os.environ.get('PZ_POLISH', '1') != '0'  # card-level tuning of the finalists (paired games)
# final score = (1 - FIELD_SHARE) * meta (weighted) + FIELD_SHARE * field (uniform). The field is
# made of the search's own archetypes (mostly ex decks): at 50 % it rewarded Oricorio walls that
# lose to the real meta, so the real meta now dominates.
FIELD_SHARE = float(os.environ.get('PZ_FIELD_SHARE', '0.2'))
# Fewest Pokémon in a deck: the leanest real meta lists (Hoopa, Flygon) play 4.
MIN_POKEMON = int(os.environ.get('PZ_MIN_POKEMON', '4'))

# ------------------------------------------------------------------ cards
DB = json.load(open(os.path.join(ROOT, 'database.json'), encoding='utf-8'))
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
def mk(cards, energy=None): return {'e': with_energy(energy or auto_energy(cards)), 'c': list(cards)}

ENERGY = [t for t in os.environ.get('PZ_ENERGY', '').split(',') if t]  # required Energy types (--energy)

def with_energy(e):
    """The deck's Energy types with the required ones in (at most 3, the required first)."""
    return sorted(list(dict.fromkeys(ENERGY + [t for t in e if t not in ENERGY]))[:3])

def playable_lines(c):
    """Every evolution card can come into play: the card it evolves from is in the deck (any
    printing, fossils included), or, for a Stage 2, Rare Candy and the Basic of its line.
    A Luxray without Shinx and Luxio is a dead card the search used to keep."""
    names = {CARD[x]['name'] for x in c}
    for x in set(c):
        prev = CARD[x].get('evolves_from')
        if not is_poke(CARD[x]) or not prev or prev in names:
            continue
        ch = chain(x)
        if not (CARD[x]['stage'] == 2 and RARE_CANDY in c and ch and CARD[ch[0]]['name'] in names):
            return False
    return True

def valid(d, lock=(), strict=True):
    c = d['c']
    if len(c) != 20: return False
    if not 1 <= len(d['e']) <= 3 or any(t not in E for t in d['e']): return False
    names = collections.Counter(CARD[x]['name'] for x in c)
    if any(v > 2 for v in names.values()): return False
    if not any(is_basic(CARD[x]) for x in c): return False
    if not playable_lines(c): return False
    if strict and sum(is_poke(CARD[x]) for x in c) < MIN_POKEMON: return False  # screening shells may be leaner
    if OFF_META and lead(d) in meta_leads(): return False
    if strict and any(t not in d['e'] for t in ENERGY): return False
    return all(x in c for x in lock)


OFF_META = bool(os.environ.get('PZ_OFF_META'))  # run.bat search --off-meta
_META_LEADS = []


def meta_leads():
    """Both attackers of every reference meta list: with --off-meta no deck may be built around one
    (Mega Altaria ex as well as the Dragonair line it shares the list with)."""
    if not _META_LEADS:
        _META_LEADS.append({a for f in os.listdir(META) if f.endswith('.txt')
                            for a in species(parse_deck(os.path.join(META, f))).split(' + ')})
    return _META_LEADS[0]

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


def lead(d):
    """The main attacker: diversity is counted on it, otherwise one archetype with a different
    second card each time (Ninetales + Milotic, + Feebas, + Porygon...) fills the whole search."""
    return species(d).split(' + ')[0]

def write_deck(d, path):
    lines = ['Energy: ' + ', '.join(d['e'])]
    for cid, k in sorted(collections.Counter(d['c']).items()):
        lines.append(f"{k} {CARD[cid]['name']} {cid}")
    open(path, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')

def parse_deck(path):
    cards, energy = [], None
    for l in open(path, encoding='utf-8'):
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

_JUDGE = []  # the GPU judge, created on first use (None: no GPU)
_SEEDS = random.Random()  # its own generator: the stages' seeded generators stay untouched


def gpu_sims(player):
    """Simulations of a network player the GPUs can play (z8, z32...), else None: CPU engine."""
    m = re.fullmatch(r'z(\d+)', player)
    if not m or os.environ.get('PZ_NO_GPU'):
        return None
    if not _JUDGE:
        from zero import engine, gpu_eval
        judge = gpu_eval.GpuJudge(os.environ['POCKET_ZERO_WEIGHTS'], engine.info()) \
            if gpu_eval.available() and os.environ.get('POCKET_ZERO_WEIGHTS') else None
        if judge:
            atexit.register(judge.close)
            log(f'parties {player} sur GPU : {", ".join(judge.devices)}', f'{player} games on GPU: {", ".join(judge.devices)}')
        _JUDGE.append(judge)
    return int(m[1]) if _JUDGE[0] else None


def on_gpu(player): return gpu_sims(player) is not None  # z0 = 0 simulations, still on the GPU


def run_games_many(decks, per_opp, player, folder, seed=None, progress=None):
    """GPU path: every deck against every opponent of `folder`, `per_opp` games each, in one batch.
    One {opponent: (wins, games)} per deck, ties counted half. With `seed`, game k against
    opponent j gets the same seed whatever the deck (paired comparisons)."""
    sims = gpu_sims(player)
    judge, opps, n = _JUDGE[0], opponents(folder), len(decks)
    tmp = tempfile.mkdtemp(prefix='pz_search_')
    try:
        paths = []
        for i, d in enumerate(decks):
            paths.append(os.path.join(tmp, f'd{i}.txt'))
            write_deck(d, paths[-1])
        paths += [os.path.join(folder, o + '.txt') for o in opps]
        games = [(i, n + j, (seed * 7919 + j * 104_729 + k) % 2**31 if seed is not None else _SEEDS.getrandbits(31))
                 for i in range(n) for j in range(len(opps)) for k in range(per_opp)]
        res, step = [], 4096 * len(judge.devices)
        for at in range(0, len(games), step):
            res += judge.play(paths, games[at:at + step], sims)
            if progress:
                progress(len(res), len(games))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = [{} for _ in decks]
    for (i, b, _), r in zip(games, res):
        w, g = out[i].get(opps[b - n], (0, 0))
        out[i][opps[b - n]] = (w + r, g + 1)
    return out


def run_games(d, per_opp, player, folder, seed=None):
    if on_gpu(player):
        return run_games_many([d], per_opp, player, folder, seed)[0]
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f: path = f.name
    write_deck(d, path)
    n_opp = len(opponents(folder))
    cmd = [BIN, 'simulate', path, folder, '--num', str(per_opp * n_opp), '--players', f'{player},{player}', '-p',
           '-j', os.environ.get('THREADS', str(os.cpu_count() or 2))]  # no --seed: older engines replayed one game
    if seed is not None:  # paired comparisons (polish): same games for two decks
        cmd += ['--seed', str(seed)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                             timeout=3600, cwd=ROOT)
        txt = out.stdout + out.stderr
    except subprocess.TimeoutExpired:
        txt = ''
    os.unlink(path)
    per = {}
    for blk in txt.split('Simulating against deck')[1:]:
        m = re.search(r': (\S+)\.txt', blk); w = re.search(r'Player 0 won: (\d+)', blk); dr = re.search(r'Draws: (\d+)', blk)
        g = re.search(r'Running (\d+) games', blk)
        if m and w and g:
            # The engine may print the folder with the name ("meta\\altaria" on Windows): keep the file name.
            per[re.split(r'[\\/]', m.group(1))[-1]] = (int(w.group(1)) + (int(dr.group(1)) if dr else 0) / 2, int(g.group(1)))
    if not per:  # never score a deck 0 because the engine failed: say why and stop
        raise SystemExit(f'Le moteur n\'a rendu aucun résultat.\nCommande : {" ".join(cmd)}\n'
                         f'Sortie :\n{txt[-3000:] or "(vide)"}')
    return per

def check():
    """--check: one deck against the meta, with the engine's raw output and what was read."""
    d = parse_deck(os.path.join(META, 'dedenne.txt'))
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f: path = f.name
    write_deck(d, path)
    cmd = [BIN, 'simulate', path, META, '--num', str(len(opponents(META))), '--players', f'{SCREEN},{SCREEN}', '-p']
    out = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=ROOT)
    print('Commande :', ' '.join(cmd))
    print('Code de sortie :', out.returncode)
    print('--- début de la sortie ---'); print((out.stdout + out.stderr)[:2500]); print('--- fin ---')
    print('Lu :', run_games(d, 1, SCREEN, META))
    print('Score (méta) :', evaluate(d, 1, SCREEN))

def evaluate(d, per_opp, player=SCREEN, folder=None):
    """Accumulates games in the cache; returns score(d)."""
    rec = CACHE.setdefault(f'{player}:{key(d)}', {})
    for opp, (w, n) in run_games(d, per_opp, player, folder or META).items():
        a = rec.get(opp, [0, 0]); rec[opp] = [a[0] + w, a[1] + n]
    save_cache()
    return score(d, player)

def evaluate_many(decks, per_opp, player, folder=None, progress=None):
    """evaluate() for many decks: a single GPU batch when the player runs on the GPUs."""
    decks = list(decks)
    if not on_gpu(player):
        for i, d in enumerate(decks, 1):
            evaluate(d, per_opp, player, folder)
            if progress:
                progress(i, len(decks))
        return
    for d, per in zip(decks, run_games_many(decks, per_opp, player, folder or META, progress=progress)):
        rec = CACHE.setdefault(f'{player}:{key(d)}', {})
        for opp, (w, n) in per.items():
            a = rec.get(opp, [0, 0]); rec[opp] = [a[0] + w, a[1] + n]
    save_cache()

def winrates(d, player):
    rec = CACHE.get(f'{player}:{key(d)}', {})
    return {o: v[0] / v[1] for o, v in rec.items() if v[1]}, sum(v[1] for v in rec.values())

def score(d, player=SCREEN):
    """(total, games). total mixes the weighted meta winrate and the uniform field winrate."""
    s = full_score(d, player); return s['total'], s['games']

def full_score(d, player=SCREEN):
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
        nd = {'e': with_energy(e), 'c': c}
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
    nd = {'e': with_energy(a['e']), 'c': c[:20]}
    return nd if valid(nd, lock) else None

# ------------------------------------------------------------------ stages
def stage_file(s): return os.path.join(RES, f'stage_{s}.json')

def stage_A(per_opp=4):
    if os.path.exists(stage_file('A')): return json.load(open(stage_file('A')))
    finals = attacker_finals(); log(f'A: {len(finals)} lignées', f'A: {len(finals)} lines')
    res = []
    for f in finals:
        ch = chain(f)
        d = shell(line_cards(ch)) if ch else None
        if d and valid(d, strict=False):
            res.append({'final': f, 'deck': d})
    todo = [r['deck'] for r in res if score(r['deck'])[1] < per_opp * len(WEIGHTS)]
    for at in range(0, len(todo), 64):  # batches: the GPUs play many decks at once
        evaluate_many(todo[at:at + 64], per_opp, SCREEN)
        done = min(at + 64, len(todo))
        best = max(score(d)[0] for d in todo[:done])
        log(f'A {done}/{len(todo)} best={best:.3f}')
        LIVE.set(phase='screen', progress=(done, len(todo)), current={'lines': f'{done}/{len(todo)}', 'best': round(best, 4)})
    for r in res:
        r['score'] = score(r['deck'])[0]
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
        row = []
        for j in range(i + 1, len(finals)):
            c1, c2 = chain(finals[i]), chain(finals[j])
            d = shell(line_cards(c1, True) + line_cards(c2, True))
            if not d or not valid(d, strict=False): continue
            row.append({'finals': [finals[i], finals[j]], 'deck': d})
        evaluate_many([r['deck'] for r in row if score(r['deck'])[1] < per_opp * len(WEIGHTS)], per_opp, SCREEN)
        for r in row:
            r['score'] = score(r['deck'])[0]
        res += row
        log(f'B {i+1}/{len(finals)} best={max([r["score"] for r in res] or [0]):.3f}')
        LIVE.set(phase='pairs', progress=(i + 1, len(finals)), current={'best': round(max([r['score'] for r in res] or [0]), 4)})
    res.sort(key=lambda r: -r['score'])
    json.dump(res, open(stage_file('B'), 'w')); save_cache(True)
    return res

def stage_F(A, B, size=10):
    """Field = best off-meta archetypes found so far. Decks must also beat these to score well."""
    if os.path.exists(stage_file('F')):
        field = json.load(open(stage_file('F')))
    else:
        meta_leads = {lead(parse_deck(os.path.join(META, f))) for f in os.listdir(META) if f.endswith('.txt')}
        field, seen = [], set(meta_leads)
        for r in sorted(A + B, key=lambda r: -r['score']):
            sp = species(r['deck'])
            if lead(r['deck']) in seen: continue
            seen.add(lead(r['deck'])); field.append({'species': sp, 'deck': r['deck'], 'score_e2': r['score']})
            if len(field) == size: break
        json.dump(field, open(stage_file('F'), 'w'))
    shutil.rmtree(GAUNTLET, ignore_errors=True); os.makedirs(GAUNTLET)
    for f in os.listdir(META):
        if f.endswith('.txt'): shutil.copy(os.path.join(META, f), GAUNTLET)
    for i, r in enumerate(field):
        name = re.sub(r'[^a-z0-9]+', '-', r['species'].lower()).strip('-')
        write_deck(r['deck'], os.path.join(GAUNTLET, f'field-{i:02d}-{name}.txt'))
    field_txt = ' | '.join(r['species'] for r in field)
    log('F: terrain = ' + field_txt, 'F: field = ' + field_txt)
    LIVE.event('terrain : ' + ' | '.join(r['species'] for r in field))
    return field

def live_rows(P, player, final=False):
    """The live table: best deck of each archetype (the best one saved for run.bat ui)."""
    rows, seen = [], set()
    for d in P:
        sp = species(d)
        if sp in seen: continue
        seen.add(sp)
        s = full_score(d, player)
        name = f'{len(rows) + 1}-' + re.sub(r'[^a-z0-9]+', '-', sp.lower()).strip('-') if final else 'meilleur-en-cours'
        path = os.path.join(RES, 'decks', name + '.txt')
        if final or not rows:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_deck(d, path)
        rows.append({'key': sp, 'species': sp, 'total': round(s['total'], 4), 'meta': round(s['meta'], 4),
                     'field': None if s['field'] is None else round(s['field'], 4), 'games': s['games'],
                     'path': os.path.relpath(path, ROOT).replace(os.sep, '/') if final or len(rows) == 0 else None})
        if len(rows) == 8: break
    LIVE.rows(rows)

def select(cands, pop, per_species=2):
    """Best decks first, but at most per_species of each archetype, so the search keeps several ideas alive."""
    out, cnt = [], collections.Counter()
    for d in sorted(cands, key=lambda x: -score(x, SEARCH)[0]):
        sp = lead(d)
        if cnt[sp] < per_species: out.append(d); cnt[sp] += 1
        if len(out) == pop: break
    return out

def stage_C(seeds, immigrants_pool, gens=25, pop=16, kids=3, per_opp=4, lock=(), tag='C'):
    f = stage_file(tag)
    state = json.load(open(f)) if os.path.exists(f) else {'gen': 0, 'pop': seeds}
    P = [p for p in state['pop'] if valid(p, lock)]
    if not P:
        raise SystemExit(f'{tag} : aucun deck de départ valide' + (' avec les cartes imposées' if lock else '') +
                         ' (trop peu de Pokémon, ou cartes incompatibles).\n'
                         f'{tag}: no valid starting deck' + (' with the required cards' if lock else '') +
                         ' (too few Pokémon, or incompatible cards).')

    def measure(decks, n, what):  # a generation is long: show that it moves
        shown = [-1]

        def progress(i, t):
            LIVE.set(progress=(i, t))
            if i == t or 10 * i // t > shown[0]:
                shown[0] = 10 * i // t
                log(f'{tag} {what[0]} : {100 * i // t} %', f'{tag} {what[1]}: {100 * i // t} %')
        evaluate_many(decks, n, SEARCH, GAUNTLET, progress)
    if state['gen'] == 0:
        measure(P, per_opp, ('départ', 'start'))
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
        measure(children, per_opp, (f'gen {g+1}/{gens}',) * 2)
        evaluate_many(P, max(1, per_opp // 2), SEARCH, GAUNTLET)  # elites accumulate games -> less noise
        P = select({key(x): x for x in P + children}.values(), pop)
        b = P[0]; s = full_score(b, SEARCH)
        tail = f'[{species(b)}] {",".join(b["e"])} | {short(b)}'
        log(f'{tag} gen {g+1}/{gens} best={s["total"]:.3f} (méta {s["meta"]:.3f}, terrain {s["field"] or 0:.3f}, '
            f'{s["games"]} parties) {tail}',
            f'{tag} gen {g+1}/{gens} best={s["total"]:.3f} (meta {s["meta"]:.3f}, field {s["field"] or 0:.3f}, '
            f'{s["games"]} games) {tail}')
        alive = ' | '.join(sorted({species(x) for x in P}))
        log(f'{tag}   archétypes vivants: ' + alive, f'{tag}   archetypes alive: ' + alive)
        live_rows(P, SEARCH)
        LIVE.set(True, phase='genetic', progress=(g + 1, gens),
                 current={'gen': f'{g + 1}/{gens}', 'best': round(s['total'], 4), 'meta': round(s['meta'], 4),
                          'field': round(s['field'] or 0, 4), 'archetypes': len({species(x) for x in P})})
        LIVE.event(f'génération {g + 1}/{gens} : {species(b)} {100 * s["total"]:.1f} %')
        json.dump({'gen': g + 1, 'pop': P}, open(f, 'w')); save_cache(True)
    return P

def score_of(per):
    """full_score's total from one run's results {opponent: (wins, games)}."""
    wr = {o: w / n for o, (w, n) in per.items() if n}
    meta = [(WEIGHTS[o], v) for o, v in wr.items() if o in WEIGHTS]
    field = [v for o, v in wr.items() if o.startswith('field-')]
    m = sum(w * v for w, v in meta) / sum(w for w, _ in meta) if meta else 0.0
    return m if not field else (1 - FIELD_SHARE) * m + FIELD_SHARE * sum(field) / len(field)

def polish(d, lock=(), per_opp=6, n_alts=8, margin=0.01):
    """Card-level tuning. A one-card change moves the win rate by ~1 %, far below the noise of
    a few hundred games, so the search leaves filler cards to chance (a Potion where Lucky Ice
    Pop does better). Here the deck and each variant play the SAME seeded games (same shuffles,
    same coin flips): only the card's effect differs. Each trainer card is tried against the
    trainers most played in the real meta lists; the gains are combined, then confirmed on new
    seeds."""
    usage = collections.Counter()
    for f in os.listdir(META):
        if f.endswith('.txt'):
            for c in set(parse_deck(os.path.join(META, f))['c']):
                if CARD[c]['kind'] == 'Trainer' and not is_fossil(CARD[c]): usage[CARD[c]['name']] += 1
    alts = [BY_NAME[n][0] for n, _ in usage.most_common(n_alts) if BY_NAME.get(n)]
    seed = random.randrange(1 << 30)
    base = score_of(run_games(d, per_opp, SEARCH, GAUNTLET, seed))
    gains = []
    for out in sorted({c for c in d['c'] if CARD[c]['kind'] == 'Trainer' and c not in lock}):
        for alt in alts:
            if CARD[alt]['name'] == CARD[out]['name']: continue
            cards = list(d['c']); cards.remove(out); cards.append(alt)
            v = mk(cards, d['e'])
            if not valid(v, lock): continue
            g = score_of(run_games(v, per_opp, SEARCH, GAUNTLET, seed)) - base
            if g > margin: gains.append((g, out, alt))
    best, used = d, set()
    for g, out, alt in sorted(gains, reverse=True):
        if out in used or alt in used: continue
        cards = list(best['c']); cards.remove(out); cards.append(alt)
        v = mk(cards, best['e'])
        if valid(v, lock): best = v; used |= {out, alt}; log(f'  polissage : {CARD[out]["name"]} -> {CARD[alt]["name"]} (+{100 * g:.1f} pts)', f'  polish: {CARD[out]["name"]} -> {CARD[alt]["name"]} (+{100 * g:.1f} pts)')
    if best is d: return d
    seed = random.randrange(1 << 30)  # confirm the combination on games it was not chosen on
    a = score_of(run_games(d, per_opp * 2, SEARCH, GAUNTLET, seed))
    b = score_of(run_games(best, per_opp * 2, SEARCH, GAUNTLET, seed))
    log(f'  polissage : confirmation {100 * a:.1f} % -> {100 * b:.1f} %' + ('' if b > a else ' (refusé)'),
        f'  polish: confirmation {100 * a:.1f} % -> {100 * b:.1f} %' + ('' if b > a else ' (rejected)'))
    return best if b > a else d

def stage_D(P, top=6, per_opp=30, tag='D', lock=()):
    out, seen = [], set()
    for d in P:  # best deck of each archetype
        if lead(d) in seen: continue
        seen.add(lead(d))
        if POLISH:
            log(f'{tag}: polissage de {species(d)}', f'{tag}: polishing {species(d)}')
            d = polish(d, lock)
        evaluate(d, per_opp, FINAL, GAUNTLET)
        out.append({'deck': d, **full_score(d, FINAL)})
        log(f'{tag}: {species(d)} -> {out[-1]["total"]:.3f} ({out[-1]["games"]} parties)', f'{tag}: {species(d)} -> {out[-1]["total"]:.3f} ({out[-1]["games"]} games)')
        LIVE.set(True, phase='final', progress=(len(out), top), current={'deck': species(d), 'best': round(max(r['total'] for r in out), 4)})
        LIVE.event(f'validation : {species(d)} {100 * out[-1]["total"]:.1f} %')
        if len(out) == top: break
    out.sort(key=lambda r: -r['total'])
    live_rows([r['deck'] for r in out], FINAL, final=True)
    LIVE.set(True, phase='refs', current={})
    refs = {}
    ref_decks = {name: parse_deck(os.path.join(META, name + '.txt'))  # every meta deck, same bot, same gauntlet
                 for name in sorted(WEIGHTS, key=lambda n: -WEIGHTS[n])}
    evaluate_many(ref_decks.values(), per_opp // 3, FINAL, GAUNTLET)
    for name, ref in ref_decks.items():
        refs[name] = {k: v for k, v in full_score(ref, FINAL).items() if k != 'per_opp'}
        log(f'{tag}: réf {name} -> {refs[name]["total"]:.3f}', f'{tag}: ref {name} -> {refs[name]["total"]:.3f}')
    json.dump({'top': out, 'refs': refs}, open(stage_file(tag), 'w'))
    report(out, refs, tag)
    return out

def pct(x): return '—' if x is None else f'{x*100:.1f} %'

def report(out, refs, tag):
    L = [f'# Résultats ({tag}) — {time.strftime("%Y-%m-%d %H:%M")}', '',
         f'Joueur {FINAL} des deux côtés. Score = '
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
    open(os.path.join(RES, f'RESULTATS_{tag}.md'), 'w', encoding='utf-8').write('\n'.join(L))
    log(f'rapport écrit : RESULTATS_{tag}.md', f'report written: RESULTATS_{tag}.md')

def lock_seeds(pool, lock, n=24):
    """Starting decks holding the imposed cards: the screened decks that already do, then the
    imposed cards' lines grafted onto the Pokémon of the best screened decks (a lone Basic such as
    Pinsir makes no valid deck on its own: too few Pokémon)."""
    out, seen = [], set()
    def add(d):
        d = d and {'e': with_energy(d['e']), 'c': d['c']}
        if d and valid(d, lock) and key(d) not in seen: seen.add(key(d)); out.append(d)
    for d in pool:
        if all(c in d['c'] for c in lock): add(d)
    graft = []
    for l in lock:
        ch = chain(l) or [l]
        graft += [c for c in line_cards(ch, True) if c not in graft or graft.count(c) < 2]
    for d in pool:
        if len(out) >= n: break
        partner = [c for c in d['c'] if is_poke(CARD[c]) and CARD[c]['name'] not in {CARD[g]['name'] for g in graft}]
        add(shell(graft + partner))
    return out[:n]

def main(a, lock):
    t0 = time.time()
    A = stage_A(); top = ', '.join(CARD[r['final']]['name'] for r in A[:8])
    log(f'A terminé ({(time.time()-t0)/60:.0f} min) top : ' + top, f'A done ({(time.time()-t0)/60:.0f} min) top: ' + top)
    B = stage_B(A); top = ' | '.join(' + '.join(CARD[f]['name'] for f in r['finals']) for r in B[:5])
    log(f'B terminé ({(time.time()-t0)/60:.0f} min) top : ' + top, f'B done ({(time.time()-t0)/60:.0f} min) top: ' + top)
    stage_F(A, B)
    pool = [r['deck'] for r in B + A]
    seeds, seen = [], collections.Counter()
    for d in sorted(pool, key=lambda d: -score(d)[0]):  # 24 seeds, at most 2 per archetype
        if seen[lead(d)] < 2: seeds.append(d); seen[lead(d)] += 1
        if len(seeds) == 24: break
    immigrants = [r['deck'] for r in A[:150]]
    tag = 'C' if not lock else 'C_lock'
    if lock:
        seeds = lock_seeds(pool, lock)
        immigrants = [d for d in immigrants if all(c in d['c'] for c in lock)] or seeds
    if ENERGY:  # the screened decks predate the required Energy: give it to them
        seeds, immigrants = ([{'e': with_energy(d['e']), 'c': d['c']} for d in x] for x in (seeds, immigrants))
    P = stage_C(seeds, immigrants, gens=a.gens, lock=lock, tag=tag)
    stage_D(P, tag='D' if not lock else 'D_lock', lock=lock)
    log(f'FINI en {(time.time()-t0)/3600:.1f} h', f'DONE in {(time.time()-t0)/3600:.1f} h')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lock', default='', help='cartes imposées, ex "B1a 42,B1a 62"')
    ap.add_argument('--gens', type=int, default=25)
    ap.add_argument('--energy', default=os.environ.get('PZ_ENERGY', ''),
                    help='types d\'énergie imposés, ex. "Grass" ou "Grass,Fighting" (défaut : ceux des Pokémon imposés)')
    ap.add_argument('--check', action='store_true', help='diagnostic : un deck contre la méta, sortie brute du moteur')
    a = ap.parse_args()
    if a.check:
        check()
        raise SystemExit(0)
    lock = tuple(f'{x.split()[0]} {int(x.split()[1]):03d}' for x in a.lock.split(',') if x.strip())
    lock = tuple(CANON.get(sig(CARD[c]), c) for c in lock)
    if a.energy:
        ENERGY[:] = [next((t for t in E if t.lower() == x.strip().lower()), None) for x in a.energy.split(',') if x.strip()]
        if None in ENERGY or len(ENERGY) > 3:
            raise SystemExit(f'--energy : 1 à 3 types parmi / 1 to 3 types among {", ".join(E)}')
    elif lock:  # by default, the Energy the imposed Pokémon's attacks need
        ENERGY[:] = list(dict.fromkeys(t for c in lock if is_poke(CARD[c])
                                       for at in CARD[c].get('attacks') or [] for t in at['energy_required'] if t in E))[:3]
    if ENERGY:
        log('énergie imposée : ' + ', '.join(ENERGY), 'required Energy: ' + ', '.join(ENERGY))
    LIVE = Live('search', os.path.basename(os.path.normpath(RES)), screen=SCREEN, search=SEARCH, final=FINAL,
                gens=a.gens, lock=a.lock, energy=','.join(ENERGY))
    try:
        main(a, lock)
    except BaseException as e:
        LIVE.event(f'arrêt : {type(e).__name__} {e}')
        LIVE.done('stopped' if isinstance(e, KeyboardInterrupt) else 'failed')
        raise
    LIVE.done()
