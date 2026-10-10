"""Deck pool for self-play: the bot has to learn to play every card, not just the meta.

- every real meta list (lab/decks/meta), repeated so they stay a big share of the games;
- one deck per attacker line in the whole card pool (generic shell, Rare Candy for Stage 2);
- random two-line decks and random mutations of meta decks (odd cards, incomplete lines...).
"""
import collections
import importlib.util
import json
import os
import random
import shutil

from .engine import ROOT


def _pipeline():
    spec = importlib.util.spec_from_file_location('pipeline', os.path.join(ROOT, 'lab', 'search', 'pipeline.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_pool(out_dir, meta_copies=15, pairs=300, mutants=300, seed=0, log=print):
    p = _pipeline()
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    rng = random.Random(seed)
    n = 0
    meta = []
    for f in sorted(os.listdir(p.META)):
        if f.endswith('.txt'):
            d = p.parse_deck(os.path.join(p.META, f))
            meta.append(d)
            for k in range(meta_copies):
                shutil.copy(os.path.join(p.META, f), os.path.join(out_dir, f'meta-{f[:-4]}-{k:02d}.txt'))
                n += 1
    finals = p.attacker_finals()
    lines = []
    for f in finals:
        ch = p.chain(f)
        if not ch:
            continue
        d = p.shell(p.line_cards(ch))
        if d and p.valid(d):
            p.write_deck(d, os.path.join(out_dir, f'line-{f.replace(" ", "_")}.txt'))
            lines.append(ch)
            n += 1
    for i in range(pairs):
        c1, c2 = rng.sample(lines, 2)
        d = p.shell(p.line_cards(c1, True) + p.line_cards(c2, True))
        if d and p.valid(d):
            p.write_deck(d, os.path.join(out_dir, f'pair-{i:04d}.txt'))
            n += 1
    for i in range(mutants):
        d = rng.choice(meta)
        for _ in range(rng.randint(1, 4)):
            d = p.mutate(d, rng) or d
        p.write_deck(d, os.path.join(out_dir, f'mutant-{i:04d}.txt'))
        n += 1
    log(f'pool de decks : {n} fichiers dans {out_dir} ({len(meta)} listes méta x{meta_copies}, '
        f'{len(lines)} lignées, paires et mutants)')
    extend_pool(out_dir, log=log)
    add_solo_decks(out_dir, log=log)
    return out_dir


COVERAGE_MARK = '.couverture'


def extend_pool(out_dir, seed=1, log=print):
    """Adds one deck for every card that no deck of `out_dir` contains yet: a Pokémon's line next
    to a real attacker's line, a Trainer in a meta list. Without it a third of the Pokémon (and
    the Fossils) never appeared in self-play, and the deck search, which may use any card, would
    judge them with a network that has never seen them played."""
    p = _pipeline()
    rng = random.Random(seed)
    seen = set()
    for f in os.listdir(out_dir):
        if f.endswith('.txt'):
            seen.update(p.sig(p.CARD[c]) for c in p.parse_deck(os.path.join(out_dir, f))['c'])
    attackers = [ch for ch in (p.chain(f) for f in p.attacker_finals()) if ch]
    meta = [p.parse_deck(os.path.join(p.META, f)) for f in sorted(os.listdir(p.META)) if f.endswith('.txt')]

    def add(d, cid):
        p.write_deck(d, os.path.join(out_dir, f'cover-{cid.replace(" ", "_")}.txt'))
        seen.update(p.sig(p.CARD[c]) for c in d['c'])

    added = 0
    for cid in p.POKEMON:
        if p.sig(p.CARD[cid]) in seen:
            continue
        ch = p.chain(cid) or [cid]
        for _ in range(30):
            d = p.shell(p.line_cards(ch, True) + p.line_cards(rng.choice(attackers), True))
            if d and p.valid(d) and cid in d['c']:
                add(d, cid)
                added += 1
                break
    trainers = p.TRAINERS + [c for c in p.UNIQUE if p.is_fossil(p.CARD[c])]
    for cid in trainers:
        if p.sig(p.CARD[cid]) in seen:
            continue
        for _ in range(50):
            base = rng.choice(meta)
            c = list(base['c'])
            slots = [i for i, x in enumerate(c) if not (p.is_poke(p.CARD[x]) or p.is_fossil(p.CARD[x]))]
            for i in rng.sample(slots, min(2, len(slots))):
                c[i] = cid
            d = {'e': base['e'], 'c': c}
            if p.valid(d):
                add(d, cid)
                added += 1
                break
    open(os.path.join(out_dir, COVERAGE_MARK), 'w').close()
    log(f'couverture : {added} decks ajoutés pour les cartes jamais jouées à l\'entraînement')
    return added


KINDS = {'meta': 'méta', 'line': 'lignée', 'pair': 'paire', 'mutant': 'mutant', 'cover': 'découverte',
         'solo': 'solo'}
RANKS = {'pokeball': 'Poké Ball', 'superball': 'Super Ball', 'hyperball': 'Hyper Ball', 'masterball': 'Master Ball'}
SOLO_DIR = os.path.join(ROOT, 'lab', 'decks', 'solo')
SOLO_MARK = '.solo'


def solo_name(file_stem):
    """'celestialguardians-alolan-muk-ex-and-toxapex' -> 'Alolan Muk ex & Toxapex'."""
    words = file_stem.split('-')[1:]
    pretty = {'ex': 'ex', 'and': '&'}
    return ' '.join(pretty.get(w, w.capitalize()) for w in words).replace("Rocket S ", "Rocket's ")


def add_solo_decks(out_dir, log=print):
    """Copies the developers' decks of the step-up solo battles (lab/decks/solo/<rank>, four
    ranks from Poké Ball to Master Ball) into the self-play pool: hand-made decks of every
    level, and they disappear from the game at the end of October 2026."""
    n = 0
    for rank in RANKS:
        folder = os.path.join(SOLO_DIR, rank)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if f.endswith('.txt'):
                shutil.copy(os.path.join(folder, f), os.path.join(out_dir, f'solo-{rank}-{f}'))
                n += 1
    open(os.path.join(out_dir, SOLO_MARK), 'w').close()
    log(f'decks du mode solo : {n} decks des développeurs ajoutés (Poké Ball à Master Ball)')
    return n
LABELS_FILE = '.noms.json'


def deck_labels(decks_dir):
    """{file stem: [label, archetype, kind]} for the highlights: meta lists by their name, other
    decks by their two strongest attackers; the archetype is the strongest attacker. Cached."""
    cache_f = os.path.join(decks_dir, LABELS_FILE)
    try:
        labels = json.load(open(cache_f, encoding='utf-8'))
    except (OSError, ValueError):
        labels = {}
    stems = [f[:-4] for f in os.listdir(decks_dir) if f.endswith('.txt')]
    missing = [s for s in stems if s not in labels]
    if missing:
        p = _pipeline()
        for stem in missing:
            d = p.parse_deck(os.path.join(decks_dir, stem + '.txt'))
            cnt = collections.Counter(d['c'])
            pre = {p.CARD[x].get('evolves_from') for x in cnt}
            att = collections.Counter()
            for cid, k in cnt.items():
                c = p.CARD[cid]
                if p.is_poke(c) and c['name'] not in pre:
                    att[c['name']] = max(att[c['name']], p.atk_value(c) * k + 1)
            top = [n for n, _ in att.most_common(2)] or ['?']
            kind = stem.split('-', 1)[0]
            if kind == 'meta':
                name = ' '.join(w.capitalize() for w in stem.split('-')[1:-1])
                labels[stem] = [f'{name} (méta)', name, kind]
            elif kind == 'solo':
                rank = stem.split('-')[1]
                labels[stem] = [f'{solo_name(stem.split("-", 2)[2])} ({RANKS.get(rank, rank)})', top[0], kind]
            else:
                labels[stem] = [f"{' + '.join(top)} ({KINDS.get(kind, kind)})", top[0], kind]
        try:
            json.dump(labels, open(cache_f, 'w', encoding='utf-8'), ensure_ascii=False)
        except OSError:
            pass
    return labels
