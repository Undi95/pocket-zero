"""Deck pool for self-play: the bot has to learn to play every card, not just the meta.

- every real meta list (lab/decks/meta), repeated so they stay a big share of the games;
- one deck per attacker line in the whole card pool (generic shell, Rare Candy for Stage 2);
- random two-line decks and random mutations of meta decks (odd cards, incomplete lines...).
"""
import importlib.util
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
    return out_dir
