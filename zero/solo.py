"""Tournament of the developers' decks of the step-up solo battles (Poké Ball to Master Ball),
played by the trained network: a ranking per rank, then a grand tournament of the best of each
rank against the meta lists.

    python -m zero.solo [--weights runs\\<run>\\weights\\latest.bin] [--player z8]

Results: results/solo/CLASSEMENT.md. Run it with training stopped (they share the CPU).
"""
import argparse
import os
import shutil
import sys
import time

from . import engine
from .decks import RANKS, SOLO_DIR, solo_name


def ranking(res):
    rows = [(k, w / g, int(w), g) for k, (w, g) in res['decks'].items() if g]
    return sorted(rows, key=lambda r: (-r[1], -r[3]))


def main():
    ap = argparse.ArgumentParser(description='Tournoi des decks du mode solo')
    ap.add_argument('--weights', default=engine.default_weights())
    ap.add_argument('--player', default='z8', help='joueur des deux côtés (z8 : 8 simulations)')
    ap.add_argument('--games', type=int, default=40, help='parties par deck dans chaque tournoi')
    ap.add_argument('--top', type=int, default=4, help='decks de chaque rang qualifiés pour le grand tournoi')
    ap.add_argument('--threads', type=int, default=0)
    a = ap.parse_args()
    if not os.path.exists(a.weights):
        sys.exit(engine.NO_NETWORK.format(a.weights))
    player = engine.zspec(int(a.player[1:]), os.path.abspath(a.weights)) if a.player.startswith('z') else a.player
    threads = a.threads or None
    out_dir = os.path.join(engine.ROOT, 'results', 'solo')
    os.makedirs(out_dir, exist_ok=True)
    md = ['# Tournoi des decks du mode solo', '',
          f'Joueur : `{a.player}` des deux côtés, {a.games} parties par deck contre des decks tirés au hasard.', '']
    finals = os.path.join(out_dir, 'grand-tournoi')
    shutil.rmtree(finals, ignore_errors=True)
    os.makedirs(finals)
    t0 = time.time()
    for rank, pretty in RANKS.items():
        folder = os.path.join(SOLO_DIR, rank)
        n = len([f for f in os.listdir(folder) if f.endswith('.txt')])
        print(f'=== {pretty} : {n} decks', flush=True)
        rows = ranking(engine.league(folder, player, a.games, threads))
        md += [f'## {pretty} ({n} decks)', '', '| # | Deck | Victoires |', '|---|---|---|']
        for i, (k, r, w, g) in enumerate(rows, 1):
            md.append(f'| {i} | {solo_name(k)} | {100 * r:.0f} % ({w}/{g}) |')
        md.append('')
        for i, (k, r, w, g) in enumerate(rows[:5], 1):
            print(f'  {i}. {solo_name(k)} - {100 * r:.0f} %', flush=True)
        print(f'  ... lanterne rouge : {solo_name(rows[-1][0])} - {100 * rows[-1][1]:.0f} %', flush=True)
        for k, *_ in rows[:a.top]:
            shutil.copy(os.path.join(folder, k + '.txt'), os.path.join(finals, f'{rank}-{k}.txt'))
    meta = os.path.join(engine.ROOT, 'lab', 'decks', 'meta')
    for f in os.listdir(meta):
        if f.endswith('.txt'):
            shutil.copy(os.path.join(meta, f), os.path.join(finals, f'meta-{f}'))
    print('=== Grand tournoi : les meilleurs de chaque rang contre la méta', flush=True)
    rows = ranking(engine.league(finals, player, a.games * 2, threads))

    def name(stem):
        kind, rest = stem.split('-', 1)
        if kind == 'meta':
            return f'{rest.replace("-", " ").title()} (méta)'
        return f'{solo_name(rest)} ({RANKS[kind]})'

    md += ['## Grand tournoi : les meilleurs de chaque rang contre les 17 listes de la méta', '',
           '| # | Deck | Victoires |', '|---|---|---|']
    for i, (k, r, w, g) in enumerate(rows, 1):
        md.append(f'| {i} | {name(k)} | {100 * r:.0f} % ({w}/{g}) |')
        print(f'  {i:2}. {name(k)} - {100 * r:.0f} %', flush=True)
    path = os.path.join(out_dir, 'CLASSEMENT.md')
    open(path, 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print(f'Classement complet : {path} ({(time.time() - t0) / 60:.1f} min)', flush=True)


if __name__ == '__main__':
    main()
