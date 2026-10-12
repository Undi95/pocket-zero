"""Deck search over the WHOLE card pool, played by the trained network.

    python -m zero.search --name recherche1 [--weights runs\\principal\\weights\\latest.bin]
                          [--screen z0] [--search z8] [--final z32] [--gens 25] [--lock "B1a 42,B1a 62"]

Runs lab/search/pipeline.py (screening of every attacker line, pairs, genetic search with any
card, final validation) with the network as both players. Results: results/<name>/RESULTATS_D.md
(resumable: rerun the same command to continue).
"""
import argparse
import os
import subprocess
import sys

from .engine import ROOT


def main():
    ap = argparse.ArgumentParser(description='Recherche de decks avec le bot pocket-zero')
    ap.add_argument('--name', default='recherche')
    ap.add_argument('--weights', default=os.path.join(ROOT, 'runs', 'principal', 'weights', 'latest.bin'))
    # Network of iteration 81 against e3 (400 games): z0 43.8 %, z8 54.8 %, z32 56.6 %.
    ap.add_argument('--screen', default='z0', help='joueur du criblage (z0 = politique du réseau, rapide)')
    ap.add_argument('--search', default='z8',
                    help="joueur de l'algorithme génétique (z8 : 8 simulations, déjà plus fort qu'e3)")
    ap.add_argument('--final', default='z32', help='joueur de la validation finale (plus fort, plus lent)')
    ap.add_argument('--gens', type=int, default=25)
    ap.add_argument('--lock', default='', help='cartes imposées, ex. "B1a 42,B1a 62"')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--field-share', type=float, default=0.2,
                    help='part du terrain (archétypes trouvés par la recherche) dans le score, le reste = méta réelle')
    ap.add_argument('--min-pokemon', type=int, default=4, help='Pokémon minimum par deck')
    ap.add_argument('--check', action='store_true', help='diagnostic : un deck contre la méta, sortie brute du moteur')
    a = ap.parse_args()

    if not os.path.exists(a.weights):
        sys.exit(f"Réseau introuvable : {a.weights}\nEntraîne d'abord le bot (run.bat train) ou passe --weights.")
    env = dict(os.environ)
    env.update({
        'POCKET_ZERO_WEIGHTS': os.path.abspath(a.weights),
        'PZ_SCREEN': a.screen,
        'PZ_SEARCH': a.search,
        'PZ_FINAL': a.final,
        'PZ_RESULTS': os.path.join(ROOT, 'results', a.name),
        'PZ_FIELD_SHARE': str(a.field_share),
        'PZ_MIN_POKEMON': str(a.min_pokemon),
    })
    if a.threads:
        env['THREADS'] = str(a.threads)
    cmd = [sys.executable, os.path.join(ROOT, 'lab', 'search', 'pipeline.py'), '--gens', str(a.gens)]
    if a.lock:
        cmd += ['--lock', a.lock]
    if a.check:
        cmd += ['--check']
    print(f'Recherche "{a.name}" : criblage {a.screen}, génétique {a.search}, validation {a.final}, '
          f'réseau {a.weights}', flush=True)
    sys.exit(subprocess.call(cmd, env=env, cwd=ROOT))


if __name__ == '__main__':
    main()
