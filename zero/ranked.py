"""Simulated ranked season, with the official ladder rules of Pokémon TCG Pocket.

    python -m zero.ranked [--player z8] [--games 20] [--players 5000] [--matches 120] [--deck mon_deck.txt]

1. Population: the meta archetypes with their real share (lab/decks/sources/meta_share.json,
   DotGG), each played with its known lists (Limitless, Game8...), plus your --deck files.
2. Every pair of decks plays --games games with the trained network (both sides): a matchup
   matrix, computed once and cached (results/ranked/matrix.json).
3. A season: --players players, each with a deck drawn by meta share, play --matches games
   against opponents of their rank; results are drawn from the matrix. Rules (Serebii,
   ptcgpocket.gg, season B4a): win +10; loss 0 in Beginner, -5 in Poke/Great Ball, -7 in Ultra
   Ball, -10 in Master Ball; win-streak bonus +3/+6/+9/+12 (2nd to 5th win in a row) up to
   Great Ball; no demotion below the start of the current tier.
Result: results/ranked/CLASSEMENT.md, win rate per deck and per rank, share of players
reaching Master Ball, games needed.
"""
import argparse
import itertools
import json
import os
import random

from . import engine

TIERS = [('Débutant', 0), ('Poké Ball', 145), ('Super Ball', 355), ('Hyper Ball', 710), ('Master Ball', 1450)]
SUB = [0, 20, 50, 95, 145, 195, 245, 300, 355, 420, 490, 600, 710, 860, 1010, 1225, 1450]


def tier(points):
    return max(i for i, (_, t) in enumerate(TIERS) if points >= t)


def rank_name(points):
    if points >= 1450:
        return 'Master Ball'
    k = max(i for i, t in enumerate(SUB) if points >= t)
    return f'{TIERS[k // 4][0]} {k % 4 + 1}'


def play(points, streak, won):
    t = tier(points)
    if won:
        streak += 1
        bonus = [0, 0, 3, 6, 9, 12][min(streak, 5)] if t <= 2 else 0
        return points + 10 + bonus, streak
    loss = [0, 5, 5, 7, 10][t]
    return max(TIERS[t][1], points - loss), 0


def population(extra):
    share_f = os.path.join(engine.ROOT, 'lab', 'decks', 'sources', 'meta_share.json')
    arch = json.load(open(share_f, encoding='utf-8'))['archetypes']
    decks = {}  # path -> (archetype, weight)
    for name, a in arch.items():
        for d in a['decks']:
            decks[os.path.join(engine.ROOT, d)] = (name, a['share'] / len(a['decks']))
    mean = sum(w for _, w in decks.values()) / max(len(decks), 1)
    for d in extra:
        decks[os.path.abspath(d)] = (os.path.splitext(os.path.basename(d))[0] + ' (à toi)', mean)
    return decks


def matrix(decks, player, games, threads, cache_f):
    cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
    paths = sorted(decks)
    pairs = [(a, b) for a, b in itertools.combinations(paths, 2)
             if f'{player}|{games}|{a}|{b}' not in cache]
    for n, (a, b) in enumerate(pairs, 1):
        r = engine.match(a, b, player, games, threads)
        cache[f'{player}|{games}|{a}|{b}'] = r['score_a']
        if n % 20 == 0 or n == len(pairs):
            print(f'  matchups : {n}/{len(pairs)}', flush=True)
            json.dump(cache, open(cache_f, 'w'))

    def score(a, b):
        if a == b:
            return 0.5
        k = f'{player}|{games}|{min(a, b)}|{max(a, b)}'
        return cache[k] if a < b else 1 - cache[k]
    return score


def main():
    ap = argparse.ArgumentParser(description='Saison de ranked simulée (règles officielles)')
    ap.add_argument('--player', default='z8', help='bot des deux côtés (z8 : 8 simulations du réseau)')
    ap.add_argument('--weights', default=os.path.join(engine.ROOT, 'runs', 'principal', 'weights', 'latest.bin'))
    ap.add_argument('--games', type=int, default=20, help='parties par paire de decks pour la matrice')
    ap.add_argument('--players', type=int, default=5000)
    ap.add_argument('--matches', type=int, default=120, help='parties classées par joueur et par saison')
    ap.add_argument('--deck', action='append', default=[], help='ton deck, à comparer à la méta (répétable)')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    out = os.path.join(engine.ROOT, 'results', 'ranked')
    os.makedirs(out, exist_ok=True)
    player = engine.zspec(a.player[1:], a.weights) if a.player.startswith('z') and os.path.exists(a.weights) else a.player
    decks = population(a.deck)
    print(f'{len(decks)} decks, matrice {a.games} parties par paire avec {a.player}', flush=True)
    score = matrix(decks, player, a.games, a.threads or None, os.path.join(out, 'matrix.json'))

    rng = random.Random(a.seed)
    paths = sorted(decks)
    weights = [decks[p][1] for p in paths]
    pl = [{'deck': d, 'pts': 0, 'streak': 0, 'w': 0, 'g': 0, 'master_at': None,
           'by_tier': [[0, 0] for _ in TIERS]} for d in rng.choices(paths, weights, k=a.players)]
    for rnd in range(a.matches):
        order = sorted(range(len(pl)), key=lambda i: pl[i]['pts'] + rng.random() * 30)
        for i, j in zip(order[::2], order[1::2]):  # neighbours on the ladder meet
            p, q = pl[i], pl[j]
            won = rng.random() < score(p['deck'], q['deck'])
            for x, w in ((p, won), (q, not won)):
                t = tier(x['pts'])
                x['by_tier'][t][0] += w
                x['by_tier'][t][1] += 1
                x['pts'], x['streak'] = play(x['pts'], x['streak'], w)
                x['w'] += w
                x['g'] += 1
                if x['master_at'] is None and x['pts'] >= 1450:
                    x['master_at'] = rnd + 1

    rows = {}
    for x in pl:
        r = rows.setdefault(x['deck'], {'n': 0, 'w': 0, 'g': 0, 'master': 0, 'at': [], 'pts': 0,
                                        'tier': [[0, 0] for _ in TIERS]})
        r['n'] += 1
        r['w'] += x['w']
        r['g'] += x['g']
        r['pts'] += x['pts']
        if x['master_at']:
            r['master'] += 1
            r['at'].append(x['master_at'])
        for t in range(len(TIERS)):
            r['tier'][t][0] += x['by_tier'][t][0]
            r['tier'][t][1] += x['by_tier'][t][1]
    lines = [f'# Saison de ranked simulée', '',
             f'{a.players} joueurs, {a.matches} parties chacun, bot {a.player}, matrice {a.games} parties par paire. '
             'Règles officielles (B4a) : +10 par victoire, défaite 0 / -5 / -5 / -7 / -10 selon le rang, '
             'bonus de série jusqu\'à Super Ball, pas de descente sous le début du rang.', '',
             '| Deck | Archétype | Joueurs | Victoires | Master Ball | Parties pour Master | Rang moyen final | '
             + ' | '.join(f'% en {n}' for n, _ in TIERS) + ' |',
             '|' + '---|' * (7 + len(TIERS))]
    for d, r in sorted(rows.items(), key=lambda kv: -kv[1]['w'] / max(kv[1]['g'], 1)):
        at = sorted(r['at'])
        lines.append(f"| {os.path.relpath(d, engine.ROOT)} | {decks[d][0]} | {r['n']} | {100 * r['w'] / max(r['g'], 1):.1f} % | "
                     f"{100 * r['master'] / r['n']:.0f} % | {at[len(at) // 2] if at else '-'} | {rank_name(r['pts'] / r['n'])} | "
                     + ' | '.join(f"{100 * w / g:.0f} %" if g else '-' for w, g in r['tier']) + ' |')
    open(os.path.join(out, 'CLASSEMENT.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines[4:16]))
    print(f'\nClassement complet : {os.path.join(out, "CLASSEMENT.md")}')


if __name__ == '__main__':
    main()
