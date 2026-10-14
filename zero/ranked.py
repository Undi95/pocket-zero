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
   Great Ball; no demotion below the start of the current tier; a tie changes no points and
   ends the win streak (Pokémon Zone's ranked guide).
Result: results/ranked/CLASSEMENT.md, win rate per deck and per rank, share of players
reaching Master Ball, games needed.
"""
import argparse
import itertools
import json
import os
import random

from . import engine
from .live import Live, Off

TIERS = [('Débutant', 0), ('Poké Ball', 145), ('Super Ball', 355), ('Hyper Ball', 710), ('Master Ball', 1450)]
SUB = [0, 20, 50, 95, 145, 195, 245, 300, 355, 420, 490, 600, 710, 860, 1010, 1225, 1450]


def tier(points):
    return max(i for i, (_, t) in enumerate(TIERS) if points >= t)


def rank_name(points):
    if points >= 1450:
        return 'Master Ball'
    k = max(i for i, t in enumerate(SUB) if points >= t)
    return f'{TIERS[k // 4][0]} {k % 4 + 1}'


def play(points, streak, result):
    """Points and win streak after a game: result 1 (win), 0.5 (tie) or 0 (loss)."""
    t = tier(points)
    if result == 0.5:
        return points, 0
    if result == 1:
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


def matrix(decks, player, games, threads, cache_f, weights=None, live=Off()):
    cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
    paths = sorted(decks)
    key = lambda a, b: f'v2|{player}|{games}|{a}|{b}'  # v2: [score, share of ties]
    pairs = [(a, b) for a, b in itertools.combinations(paths, 2) if key(a, b) not in cache]
    from . import gpu_eval
    if pairs and player.startswith('z') and weights and gpu_eval.available():
        # Every matchup in one batched job on the GPUs.
        judge = gpu_eval.GpuJudge(weights, engine.info())
        idx = {p: i for i, p in enumerate(paths)}
        games_list = [(idx[a], idx[b], k * 7919 + idx[a] * 104_729 + idx[b]) for a, b in pairs for k in range(games)]
        print(f'  {len(games_list)} parties sur {len(judge.devices)} GPU...', flush=True)
        live.event(f'matrice : {len(games_list)} parties sur {len(judge.devices)} GPU')
        res, step = [], 4096 * len(judge.devices)  # in slices, for the progress bar (each one keeps the GPUs full)
        for k in range(0, len(games_list), step):
            res += judge.play(paths, games_list[k:k + step], int(player.split(':')[0][1:]))
            live.set(True, progress=(len(res), len(games_list)), current={'games': f'{len(res)}/{len(games_list)}'})
        judge.close()
        for n, (a, b) in enumerate(pairs):
            r = res[n * games:(n + 1) * games]
            cache[key(a, b)] = [sum(r) / games, sum(x == 0.5 for x in r) / games]
        json.dump(cache, open(cache_f, 'w'))
        pairs = []
    for n, (a, b) in enumerate(pairs, 1):
        r = engine.match(a, b, player, games, threads)
        cache[key(a, b)] = [r['score_a'], r.get('ties', 0.0)]
        live.set(progress=(n, len(pairs)), current={'pairs': f'{n}/{len(pairs)}'})
        if n % 20 == 0 or n == len(pairs):
            print(f'  matchups : {n}/{len(pairs)}', flush=True)
            json.dump(cache, open(cache_f, 'w'))

    def score(a, b):
        """(score of a against b, share of ties)."""
        if a == b:
            return 0.5, 0.0
        s, ties = cache[key(min(a, b), max(a, b))]
        return (s if a < b else 1 - s), ties
    return score


def main():
    ap = argparse.ArgumentParser(description='Saison de ranked simulée (règles officielles)')
    ap.add_argument('--player', default='z64', help='bot des deux côtés (z64 : 64 simulations, sur GPU si disponible)')
    ap.add_argument('--weights', default=engine.default_weights())
    ap.add_argument('--games', type=int, default=20, help='parties par paire de decks pour la matrice')
    ap.add_argument('--players', type=int, default=5000)
    ap.add_argument('--matches', type=int, default=120, help='parties classées par joueur et par saison')
    ap.add_argument('--deck', action='append', default=[], help='ton deck, à comparer à la méta (répétable)')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    live = Live('ranked', 'saison', player=a.player, players=a.players, matches=a.matches, games=a.games)
    try:
        season(a, live)
    except BaseException as e:
        live.event(f'arrêt : {type(e).__name__} {e}')
        live.done('stopped' if isinstance(e, KeyboardInterrupt) else 'failed')
        raise
    live.done()


def season(a, live):
    out = os.path.join(engine.ROOT, 'results', 'ranked')
    os.makedirs(out, exist_ok=True)
    player = engine.zspec(a.player[1:], a.weights) if a.player.startswith('z') and os.path.exists(a.weights) else a.player
    decks = population(a.deck)
    print(f'{len(decks)} decks, matrice {a.games} parties par paire avec {a.player}', flush=True)
    live.d['info'].update(decks=len(decks))
    live.set(True, phase='matrix')
    live.event(f'{len(decks)} decks, matrice {a.games} parties par paire avec {a.player}')
    score = matrix(decks, player, a.games, a.threads or None, os.path.join(out, 'matrix.json'), a.weights, live)
    live.set(True, phase='season', progress=(0, a.matches), current={})

    rng = random.Random(a.seed)
    paths = sorted(decks)
    weights = [decks[p][1] for p in paths]
    pl = [{'deck': d, 'pts': 0, 'streak': 0, 'w': 0, 'g': 0, 'master_at': None,
           'by_tier': [[0, 0] for _ in TIERS]} for d in rng.choices(paths, weights, k=a.players)]
    for rnd in range(a.matches):
        order = sorted(range(len(pl)), key=lambda i: pl[i]['pts'] + rng.random() * 30)
        for i, j in zip(order[::2], order[1::2]):  # neighbours on the ladder meet
            p, q = pl[i], pl[j]
            s, ties = score(p['deck'], q['deck'])
            u = rng.random()
            result = 1 if u < s - ties / 2 else 0.5 if u < s + ties / 2 else 0
            for x, w in ((p, result), (q, 1 - result)):
                t = tier(x['pts'])
                x['by_tier'][t][0] += w
                x['by_tier'][t][1] += 1
                x['pts'], x['streak'] = play(x['pts'], x['streak'], w)
                x['w'] += w
                x['g'] += 1
                if x['master_at'] is None and x['pts'] >= 1450:
                    x['master_at'] = rnd + 1
        if rnd % 10 == 9:
            live.set(progress=(rnd + 1, a.matches),
                     current={'round': rnd + 1, 'master': sum(x['master_at'] is not None for x in pl)})

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
    live.rows([{'key': d, 'deck': os.path.splitext(os.path.basename(d))[0], 'archetype': decks[d][0],
                'path': os.path.relpath(d, engine.ROOT).replace(os.sep, '/'), 'players': r['n'],
                'win': round(r['w'] / max(r['g'], 1), 4), 'master': round(r['master'] / r['n'], 4),
                'games_master': sorted(r['at'])[len(r['at']) // 2] if r['at'] else None}
               for d, r in sorted(rows.items(), key=lambda kv: -kv[1]['w'] / max(kv[1]['g'], 1))])
    live.set(True, phase='done', progress=(a.matches, a.matches))
    live.event('classement écrit : results/ranked/CLASSEMENT.md')
    print('\n'.join(lines[4:16]))
    print(f'\nClassement complet : {os.path.join(out, "CLASSEMENT.md")}')


if __name__ == '__main__':
    main()
