"""Highlights of an iteration's self-play games, so a long training run is fun to follow:
best deck, meta ranking, gems found outside the meta, streaks, fastest win, biggest upset...

Reads the game logs written by the engine (one CSV per self-play environment, see
src/zero/batch.rs) and the deck labels of zero/decks.py."""
import collections
import csv


def fmt_int(n):
    return f'{n:,}'.replace(',', ' ')


def pct(x):
    return f'{100 * x:.0f} %'


def read_games(paths):
    games = []
    for p in paths:
        try:
            with open(p, newline='', encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    games.append((r['deck_a'], r['deck_b'], int(r['first']), int(r['winner']),
                                  int(r['turns']), int(r['points_a']), int(r['points_b'])))
        except (OSError, KeyError, ValueError):
            continue
    return games


def longest_streak(results):
    best = cur = 0
    for won in results:
        cur = cur + 1 if won else 0
        best = max(best, cur)
    return best


def summarize(paths, labels, iteration):
    """Lines to print after the self-play of `iteration` (empty when there is no log)."""
    games = read_games(paths)
    if not games:
        return []
    label = lambda s: labels.get(s, [s, s, '?'])[0]
    arch = lambda s: labels.get(s, [s, s, '?'])[1]
    kind = lambda s: labels.get(s, [s, s, '?'])[2]

    deck = collections.defaultdict(lambda: [0, 0])     # stem -> [wins, games]
    group = collections.defaultdict(lambda: [0, 0])    # archetype -> [wins, games]
    gems = collections.defaultdict(lambda: [0, 0])     # archetype outside the meta
    meta = collections.defaultdict(lambda: [0, 0])     # meta list -> [wins, games]
    seq = collections.defaultdict(list)                # archetype -> results in order
    first_wins = decided = ties = 0
    for a, b, first, winner, turns, pa, pb in games:
        if winner < 0:
            ties += 1
        else:
            decided += 1
            first_wins += winner == first
        for side, stem in ((0, a), (1, b)):
            won = winner == side
            for table, k in ((deck, stem), (group, arch(stem))):
                table[k][0] += won
                table[k][1] += 1
            (meta if kind(stem) == 'meta' else gems)[arch(stem)][0] += won
            (meta if kind(stem) == 'meta' else gems)[arch(stem)][1] += 1
            seq[arch(stem)].append(won)

    rate = lambda t, k: t[k][0] / t[k][1]
    out = [f"── Moments forts de l'itération {iteration} ({fmt_int(len(games))} parties) ──"]

    cands = [k for k, (w, n) in deck.items() if n >= 6]
    if cands:
        best = max(cands, key=lambda k: (rate(deck, k), deck[k][1]))
        w, n = deck[best]
        out.append(f"🏆 Deck de l'itération : {label(best)} — {w} victoires sur {n} ({pct(w / n)})")
    ranked = sorted((k for k, (w, n) in meta.items() if n >= 30), key=lambda k: -rate(meta, k))
    if len(ranked) >= 4:
        top = ' · '.join(f'{k} {pct(rate(meta, k))}' for k in ranked[:3])
        out.append(f'📊 Méta (contre tout le pool) : {top} … lanterne rouge : '
                   f'{ranked[-1]} {pct(rate(meta, ranked[-1]))}')
    devs = [k for k, (w, n) in deck.items() if kind(k) == "solo" and n >= 6]
    if devs:
        best = max(devs, key=lambda k: (rate(deck, k), deck[k][1]))
        w, n = deck[best]
        out.append(f'🎮 Meilleur deck des devs : {label(best)} — {w} victoires sur {n} ({pct(w / n)})')
    shiny = sorted((k for k, (w, n) in gems.items() if n >= 40), key=lambda k: -rate(gems, k))
    if shiny:
        out.append('💎 Pépites hors méta : ' + ' · '.join(
            f'{k} {pct(rate(gems, k))} ({gems[k][1]} parties)' for k in shiny[:3]))
    streak_arch = max(seq, key=lambda k: longest_streak(seq[k]))
    streak = longest_streak(seq[streak_arch])
    if streak >= 2:
        out.append(f"🔥 Plus longue série : {streak_arch} — {streak} victoires d'affilée")
    wins = [g for g in games if g[3] >= 0]
    if wins:
        a, b, first, winner, turns, pa, pb = min(wins, key=lambda g: (g[4], -max(g[5], g[6])))
        w_stem, l_stem = (a, b) if winner == 0 else (b, a)
        score = f'{max(pa, pb)}-{min(pa, pb)}'
        out.append(f'⚡ Victoire éclair : {label(w_stem)} bat {label(l_stem)} au tour {turns} ({score})')
    solid = {k for k, (w, n) in group.items() if n >= 40}
    upsets = [(rate(group, arch(l)) - rate(group, arch(w)), w, l)
              for a, b, _, winner, *_ in wins
              for w, l in [((a, b) if winner == 0 else (b, a))]
              if arch(w) in solid and arch(l) in solid]
    if upsets:
        gap, w, l = max(upsets)
        if gap > 0.2:
            out.append(f'🎯 Exploit : {arch(w)} ({pct(rate(group, arch(w)))} de victoires) a battu '
                       f'{arch(l)} ({pct(rate(group, arch(l)))})')
    a, b, *_, turns, _, _ = max(games, key=lambda g: g[4])
    line = f'🐢 Marathon : {turns} tours entre {label(a)} et {label(b)}'
    if ties:
        line += f' · {ties} partie{"s" if ties > 1 else ""} nulle{"s" if ties > 1 else ""}'
    out.append(line)
    if decided:
        out.append(f'🎲 Le joueur qui commence gagne {pct(first_wins / decided)} des parties')
    return out
