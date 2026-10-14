"""Rebuilds the reference meta lists (lab/decks/meta/*.txt) and the real matchup table that
validates the bot (lab/decks/meta/reference.json), from Limitless.

    python lab/tools/meta_reference.py              # lists + matchups (run.bat meta --reference)
    python lab/tools/meta_reference.py --matchups   # matchups only, the lists stay as they are

Run it between training runs, never during one: the lists and the table are a run's yardstick.

Lists. Every reference file belongs to a Limitless archetype (its '# limitless:' line, found by
card overlap the first time). It becomes the most representative real list of that archetype in
the standard tournaments of the last 3 weeks: the medoid, the list with the fewest card
differences to all the others. Its energy is the one its player declared, else the one most
players of the archetype declare. A file whose archetype has too few lists is left alone.

Matchups. Limitless's own matchup table for the current format (hundreds of matches per pair,
far more than our sample of tournaments), between reference archetypes. Part of those matches
are best-of-3, which push a win rate away from 50 %; the bot plays single games, so each rate is
converted back to a single-game rate with the share of Bo3 matches measured in the tournaments.
"""
import collections
import html
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meta_sources as ms  # noqa: E402  fetch, database, write_if_changed

ROOT = ms.ROOT
META = os.path.join(ROOT, 'lab', 'decks', 'meta')
API = 'https://play.limitlesstcg.com/api'
SITE = 'https://play.limitlesstcg.com'


def collect(days=21, min_players=32, max_tournaments=60):
    """Standard tournaments of the last `days` days: standings (decks, lists) and pairings."""
    cutoff = time.time() - days * 86400
    when = lambda t: ms.calendar.timegm(time.strptime(t['date'][:19], '%Y-%m-%dT%H:%M:%S'))
    tours = []
    for page in range(1, 6):
        batch = json.loads(ms.fetch(f'{API}/tournaments?game=POCKET&limit=100&page={page}'))
        tours += [t for t in batch if t.get('format') in (None, 'STANDARD') and t.get('players', 0) >= min_players
                  and when(t) >= cutoff]
        if not batch or when(batch[-1]) < cutoff:
            break
    out = []
    for t in sorted(tours, key=lambda t: -t['players'])[:max_tournaments]:
        out.append({'t': t, 'details': json.loads(ms.fetch(f'{API}/tournaments/{t["id"]}/details')),
                    'standings': json.loads(ms.fetch(f'{API}/tournaments/{t["id"]}/standings')),
                    'pairings': json.loads(ms.fetch(f'{API}/tournaments/{t["id"]}/pairings'))})
        time.sleep(0.3)  # be gentle with the API
    return out


def card_id(db, c):
    cid = f'{c["set"]} {int(c["number"]):03d}' if str(c.get('number', '')).isdigit() else None
    return cid if cid in db else ms.by_name(c.get('name', ''))


def decklists(data, db):
    """{archetype id: [(cards by id, cards by name, energy, player, tournament)]} and their names."""
    lists, names = collections.defaultdict(list), {}
    for x in data:
        for r in x['standings']:
            d, dl = r.get('deck'), r.get('decklist')
            if not d or not dl:
                continue
            ids, by_name, ok = collections.Counter(), collections.Counter(), True
            for c in dl.get('pokemon', []) + dl.get('trainer', []):
                cid = card_id(db, c)
                if cid is None:
                    ok = False
                    break
                ids[cid] += int(c['count'])
                by_name[db[cid][1]['name']] += int(c['count'])
            if ok and sum(ids.values()) == 20:
                lists[d['id']].append((ids, by_name, tuple(sorted(dl.get('energy') or [])), r.get('player'), x['t']))
                names[d['id']] = d['name']
    return lists, names


def diff(a, b):
    return sum((a - b).values())


def read_ref(path, db):
    header, energy, by_name = {}, [], collections.Counter()
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        m = re.match(r'#\s*([\w-]+):\s*(.*)$', line)
        if m:
            header[m[1]] = m[2]
        elif line.lower().startswith('energy:'):
            energy = [e.strip() for e in line.split(':', 1)[1].split(',') if e.strip()]
        else:
            m = re.match(r'(\d+) (.+) ([A-Za-z0-9-]+) (\d+)$', line)
            if m:
                cid = f'{m[3]} {int(m[4]):03d}'
                by_name[db[cid][1]['name'] if cid in db else m[2]] += int(m[1])
    return header, energy, by_name


def bo3_share(data):
    """Share of the matches played as best-of-3 in these tournaments."""
    n = n3 = 0
    for x in data:
        modes = {p['phase']: p.get('mode') for p in x['details'].get('phases', [])}
        for m in x['pairings']:
            if m.get('player2') and m.get('winner') != -1:
                n += 1
                n3 += modes.get(m.get('phase')) == 'BO3'
    return n3 / n if n else 0.0


def single_game(m, f):
    """Single-game win rate p whose mix of Bo1 and Bo3 (share f) wins m of the matches."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        p = (lo + hi) / 2
        if (1 - f) * p + f * (3 * p * p - 2 * p ** 3) < m:
            lo = p
        else:
            hi = p
    return (lo + hi) / 2


def matchup_page(arch_id):
    """Limitless's matchup table of one archetype: ({opponent name: (wins, losses, ties)}, period)."""
    page = ms.fetch(f'{SITE}/decks/{arch_id}/matchups?game=POCKET')
    title = re.search(r'<title>(.*?)</title>', page, flags=re.S)
    period = html.unescape(title.group(1)).split(' - ')[1].strip() if title and ' - ' in title.group(1) else ''
    rows = {}
    for tr in re.findall(r'<tr data-name="([^"]*)" data-matches="(\d+)"[^>]*>(.*?)</tr>', page, flags=re.S):
        name, _, cells = html.unescape(tr[0]), tr[1], tr[2]
        m = re.search(r'(\d+)\s*-\s*(\d+)\s*-\s*(\d+)', ms.text(cells))
        if m and sum(map(int, m.groups())) > sum(rows.get(name, (0, 0, 0))):  # the page has other, smaller tables
            rows[name] = tuple(int(v) for v in m.groups())
    return rows, period


def write_list(path, db, header, energy, ids):
    lines = [f'# {k}: {v}' for k, v in header.items() if v]
    lines.append('Energy: ' + ', '.join(energy))
    for cid, n in sorted(ids.items(), key=lambda kv: (db[kv[0]][0] != 'Pokemon', kv[0])):
        st, num = cid.split()
        lines.append(f'{n} {db[cid][1]["name"]} {st} {int(num)}')
    return ms.write_if_changed(path, '\n'.join(lines) + '\n')


def main():
    only_matchups = '--matchups' in sys.argv
    db = ms.database()
    print('Tournois Limitless des 3 dernières semaines...', flush=True)
    data = collect()
    lists, names = decklists(data, db)
    total = sum(len(v) for v in lists.values())
    print(f'  {len(data)} tournois, {total} listes complètes, {len(lists)} archétypes')
    by_slug = {re.sub(r'[^a-z0-9]+', '-', n.lower()).strip('-'): i for i, n in names.items()}
    archetypes = {}
    for f in sorted(os.listdir(META)):
        if not f.endswith('.txt'):
            continue
        path, stem = os.path.join(META, f), f[:-4]
        header, energy, by_name = read_ref(path, db)
        arch = header.get('limitless')
        if arch not in lists:  # first time: the archetype whose lists are closest to this file
            cands = [i for i, v in lists.items() if len(v) >= 10]
            arch = min(cands, key=lambda i: sum(diff(by_name, l[1]) for l in lists[i]) / len(lists[i]), default=None)
        if not arch or len(lists.get(arch, [])) < 10:
            print(f'  {stem:20} : archétype introuvable ou trop rare, liste inchangée')
            continue
        group = lists[arch]
        archetypes[stem] = {'id': arch, 'name': names[arch], 'players': len(group), 'share': round(len(group) / total, 4)}
        if only_matchups:
            continue
        medoid = min(group, key=lambda l: sum(diff(l[1], o[1]) for o in group))
        declared = collections.Counter(l[2] for l in group if l[2])
        new_energy = list(medoid[2]) or (list(declared.most_common(1)[0][0]) if declared else energy)
        near = sum(diff(medoid[1], o[1]) <= 2 for o in group) / len(group)
        t = medoid[4]
        new_header = {'archetype': names[arch], 'limitless': arch,
                      'source': f'{SITE}/tournament/{t["id"]}/player/{medoid[3]}/decklist',
                      'list': f'la plus représentative de {len(group)} listes ({100 * near:.0f} % à 2 cartes près), '
                              f'{medoid[3]}, {t["name"]}, {t["date"][:10]}',
                      'energy': (f"{', '.join(new_energy)} pour {declared.most_common(1)[0][1]} des "
                                 f"{sum(declared.values())} joueurs qui la déclarent") if declared else ''}
        changed = write_list(path, db, new_header, new_energy, medoid[0])
        before = diff(by_name, medoid[1])
        print(f'  {stem:20} -> {names[arch]} ({len(group)} joueurs) : '
              + (f'{before} carte(s) et énergie {",".join(energy)} -> {",".join(new_energy)}' if changed else 'inchangée'))

    f3 = bo3_share(data)
    print(f'Matchups Limitless (part de Bo3 dans les tournois : {100 * f3:.0f} %)...', flush=True)
    by_name = {a['name']: stem for stem, a in archetypes.items()}
    pairs, period = {}, ''
    for stem, a in archetypes.items():
        rows, period = matchup_page(a['id'])
        for opp, (w, l, t) in rows.items():
            other = by_name.get(opp)
            if not other or other == stem:
                continue
            key = tuple(sorted((stem, other)))
            n = w + l + t
            if key in pairs and pairs[key]['matches'] >= n:
                continue
            raw = (w + t / 2) / n if n else 0.5
            raw = raw if key[0] == stem else 1 - raw
            p = single_game(raw, f3)
            pairs[key] = {'a': key[0], 'b': key[1], 'real': round(100 * p, 1), 'raw': round(100 * raw, 1),
                          'matches': n, 'se': round(100 * math.sqrt(p * (1 - p) / n), 1) if n else None}
        time.sleep(0.3)
    rows = sorted(pairs.values(), key=lambda m: -m['matches'])
    ref = {'retrieved': time.strftime('%Y-%m-%d'), 'period': period, 'source': f'{SITE}/decks?game=POCKET',
           'bo3_share': round(f3, 3), 'tournaments': len(data), 'lists': total,
           'archetypes': archetypes, 'matchups': rows}
    ms.write_if_changed(os.path.join(META, 'reference.json'), json.dumps(ref, indent=1, ensure_ascii=False) + '\n')
    print(f'  {len(rows)} matchups entre listes de référence ({period}), dont {sum(m["matches"] >= 200 for m in rows)} '
          f'avec au moins 200 matchs : lab/decks/meta/reference.json')
    for m in rows[:15]:
        print(f'    {m["a"]:18} vs {m["b"]:18} {m["real"]:5.1f} % ± {m["se"]:.1f}  (brut {m["raw"]:.1f} %, {m["matches"]} matchs)')


if __name__ == '__main__':
    main()
