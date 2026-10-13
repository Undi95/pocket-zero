"""Fetches the current meta decklists into lab/decks/sources/<source>/ (run.bat meta, and the
"Meta update" GitHub workflow twice a week).

    python lab/tools/meta_sources.py                    # every source
    python lab/tools/meta_sources.py limitless game8    # some sources
    python lab/tools/meta_sources.py --summary FILE     # also writes a Markdown report

Sources: Limitless (tournament results of the last two weeks, public API), Game8 (tier list),
ptcgpocket.gg / DotGG (meta shares). Each deck file starts with # lines (archetype, site, source
URL, dates, tier or share) that the engine ignores. A file is rewritten only when its cards
change, decks that left a source's list are removed, and a list with a card the engine does not
know yet (new expansion) is skipped and reported. Standard library only. A source whose page
layout changed prints an error and is skipped: the files already there stay as they are.
"""
import calendar
import html
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'lab', 'decks', 'sources')
UA = {'User-Agent': 'Mozilla/5.0 (pocket-zero meta refresh)'}
ENERGY = ['Grass', 'Fire', 'Water', 'Lightning', 'Psychic', 'Fighting', 'Darkness', 'Metal']


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read().decode('utf-8', 'replace')


def write_if_changed(path, content):
    """Writes `content` unless the file already says the same thing apart from its retrieval
    date: a weekly update that finds the same lists changes nothing."""
    strip = lambda t: [l for l in t.splitlines() if not l.startswith('# retrieved:') and '"retrieved"' not in l]
    try:
        with open(path, encoding='utf-8') as f:
            if strip(f.read()) == strip(content):
                return False
    except OSError:
        pass
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)
    return True


def text(s):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', s))).strip()


def slug(name):
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def database():
    cards = {}
    for e in json.load(open(os.path.join(ROOT, 'database.json'), encoding='utf-8')):
        (kind, c), = e.items()
        cards[c['id']] = (kind, c)
    return cards


def write_deck(source, name, cards, energy, header):
    """cards: {card id: count}. Returns the file path, or None (with the reason) if invalid."""
    db = database()
    total = sum(cards.values())
    names = {}
    for cid, n in cards.items():
        if cid not in db:
            return None, f'carte inconnue {cid}'
        names[db[cid][1]['name']] = names.get(db[cid][1]['name'], 0) + n
    if total != 20 or max(names.values()) > 2:
        return None, f'{total} cartes, max {max(names.values())} exemplaires'
    if not energy:  # fall back to the Pokémon types
        energy = sorted({db[c][1].get('energy_type') for c in cards if db[c][0] == 'Pokemon'}
                        & set(ENERGY))
    os.makedirs(os.path.join(OUT, source), exist_ok=True)
    path = os.path.join(OUT, source, slug(name) + '.txt')
    lines = [f'# {k}: {v}' for k, v in header.items() if v]
    lines.append('Energy: ' + ', '.join(energy))
    for cid, n in sorted(cards.items(), key=lambda kv: (db[kv[0]][0] != 'Pokemon', kv[0])):
        st, num = cid.split()
        lines.append(f'{n} {db[cid][1]["name"]} {st} {int(num)}')
    write_if_changed(path, '\n'.join(lines) + '\n')
    return path, None


def limitless(days=14, min_players=32, max_tournaments=20, min_share=0.01):
    """Limitless (play.limitlesstcg.com) public API: the standard-format tournaments of the last
    `days` days with at least `min_players` players. Per archetype (Limitless's own naming): its
    share of all the players, its win rate, and the list of its best finisher (most wins, then
    the biggest tournament). Archetypes under `min_share` are left out."""
    api = 'https://play.limitlesstcg.com/api'
    cutoff = time.time() - days * 86400
    when = lambda t: calendar.timegm(time.strptime(t['date'][:19], '%Y-%m-%dT%H:%M:%S'))
    tours = []
    for page in range(1, 4):
        batch = json.loads(fetch(f'{api}/tournaments?game=POCKET&limit=100&page={page}'))
        tours += [t for t in batch if t.get('format') in (None, 'STANDARD') and t.get('players', 0) >= min_players
                  and when(t) >= cutoff]
        if not batch or when(batch[-1]) < cutoff:
            break
    tours = sorted(tours, key=lambda t: -t['players'])[:max_tournaments]
    arch, total = {}, 0
    for t in tours:
        for r in json.loads(fetch(f'{api}/tournaments/{t["id"]}/standings')):
            if not r.get('deck') or not r.get('decklist'):
                continue
            rec = r.get('record') or {}
            w, l, d = rec.get('wins', 0), rec.get('losses', 0), rec.get('ties', 0)
            a = arch.setdefault(r['deck']['id'], {'name': r['deck']['name'], 'players': 0, 'w': 0, 'l': 0, 'd': 0, 'best': None})
            a['players'] += 1
            a['w'], a['l'], a['d'] = a['w'] + w, a['l'] + l, a['d'] + d
            score = (w - l, t['players'], -(r.get('placing') or 999))
            if a['best'] is None or score > a['best'][0]:
                a['best'] = (score, t, r)
            total += 1
        time.sleep(0.5)  # be gentle with the API
    if not total:
        raise RuntimeError('aucune liste dans les tournois récents')
    db, out, shares = database(), [], {}
    for key, a in sorted(arch.items(), key=lambda kv: -kv[1]['players']):
        share = a['players'] / total
        games = a['w'] + a['l'] + a['d']
        shares[a['name']] = {'share': round(share, 4), 'players': a['players'],
                             'win_rate': round((a['w'] + a['d'] / 2) / games, 3) if games else None}
        if share < min_share:
            continue
        _, t, r = a['best']
        counts, bad = {}, []
        for c in r['decklist'].get('pokemon', []) + r['decklist'].get('trainer', []):
            cid = f'{c["set"]} {int(c["number"]):03d}' if str(c.get('number', '')).isdigit() else None
            if cid not in db:
                cid = by_name(c.get('name', ''))
            if cid is None:
                bad.append(f'{c.get("name")} {c.get("set")} {c.get("number")}')
                continue
            counts[cid] = counts.get(cid, 0) + int(c['count'])
        rec = r.get('record') or {}
        header = {'archetype': a['name'], 'site': 'Limitless (play.limitlesstcg.com)',
                  'source': f'https://play.limitlesstcg.com/tournament/{t["id"]}/player/{r["player"]}/decklist',
                  'retrieved': time.strftime('%Y-%m-%d'), 'published': t['date'][:10],
                  'share': f'{100 * share:.1f} % des joueurs ({a["players"]} sur {total}, {len(tours)} tournois de {days} jours)',
                  'win_rate': f'{100 * shares[a["name"]]["win_rate"]:.1f} %' if shares[a['name']]['win_rate'] is not None else '',
                  'list': f'{r["player"]}, {r.get("placing")}e sur {t["players"]} ({rec.get("wins", 0)}-{rec.get("losses", 0)}-{rec.get("ties", 0)}), {t["name"]}'}
        if bad:
            out.append((a['name'], f'{100 * share:.1f} %', None, 'cartes inconnues du moteur : ' + ', '.join(bad)))
            continue
        path, err = write_deck('limitless', a['name'], counts, r['decklist'].get('energy') or [], header)
        out.append((a['name'], f'{100 * share:.1f} %', path, err))
    os.makedirs(os.path.join(OUT, 'limitless'), exist_ok=True)
    write_if_changed(os.path.join(OUT, 'limitless', 'shares.json'), json.dumps(
        {'site': 'Limitless (play.limitlesstcg.com)', 'source': f'{api}/tournaments?game=POCKET',
         'retrieved': time.strftime('%Y-%m-%d'), 'days': days, 'tournaments': len(tours), 'players': total,
         'shares': shares}, indent=1, ensure_ascii=False) + '\n')
    return out


def by_name(name):
    """Card id for a card name (any printing: identical reprints play the same)."""
    want = name.lower().replace('\u2019', "'")
    for cid, (_, c) in database().items():
        if c['name'].lower() == want:
            return cid
    return None


def game8():
    """Game8 "Best Decks Tier List": every deck table shows one image per copy, with the card
    id in the alt text ("Pokemon TCG Pocket - Suicune ex (A4a 020)"). Names with an apostrophe
    break the attribute (alt="... - Professor" s research): those are matched by name."""
    url = 'https://game8.co/games/Pokemon-TCG-Pocket/archives/477754'
    page = fetch(url)
    m = re.search(r'Last updated on:\s*([A-Z][a-z]+ \d+, \d{4})', page)
    published = time.strftime('%Y-%m-%d', time.strptime(m.group(1), '%B %d, %Y')) if m else ''
    tiers = [(mt.start(), mt.group(1)) for mt in re.finditer(r'>\s*([SABCD][+-]?) Tier Decks', page)]
    out = []
    for mt in re.finditer(r'<table.*?</table>', page, flags=re.S):
        table = mt.group(0)
        if 'Card List' not in table:
            continue
        tier = ([t for pos, t in tiers if pos < mt.start()] or [''])[-1]
        name, counts, bad = '', {}, []
        for alt, rest in re.findall(r'alt="Pokemon TCG Pocket - ([^"]*)"([^>]*?)(?:data-src|width)', table):
            mid = re.search(r'\(([A-Za-z0-9-]+) (\d+)\)\s*$', alt)
            if alt.endswith(' Deck'):
                name = alt[:-5]
                continue
            if mid:
                cid = f'{mid.group(1)} {int(mid.group(2)):03d}'
            else:
                full = alt + "'" + rest.strip() if rest.strip() else alt
                cid = by_name(full)
                if cid is None:
                    bad.append(full)
                    continue
            counts[cid] = counts.get(cid, 0) + 1
        energy = [e for e in ENERGY if f'alt="{e} Icon"' in table]
        if not name:
            continue
        header = {'archetype': name, 'site': 'Game8 (Best Decks Tier List)', 'source': url,
                  'retrieved': time.strftime('%Y-%m-%d'), 'published': published, 'tier': tier}
        if bad:
            header['notes'] = 'cartes non reconnues : ' + ', '.join(bad)
        path, err = write_deck('game8', name, counts, energy, header)
        out.append((name, tier, path, err))
    return out


def dotgg():
    """ptcgpocket.gg (DotGG) public tier-list API: decks per archetype in their latest standard
    period (e.g. "b4a-day10"), i.e. the meta share. No lists (premium): the shares are matched
    to our deck files in meta_share.json."""
    url = 'https://api.dotgg.gg/cgfw/gettierlist?game=pokepocket'
    rows = json.loads(fetch(url))

    def period(r):
        m = re.search(r'-standard-([a-z0-9]+)-(day\w+)$', r['slug'])
        return (m.group(1), 1 if m.group(2) != 'dayzero' else 0) if m else None

    periods = {period(r) for r in rows if r.get('format') == 'standard'} - {None}
    latest = max(periods)
    cur = [r for r in rows if r.get('format') == 'standard' and period(r) == latest]
    total = sum(int(r['total_decks']) for r in cur) or 1
    shares = {}
    for r in cur:
        shares[r['name']] = shares.get(r['name'], 0) + int(r['total_decks']) / total
    os.makedirs(os.path.join(OUT, 'dotgg'), exist_ok=True)
    write_if_changed(os.path.join(OUT, 'dotgg', 'shares.json'), json.dumps(
        {'source': url, 'site': 'ptcgpocket.gg (DotGG)', 'retrieved': time.strftime('%Y-%m-%d'),
         'period': '-'.join(map(str, latest)), 'decks': total,
         'shares': dict(sorted(shares.items(), key=lambda kv: -kv[1]))}, indent=1, ensure_ascii=False) + '\n')
    return [(n, f'{100 * v:.1f} %', 'dotgg/shares.json', None) for n, v in sorted(shares.items(), key=lambda kv: -kv[1])[:15]]


def deck_pokemon(path):
    names = set()
    for line in open(path, encoding='utf-8'):
        m = re.match(r'\d+ (.+) ([A-Za-z0-9-]+) (\d+)\s*$', line.strip())
        if m:
            names.add(m.group(1).lower())
    return names


def match_shares(shares):
    """meta_share.json: each archetype's share split over the deck files (Limitless lists in
    lab/decks/meta, other sources here) that contain all the Pokémon of its name."""
    db_names = sorted({c['name'].lower() for k, c in database().values() if k == 'Pokemon'}, key=len, reverse=True)
    files = [os.path.join(ROOT, 'lab', 'decks', 'meta', f) for f in sorted(os.listdir(os.path.join(ROOT, 'lab', 'decks', 'meta')))]
    for src in sorted(os.listdir(OUT)):
        d = os.path.join(OUT, src)
        if os.path.isdir(d):
            files += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith('.txt')]
    contents = {f: deck_pokemon(f) for f in files}
    out, unmatched = {}, []
    for arch, share in shares.items():
        rest, want = arch.lower(), []
        while rest.strip():
            n = next((n for n in db_names if rest.strip().startswith(n + ' ') or rest.strip() == n), None)
            if n is None:
                break
            want.append(n)
            rest = rest.strip()[len(n):]
        hit = [f for f, c in contents.items() if want and all(w in c for w in want)]
        if hit:
            out[arch] = {'share': round(share, 4), 'decks': [os.path.relpath(f, ROOT).replace(os.sep, '/') for f in hit]}
        else:
            unmatched.append(arch)
    write_if_changed(os.path.join(OUT, 'meta_share.json'),
                     json.dumps({'archetypes': out, 'unmatched': unmatched[:40]}, indent=1, ensure_ascii=False) + '\n')


def remove_stale(source, kept):
    """Deletes the deck files of a source that its latest list no longer has."""
    folder = os.path.join(OUT, source)
    gone = []
    for f in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        p = os.path.join(folder, f)
        if f.endswith('.txt') and p not in kept:
            os.remove(p)
            gone.append(f[:-4])
    return gone


SOURCES = {'limitless': limitless, 'game8': game8, 'dotgg': dotgg}


def main():
    args = sys.argv[1:]
    summary = None
    if '--summary' in args:
        i = args.index('--summary')
        summary = args[i + 1]
        del args[i:i + 2]
    report = [f'# Meta update {time.strftime("%Y-%m-%d")}', '']
    for n in args or list(SOURCES):
        try:
            res = SOURCES[n]()
        except Exception as e:  # layout change, network: keep the files already there
            print(f'{n} : échec ({e})')
            report += [f'## {n}: failed', '', f'`{e}` (previous files kept)', '']
            continue
        kept = {r[2] for r in res if r[2] and r[2].endswith('.txt')}
        gone = remove_stale(n, kept) if len(kept) >= 3 else []  # a near-empty result is suspicious
        print(f'{n} : {len(kept) or len([r for r in res if r[2]])} decks sur {len(res)}'
              + (f', retirés : {", ".join(gone)}' if gone else ''))
        report += [f'## {n}', '']
        for name, tier, path, err in res:
            print(f'  [{tier or "-"}] {name}' + (f'  -> ignoré : {err}' if err else ''))
            report.append(f'- [{tier or "-"}] {name}' + (f' — **skipped**: {err}' if err else ''))
        if gone:
            report.append(f'- removed (no longer listed): {", ".join(gone)}')
        report.append('')
    try:
        match_shares(json.load(open(os.path.join(OUT, 'dotgg', 'shares.json'), encoding='utf-8'))['shares'])
    except (OSError, ValueError, KeyError) as e:
        print(f'meta_share.json : non mis à jour ({e})')
    if summary:
        with open(summary, 'a', encoding='utf-8') as f:
            f.write('\n'.join(report) + '\n')


if __name__ == '__main__':
    main()
