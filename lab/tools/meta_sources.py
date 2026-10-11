"""Fetches meta decklists from sources other than Limitless into lab/decks/sources/<source>/.

    python lab/tools/meta_sources.py            # every source
    python lab/tools/meta_sources.py game8      # one source

Each deck file starts with # lines (archetype, site, source URL, dates, tier or share) that the
engine ignores, so the lists can be compared between sources and refreshed at every update.
Standard library only. A source whose page layout changed prints an error and is skipped:
the files already there stay as they are.
"""
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
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    return path, None


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
    with open(os.path.join(OUT, 'dotgg', 'shares.json'), 'w', encoding='utf-8') as f:
        json.dump({'source': url, 'site': 'ptcgpocket.gg (DotGG)', 'retrieved': time.strftime('%Y-%m-%d'),
                   'period': '-'.join(map(str, latest)), 'decks': total,
                   'shares': dict(sorted(shares.items(), key=lambda kv: -kv[1]))}, f, indent=1, ensure_ascii=False)
    match_shares(shares)
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
    with open(os.path.join(OUT, 'meta_share.json'), 'w', encoding='utf-8') as f:
        json.dump({'archetypes': out, 'unmatched': unmatched[:40]}, f, indent=1, ensure_ascii=False)


SOURCES = {'game8': game8, 'dotgg': dotgg}


def main():
    names = sys.argv[1:] or list(SOURCES)
    for n in names:
        try:
            res = SOURCES[n]()
        except Exception as e:  # layout change, network: keep the files already there
            print(f'{n} : échec ({e})')
            continue
        ok = [r for r in res if r[2]]
        print(f'{n} : {len(ok)} decks écrits sur {len(res)}')
        for name, tier, path, err in res:
            print(f'  [{tier or "-"}] {name}' + (f'  -> ignoré : {err}' if err else ''))


if __name__ == '__main__':
    main()
