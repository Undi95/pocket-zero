"""Rebuilds lab/decks/solo/ from Serebii's step-up battle pages (the developers' decks of the
solo mode, four ranks from Poké Ball to Master Ball).

    python lab/tools/solo_decks.py [--cache DIR]

Card ids are resolved against database.json (Serebii links give set page + number); decks
that are not legal 20-card lists are skipped and reported."""
import argparse
import collections
import html
import importlib.util
import json
import os
import re
import shutil
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = 'https://www.serebii.net/tcgpocket/'
RANK = {'beginner': 'pokeball', 'intermediate': 'superball', 'advanced': 'hyperball', 'expert': 'masterball'}


def fetch(url, cache):
    path = os.path.join(cache, os.path.basename(url)) if cache else None
    if path and os.path.exists(path):
        return open(path, encoding='latin-1').read()
    req = urllib.request.Request(url, headers={'User-Agent': 'pocket-zero deck import'})
    data = urllib.request.urlopen(req, timeout=60).read().decode('latin-1')
    if path:
        open(path, 'w', encoding='latin-1').write(data)
    time.sleep(0.3)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', help='folder keeping the downloaded pages')
    a = ap.parse_args()
    if a.cache:
        os.makedirs(a.cache, exist_ok=True)
    spec = importlib.util.spec_from_file_location('pipeline', os.path.join(ROOT, 'lab', 'search', 'pipeline.py'))
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    by_id = p.CARD
    codes = sorted({i.split()[0] for i in by_id})
    norm = lambda n: re.sub(r'[^a-z0-9]', '', n.lower().replace('é', 'e'))
    words = lambda n: set(re.findall(r'[a-z0-9]+', n.lower()))

    hub = fetch(BASE + 'solobattles.shtml', a.cache)
    pages = sorted(set(re.findall(r'href="(solobattles/[^"]*step-up[^"]*)"', hub)))
    decks = []
    for page in pages:
        s = fetch(BASE + page, a.cache)
        try:
            s = s.encode('latin-1').decode('utf-8')
        except UnicodeError:
            pass
        exp, level = re.match(r'solobattles/(.+)-step-upbattle(\w+)\.shtml', page).groups()
        parts = re.split(r'<h2>([^<]*)</h2>', s)
        for k in range(1, len(parts), 2):
            body = parts[k + 1]
            if 'Deck Listing' not in body.split('<h2>')[0]:
                continue
            name = re.sub(r'\s*\([^)]*\)\s*$', '', html.unescape(parts[k]).strip(' ;')).strip() or 'Deck'
            body = body.split('Battle Tasks')[0]
            cards = re.findall(r'<td class="fooevo"><a href="/tcgpocket/([^/]+)/(\d+)\.shtml">([^<]+)</a>', body)
            qty = [int(q) for q in re.findall(r'<b>Quantity</b><br />\s*(\d+)', body)]
            if cards and len(cards) == len(qty):
                decks.append((exp, level, name, [(sl, int(n), html.unescape(nm), q)
                                                 for (sl, n, nm), q in zip(cards, qty)]))
    votes = collections.defaultdict(collections.Counter)
    for *_, cards in decks:
        for sl, n, nm, _ in cards:
            for c in codes:
                card = by_id.get(f'{c} {n:03d}')
                if card and norm(card['name']) == norm(nm):
                    votes[sl][c] += 1
    slug = {sl: v.most_common(1)[0][0] for sl, v in votes.items()}

    out = os.path.join(ROOT, 'lab', 'decks', 'solo')
    shutil.rmtree(out, ignore_errors=True)
    seen, stats, skipped = set(), collections.Counter(), []
    for exp, level, name, cards in decks:
        ids = []
        for sl, n, nm, q in cards:
            cid = f'{slug.get(sl, "?")} {n:03d}'
            card = by_id.get(cid)
            if not (card and words(nm) <= words(card['name'])):  # "Mega Charizard ex" = ... Y ex
                cid = next((i for i, c in by_id.items() if norm(c['name']) == norm(nm)), None)
            if cid:
                ids += [p.CANON.get(p.sig(by_id[cid]), cid)] * q
        d = p.mk(ids)
        if len(ids) != 20 or not p.valid(d):
            skipped.append(f'{exp} {level} {name} ({len(ids)} cartes)')
            continue
        key = (level, p.key(d))
        if key in seen:
            continue
        seen.add(key)
        rank = RANK[level]
        os.makedirs(os.path.join(out, rank), exist_ok=True)
        stem = re.sub(r'[^a-z0-9]+', '-', name.lower().replace('&', 'and')).strip('-').replace('-deck', '')
        base = os.path.join(out, rank, f"{re.sub(r'[^a-z0-9]+', '', exp)}-{stem}")
        path, k = base + '.txt', 2
        while os.path.exists(path):
            path, k = f'{base}-{k}.txt', k + 1
        p.write_deck(d, path)
        stats[rank] += 1
    print(dict(stats), f'; {len(skipped)} ignorés :')
    for s in skipped:
        print('  ', s)


if __name__ == '__main__':
    main()
