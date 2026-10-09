"""Find logic that names cards by CardId but forgets some prints of the same card.

    python3 lab/tools/audit_reprints.py

For every source file, each CardId it mentions is expanded to all prints with the same name and
the same rules text (attacks + ability, or trainer effect). Prints missing from that file are
reported: the effect silently doesn't work on them (e.g. Koga's Muk/Weezing check, Dragalge ex).
"""
import collections, glob, json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = json.load(open(os.path.join(ROOT, 'database.json')))
enum_src = open(os.path.join(ROOT, 'src/card_ids.rs')).read()

def sig(c):
    return (c['name'], json.dumps(c.get('attacks'), sort_keys=True), json.dumps(c.get('ability'), sort_keys=True),
            c.get('effect'))

id2enum = {}
for x in DB:
    _, c = next(iter(x.items()))
    s, n = c['id'].split()
    m = re.search(r'\b(' + re.escape(s.replace('-', '') + n) + r'[A-Za-z]\w*)\b', enum_src)
    if m: id2enum[c['id']] = m.group(1)
enum2card = {}
groups = collections.defaultdict(list)
for x in DB:
    _, c = next(iter(x.items()))
    if c['id'] in id2enum:
        enum2card[id2enum[c['id']]] = c
        groups[sig(c)].append(id2enum[c['id']])

found = 0
for f in sorted(glob.glob(os.path.join(ROOT, 'src/**/*.rs'), recursive=True)):
    rel = os.path.relpath(f, ROOT)
    # tools.rs / stadiums.rs / has_tool(...) match Tools and Stadiums by their effect text, so every
    # print is covered; players/ and temp_deck.rs are not game rules.
    if rel in ('src/database.rs', 'src/card_ids.rs', 'src/tools.rs', 'src/stadiums.rs', 'src/temp_deck.rs') \
            or '/bin/' in rel or '/players/' in rel:
        continue
    text = open(f).read()
    i = text.find('#[cfg(test)]')
    code = text if i < 0 else text[:i]
    used = {e for e in set(re.findall(r'CardId::(\w+)', code)) & set(enum2card)
            if enum2card[e].get('trainer_card_type') not in ('Tool', 'Stadium')}
    missing = collections.defaultdict(set)
    for e in used:
        for other in groups[sig(enum2card[e])]:
            if other not in used:
                missing[enum2card[e]['name']].add(other)
    for name, ms in sorted(missing.items()):
        found += 1
        print(f'{rel}: {name} -> manquent {sorted(ms)}')
print(f'{found} trous')
