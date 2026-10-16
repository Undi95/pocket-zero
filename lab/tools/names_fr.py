"""French card names for the French logs: lab/data/names_fr.json, {English name: French name}.

    python lab/tools/names_fr.py

Source: TCGdex (api.tcgdex.net), French and English names of every card: the Pocket sets
first, then the whole TCG for the species. Names TCGdex has not translated yet are built from
their parts (Mega X ex -> Méga-X-ex, Team Rocket's X -> X de la Team Rocket); what is left keeps
its English name.
"""
import json, os, re, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'lab', 'data', 'names_fr.json')
API = 'https://api.tcgdex.net/v2'
FORMS = [(r'^(.+) ex$', '{}-ex'), (r'^Mega (.+)$', 'Méga-{}'), (r"^Team Rocket's (.+)$", '{} de la Team Rocket'),
         (r'^Alolan (.+)$', "{} d'Alola"), (r'^Galarian (.+)$', '{} de Galar'), (r'^Hisuian (.+)$', '{} de Hisui'),
         (r'^Paldean (.+)$', '{} de Paldea')]


def get(path):
    req = urllib.request.Request(f'{API}/{path}', headers={'User-Agent': 'pocket-zero'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def pairs(fr, en):
    fr = {c['id']: c['name'] for c in fr}
    return {c['name']: fr[c['id']] for c in en if fr.get(c['id'])}


def translate(name, known):
    if name in known:
        return known[name]
    for pat, fmt in FORMS:
        m = re.match(pat, name)
        if m:
            inner = translate(m[1], known)
            return fmt.format(inner) if inner else None
    return None


def main():
    pocket = {}
    for s in get('fr/series/tcgp')['sets']:
        pocket.update(pairs(get(f"fr/sets/{s['id']}").get('cards', []), get(f"en/sets/{s['id']}").get('cards', [])))
    known = {**pairs(get('fr/cards'), get('en/cards')), **pocket}  # Pocket spellings win
    db = json.load(open(os.path.join(ROOT, 'database.json'), encoding='utf-8'))
    names = sorted({c['name'] for e in db for c in e.values()})
    out = {n: f for n in names if (f := translate(n, known))}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=0, sort_keys=True)
    print(f'{len(out)}/{len(names)} noms traduits -> {OUT}')
    print('restent en anglais :', ', '.join(n for n in names if n not in out))


if __name__ == '__main__':
    main()
