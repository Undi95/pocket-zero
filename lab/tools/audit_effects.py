"""Static audit of the effect-text -> mechanic maps. Run after adding a set:

    python3 lab/tools/audit_effects.py

1. Numbers: every damage/heal amount (>= 10) written on a card must appear in its mechanic,
   and the mechanic must not carry amounts that aren't on the card.
2. Qualifiers: words that restrict an effect (Basic, Benched, [X] Pokémon, coin flips, ex)
   must be reflected in the mechanic's name or fields (this is how Pichu's Crackly Toss,
   mapped to a generic "any Benched Pokémon" mechanic, was caught).
Output is a list of suspects to review by hand; known false positives: "During your next turn"
effects use duration 2, and parameterless named mechanics (e.g. MoltresExInfernoDance) hide
their qualifiers in code.
"""
import os, re
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPS = ['src/actions/effect_mechanic_map.rs', 'src/actions/effect_ability_mechanic_map.rs']
L = {'G': 'Grass', 'R': 'Fire', 'W': 'Water', 'L': 'Lightning', 'P': 'Psychic', 'F': 'Fighting', 'D': 'Darkness', 'M': 'Metal'}
QUALIFIERS = [(r'\bBasic\b', r'[Bb]asic|[Ss]tage'), (r'[Ff]lip', r'[Cc]oin|[Ff]lip|[Hh]eads|binomial|[Tt]ails|Chance'),
              (r'Pokémon ex\b', r'[Ee]x\b|Ex|_ex|ex_'), (r'Stage 2', r'[Ss]tage'), (r'Benched', r'[Bb]ench')]

def entries():
    for p in MAPS:
        s = open(os.path.join(ROOT, p)).read()
        s = '\n'.join(l for l in s.split('\n') if not l.strip().startswith('//'))
        for m in re.finditer(r'map\.insert\(\s*"((?:[^"\\]|\\.)*)",\s*(.*?)\);\n', s, re.S):
            yield p, m.group(1).replace('\\u{e9}', 'é').replace('\\u{a0}', ' ').replace('\\u{201c}', '"').replace('\\u{201d}', '"'), re.sub(r'//.*', '', m.group(2))

def nums(t): return {int(x) for x in re.findall(r'(?<![A-Za-z\d])(\d+)(?!\d)', t)}

found = 0
for path, text, expr in entries():
    e = re.sub(r'\b[A-Za-z_]*\d+[A-Za-z_]\w*', '', expr)  # drop card ids like A1035Charizard
    tn, en = nums(text), nums(e)
    issues = []
    missing = {x for x in tn - en if x >= 10} if en else set()
    if missing: issues.append(f'montant absent du code: {sorted(missing)}')
    extra = {x for x in en - tn if x >= 10}
    if extra: issues.append(f'montant absent du texte: {sorted(extra)}')
    for tr, er in QUALIFIERS:
        if re.search(tr, text) and not re.search(er, expr): issues.append(f'qualificatif "{tr}" absent')
    for x in re.findall(r'(?:Benched|Active|your) \[(\w)\] Pokémon', text):
        if x in L and L[x] not in expr: issues.append(f'type [{x}] absent')
    if issues:
        found += 1
        print(f'{os.path.basename(path)}: {text[:140]}\n    -> {" ".join(expr.split())[:160]}\n    !! {"; ".join(issues)}')
print(f'{found} suspects')
