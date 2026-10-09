"""Wire B4b trainer reprints to the existing logic of their original printing."""
import re, sys, subprocess
ROOT = sys.argv[1]
FILES = [f'{ROOT}/src/actions/apply_trainer_action.rs', f'{ROOT}/src/move_generation/move_generation_trainer.rs']
ids = open(f'{ROOT}/src/card_ids.rs').read()
out = subprocess.run([f'{ROOT}/target/release/card_status', '--incomplete-only'], capture_output=True, text=True).stdout
missing = re.findall(r'^(B4b) (\d+)\s', out, re.M)
strip = lambda x: re.sub(r'^(?:[A-Z]\d+[a-z]?|PA|PB)\d+', '', x)
enum_of = {}
for m in re.finditer(r'^\s+(\w+),$', ids, re.M):
    e = m.group(1); mm = re.match(r'^(B4b)(\d+)', e)
    if mm: enum_of[(mm.group(1), mm.group(2))] = e
done, fail = [], []
for st, num in missing:
    new = enum_of.get((st, num))
    if not new: fail.append((st, num, 'enum?')); continue
    suf = strip(new); ok = 0
    for f in FILES:
        src = open(f).read()
        if f'CardId::{new}' in src: ok += 1; continue
        # first match-arm line containing an existing CardId with the same suffix
        pat = re.compile(r'(CardId::(\w+))(?=[^\n]*=>)')
        hit = None
        for m in pat.finditer(src):
            if strip(m.group(2)) == suf and m.group(2) != new: hit = m; break
        if not hit: continue
        src = src[:hit.end(1)] + f' | CardId::{new}' + src[hit.end(1):]
        open(f, 'w').write(src); ok += 1
    (done if ok == len(FILES) else fail).append((st, num, new, ok))
print('wired', len(done)); print('not fully wired', fail)
