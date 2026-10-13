"""Publishes a trained network so that everyone gets it with setup.bat / run.bat update.

    run.bat publish runs\\grand\\weights\\latest.bin "grand, iteration 150/150"

The file goes to the `weights` branch of this repository, alone with a VERSION line (the note,
the date, a checksum) and a short README. The branch is force-pushed: only the latest network is
kept there, the repository's history does not grow with every network. setup.ps1 downloads it
into weights/, where the tools find it when no run of your own is more recent.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
import time

from .engine import ROOT
from .model import weight_sizes

README = """# pocket-zero network / réseau

`pocket-zero.bin`: the published network, downloaded into `weights\\` by `setup.bat` and
`run.bat update`. `VERSION` says which one it is.

`pocket-zero.bin` : le réseau publié, téléchargé dans `weights\\` par `setup.bat` et
`run.bat update`. `VERSION` dit lequel c'est.

Publish another one / en publier un autre : `run.bat publish FILE "note"` (see docs/GUIDE.en.md, section 17).
"""


def main():
    ap = argparse.ArgumentParser(description='Publie un réseau pour tout le monde / publishes a network for everyone (branche weights)')
    ap.add_argument('weights', help='fichier du réseau / network file, ex. runs\\grand\\weights\\latest.bin')
    ap.add_argument('note', help='description, ex. "grand, iteration 150/150"')
    ap.add_argument('--remote', default='origin')
    a = ap.parse_args()
    sizes = weight_sizes(a.weights)  # fails loudly on a file that is not a network
    with open(a.weights, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    version = f'{a.note} | {time.strftime("%Y-%m-%d")} | sha256 {digest} | h1 {sizes["h1"]}'
    url = subprocess.check_output(['git', 'remote', 'get-url', a.remote], cwd=ROOT, text=True).strip()
    tmp = tempfile.mkdtemp(prefix='pz_publish_')
    shutil.copy(a.weights, os.path.join(tmp, 'pocket-zero.bin'))
    for name, text in (('VERSION', version + '\n'), ('README.md', README)):
        with open(os.path.join(tmp, name), 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
    git = lambda *args: subprocess.run(['git', *args], cwd=tmp, check=True)
    git('init', '-q', '-b', 'weights')
    git('add', '-A')
    git('-c', 'user.name=pocket-zero', '-c', 'user.email=pocket-zero@users.noreply.github.com',
        'commit', '-q', '-m', f'Network: {a.note}')
    git('push', '-q', '-f', url, 'weights')
    shutil.rmtree(tmp, ignore_errors=True)
    print(f'Publié / published: {version}\n  -> {url} (branche / branch weights)')


if __name__ == '__main__':
    main()
