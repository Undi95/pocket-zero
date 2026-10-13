"""Thin wrapper around the pocket-zero binary (self-play, arena, matches)."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = 'pocket-zero.exe' if os.name == 'nt' else 'pocket-zero'


def binary():
    for p in (os.path.join(ROOT, 'bin', EXE), os.path.join(ROOT, 'target', 'release', EXE)):
        if os.path.exists(p):
            return p
    sys.exit(f'{EXE} introuvable : lance setup.ps1 (Windows) ou `cargo build --release --bin pocket-zero`.')


NO_NETWORK = ("Réseau introuvable / network not found: {}\n"
              "  run.bat update  -> télécharge le réseau publié / downloads the published network\n"
              "  run.bat train   -> entraîne le tien / trains your own\n"
              "  --weights FILE  -> un autre fichier / another file")


def default_weights():
    """The network used when no --weights is given: the most recently trained run
    (runs/<run>/weights/latest.bin), else the published network that setup downloads
    (weights/pocket-zero.bin)."""
    import glob
    trained = glob.glob(os.path.join(ROOT, 'runs', '*', 'weights', 'latest.bin'))
    return max(trained, key=os.path.getmtime) if trained else os.path.join(ROOT, 'weights', 'pocket-zero.bin')


def run(args, threads=None, stream=True):
    cmd = [binary()] + (['--threads', str(threads)] if threads else []) + [str(a) for a in args]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None if stream else subprocess.PIPE,
                          text=True, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f'{" ".join(cmd)} a échoué ({proc.returncode}): {proc.stderr or ""}')
    out = proc.stdout.strip().splitlines()
    return json.loads(out[-1]) if out else None


def info():
    proc = subprocess.run([binary(), 'info'], stdout=subprocess.PIPE, text=True, check=True, cwd=ROOT)
    return json.loads(proc.stdout)


def selfplay(decks, games, sims, out, weights=None, threads=None, seed=0, temperature_moves=20,
             noise=0.25):
    args = ['selfplay', '--decks', decks, '--games', games, '--sims', sims, '--out', out,
            '--seed', seed, '--temperature-moves', temperature_moves, '--noise', noise]
    if weights:
        args += ['--weights', weights]
    return run(args, threads)


def imitate(decks, games, out, player='e3', threads=None, seed=0):
    return run(['imitate', '--decks', decks, '--games', games, '--out', out, '--player', player,
                '--seed', seed], threads)


def arena(a, b, decks, games, threads=None, seed=0):
    return run(['arena', '--a', a, '--b', b, '--decks', decks, '--games', games, '--seed', seed],
               threads)


def league(decks, player, games_per_deck, threads=None, seed=0):
    """{'games': n, 'decks': {file stem: [wins, games]}}: every deck of the folder against random
    others, `player` on both sides."""
    return run(['league', '--decks', decks, '--player', player, '--games-per-deck', games_per_deck,
                '--seed', seed], threads)


def match(deck_a, deck_b, player, games, threads=None, seed=0):
    return run(['match', '--deck-a', deck_a, '--deck-b', deck_b, '--player', player,
                '--games', games, '--seed', seed], threads, stream=False)


def parity(weights, shard, n=16):
    return run(['parity', '--weights', weights, '--shard', shard, '--n', n], stream=False)


def zspec(sims, weights):
    return f'z{sims}:{weights}' if weights else f'z{sims}'
