"""AlphaZero loop: self-play -> train -> evaluate, repeated. Resumable: rerun the same command.

    python -m zero.loop --run principal --iterations 40

Everything for a run lives in runs/<run>/: deck pool, shards, weights, metrics.csv, RAPPORT.md.
The latest network is runs/<run>/weights/latest.bin; use it anywhere with the `z` player:
    set POCKET_ZERO_WEIGHTS=runs\\principal\\weights\\latest.bin
"""
import argparse
import csv
import json
import os
import shutil
import time

import torch

from . import engine, gpu_selfplay
from .data import load_buffer
from .decks import build_pool
from .model import ZeroNet, export_weights
from .train import train

# Real Limitless winrates (format B4a), same as lab/tools/validate.py.
REAL = {('butterfree', 'lucario'): 62.4, ('butterfree', 'altaria'): 48.0, ('butterfree', 'dedenne'): 55.7,
        ('butterfree', 'hoopa'): 60.6, ('lucario', 'altaria'): 27.2, ('lucario', 'dedenne'): 64.4,
        ('lucario', 'hoopa'): 57.7, ('altaria', 'dedenne'): 48.4, ('altaria', 'hoopa'): 25.8,
        ('dedenne', 'hoopa'): 67.3}


def validate(player, games, threads, log):
    meta = os.path.join(engine.ROOT, 'lab', 'decks', 'meta')
    errs = []
    for (a, b), real in REAL.items():
        r = engine.match(os.path.join(meta, a + '.txt'), os.path.join(meta, b + '.txt'), player, games, threads)
        sim = 100 * r['score_a']
        errs.append(sim - real)
        log(f'    {a:10} vs {b:8} sim {sim:5.1f}  réel {real:5.1f}  écart {sim - real:+5.1f}')
    return (sum(e * e for e in errs) / len(errs)) ** 0.5


def main():
    ap = argparse.ArgumentParser(description='Boucle AlphaZero pocket-zero')
    ap.add_argument('--run', default='principal')
    ap.add_argument('--iterations', type=int, default=40)
    ap.add_argument('--games', type=int, default=1500, help='parties de self-play par itération')
    ap.add_argument('--sims', type=int, default=128, help='simulations MCTS par décision en self-play')
    ap.add_argument('--first-sims', type=int, default=200,
                    help="simulations de la 1re itération si pas d'échauffement (sans réseau)")
    ap.add_argument('--warmup-games', type=int, default=8000,
                    help="échauffement : parties où le réseau imite le bot e3 avant le self-play (0 = aucun)")
    ap.add_argument('--warmup-player', default='e3')
    ap.add_argument('--warmup-epochs', type=int, default=4)
    ap.add_argument('--threads', type=int, default=0, help='0 = tous les cœurs')
    ap.add_argument('--window', type=int, default=600_000, help='échantillons gardés pour l\'entraînement')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                    help="appareil d'entraînement")
    ap.add_argument('--selfplay-devices', default='auto',
                    help="GPU du self-play, ex. 'cuda:0,cuda:1' (auto = tous ; cpu = sans GPU)")
    ap.add_argument('--concurrency', type=int, default=1024, help='parties simultanées par GPU')
    ap.add_argument('--eval-games', type=int, default=200)
    ap.add_argument('--eval-sims', type=int, default=32)
    ap.add_argument('--validate-every', type=int, default=5, help='validation Limitless toutes les N itérations (0 = jamais)')
    ap.add_argument('--validate-games', type=int, default=100)
    a = ap.parse_args()

    run_dir = os.path.join(engine.ROOT, 'runs', a.run)
    for sub in ('shards', 'weights'):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    log_f = open(os.path.join(run_dir, 'log.txt'), 'a', encoding='utf-8')

    def log(*x):
        s = time.strftime('%H:%M:%S ') + ' '.join(str(v) for v in x)
        print(s, flush=True)
        log_f.write(s + '\n')
        log_f.flush()

    threads = a.threads or None
    info = engine.info()
    state_f = os.path.join(run_dir, 'state.json')
    state = json.load(open(state_f)) if os.path.exists(state_f) else {'iteration': 0, 'info': info}
    if state.get('info') != info:
        raise SystemExit('Le format des entrées a changé depuis le début de ce run : utilise un nouveau --run.')
    decks = os.path.join(run_dir, 'decks')
    if not os.path.isdir(decks):
        build_pool(decks, log=log)

    model = ZeroNet(info).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    ckpt = os.path.join(run_dir, 'weights', 'checkpoint.pt')
    if os.path.exists(ckpt):
        c = torch.load(ckpt, map_location=a.device)
        model.load_state_dict(c['model'])
        optimizer.load_state_dict(c['optimizer'])
    latest = os.path.join(run_dir, 'weights', 'latest.bin')
    log(f'run {a.run} : reprise à l\'itération {state["iteration"]}, appareil {a.device}, '
        f'{torch.cuda.device_count()} GPU')

    metrics_f = os.path.join(run_dir, 'metrics.csv')
    new_metrics = not os.path.exists(metrics_f)
    with open(metrics_f, 'a', newline='') as mf:
        w = csv.writer(mf)
        if new_metrics:
            w.writerow(['iteration', 'samples', 'policy_loss', 'value_loss', 'val_policy_loss', 'val_value_loss',
                        'vs_previous', 'vs_e2', 'rmse_limitless', 'selfplay_s', 'train_s'])

    meta = os.path.join(engine.ROOT, 'lab', 'decks', 'meta')
    if a.warmup_games and not state.get('warmup_done') and state['iteration'] == 0:
        log(f'=== échauffement : le réseau apprend en imitant {a.warmup_player} ({a.warmup_games} parties)')
        shard = os.path.join(run_dir, 'shards', '000_warmup.bin')
        t0 = time.time()
        sp = engine.imitate(decks, a.warmup_games, shard, a.warmup_player, threads)
        log(f'  {sp["games"]} parties, {sp["samples"]} positions ({(time.time() - t0) / 60:.1f} min)')
        buf = load_buffer([shard], a.window)
        train(model, optimizer, buf, a.device, epochs=a.warmup_epochs, batch=a.batch, q_mix=0.0, log=log)
        del buf
        export_weights(model, latest)
        shutil.copy(latest, os.path.join(run_dir, 'weights', 'warmup.bin'))
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, ckpt)
        vs_e2 = engine.arena(engine.zspec(a.eval_sims, latest), 'e2', meta, a.eval_games, threads)['a_score']
        log(f'  après échauffement : {100 * vs_e2:.1f} % contre e2')
        state['warmup_done'] = True
        json.dump(state, open(state_f, 'w'))

    for it in range(state['iteration'], a.iterations):
        log(f'=== itération {it + 1}/{a.iterations}')
        prev = latest if os.path.exists(latest) else None
        shard = os.path.join(run_dir, 'shards', f'iter_{it:03d}')
        t0 = time.time()
        if prev is None:
            # No network yet: CPU self-play with the engine's hand-written evaluation.
            sp = engine.selfplay(decks, a.games, a.first_sims, shard + '_w0.bin', None, threads, seed=it)
        else:
            devices = None if a.selfplay_devices == 'auto' else a.selfplay_devices.split(',')
            deck_paths = sorted(os.path.join(decks, f) for f in os.listdir(decks))
            sp, _ = gpu_selfplay.run(deck_paths, a.games, a.sims, shard, prev, info, seed=it,
                                     devices=devices, concurrency=a.concurrency)
        t_sp = time.time() - t0
        log(f'  self-play : {sp["games"]} parties, {sp["samples"]} positions ({t_sp / 60:.1f} min)')

        shards = sorted(os.path.join(run_dir, 'shards', f) for f in os.listdir(os.path.join(run_dir, 'shards')))
        buf = load_buffer(shards, a.window)
        t0 = time.time()
        m = train(model, optimizer, buf, a.device, epochs=a.epochs, batch=a.batch, log=log, seed=it)
        t_tr = time.time() - t0
        del buf
        new = os.path.join(run_dir, 'weights', f'iter_{it:03d}.bin')
        export_weights(model, new)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, ckpt)

        vs_prev = engine.arena(engine.zspec(a.eval_sims, new), engine.zspec(a.eval_sims, prev), meta,
                               a.eval_games, threads, seed=it)['a_score']
        vs_e2 = engine.arena(engine.zspec(a.eval_sims, new), 'e2', meta, a.eval_games, threads, seed=it)['a_score']
        log(f'  nouveau réseau : {100 * vs_prev:.1f} % contre le précédent, {100 * vs_e2:.1f} % contre e2')
        shutil.copy(new, latest)

        rmse = ''
        if a.validate_every and (it + 1) % a.validate_every == 0:
            log('  validation contre les vrais matchups Limitless :')
            rmse = validate(engine.zspec(a.eval_sims, latest), a.validate_games, threads, log)
            log(f'  écart moyen (RMSE) : {rmse:.1f} points (référence e3 : 11-12)')
        with open(metrics_f, 'a', newline='') as mf:
            csv.writer(mf).writerow([it + 1, m['samples'], f'{m["policy_loss"]:.4f}', f'{m["value_loss"]:.4f}',
                                     f'{m.get("val_policy_loss", 0):.4f}', f'{m.get("val_value_loss", 0):.4f}',
                                     f'{vs_prev:.3f}', f'{vs_e2:.3f}', f'{rmse:.1f}' if rmse != '' else '',
                                     f'{t_sp:.0f}', f'{t_tr:.0f}'])
        state['iteration'] = it + 1
        json.dump(state, open(state_f, 'w'))
    log('terminé. Réseau final :', latest)


if __name__ == '__main__':
    main()
