"""AlphaZero loop: self-play -> train -> evaluate, repeated. Resumable: rerun the same command.

    python -m zero.loop --run principal --iterations 40

Everything for a run lives in runs/<run>/: deck pool, shards, weights, metrics.csv, RAPPORT.md.
The latest network is runs/<run>/weights/latest.bin; use it anywhere with the `z` player:
    set POCKET_ZERO_WEIGHTS=runs\\principal\\weights\\latest.bin
"""
import argparse
import csv
import json
import math
import os
import shutil
import time

import torch

from . import engine, gpu_selfplay
from .data import load_buffer, stale_shards
from .decks import COVERAGE_MARK, SOLO_MARK, add_solo_decks, build_pool, deck_labels, extend_pool
from .highlights import summarize
from .model import ZeroNet, export_weights, migrate_state_dict
from .train import train

TD_LAMBDA = 0.9

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


def compatible(old, new):
    """A run can continue across these input changes: layout 1 -> 2 and vocabulary changes
    (migrate_state_dict), everything else equal."""
    if not old:
        return False
    keep = ('n_slots', 'bags', 'action_dense', 'action_tokens', 'action_type_buckets',
            'action_detail_buckets', 'max_actions', 'embed_dim', 'action_embed', 'h1', 'h2', 'h',
            'vh', 'ph', 'action_in')
    return all(old.get(k) == new.get(k) for k in keep) and \
        old.get('trunk_in') in (new['trunk_in'], new['trunk_in_v1'])


def migrate_checkpoint(c, model, info):
    """Checkpoint of an older layout or vocabulary: weights and their AdamW moments are moved
    the same way (new inputs start at zero, embedding rows follow their card ids)."""
    names = [n for n, _ in model.named_parameters()]
    c['model'] = migrate_state_dict(dict(c['model']), info, c.get('vocab_table'))
    for i, st in c['optimizer']['state'].items():
        name = names[int(i)]
        if name in ('fc1.weight', 'card_emb.weight'):
            for k in ('exp_avg', 'exp_avg_sq'):
                if k in st:
                    st[k] = migrate_state_dict({'fc1.weight': c['model']['fc1.weight'], 'card_emb.weight': st[k]}
                                               if name == 'card_emb.weight' else
                                               {'card_emb.weight': c['model']['card_emb.weight'], 'fc1.weight': st[k]},
                                               info, c.get('vocab_table'))[name]


def records(metrics_f, vs_e3, streak=None):
    """Record lines against e3, compared with the previous iterations of the run."""
    best = None
    try:
        with open(metrics_f, newline='') as f:
            for r in csv.DictReader(f):
                if r.get('vs_e3'):
                    v = float(r['vs_e3'])
                    if best is None or v > best[0]:
                        best = (v, r['iteration'])
    except (OSError, ValueError):
        pass
    out = []
    if best is None or vs_e3 > best[0]:
        old = f' (ancien : {100 * best[0]:.1f} % à l\'itération {best[1]})' if best else ''
        out.append(f'[RECORD]   Nouveau record contre e3 : {100 * vs_e3:.1f} % !{old}')
    else:
        out.append(f'[RECORD]   Record du run contre e3 : {100 * best[0]:.1f} % (itération {best[1]})')
    if streak and streak >= 2:
        out.append(f"[SERIE]    Plus longue série contre e3 : {streak} victoires d'affilée")
    return out


def trend(metrics_f, key, n=5):
    """Mean of `key` over the last n rows and over the n rows before (None when missing)."""
    try:
        with open(metrics_f, newline='') as f:
            vals = [float(r[key]) for r in csv.DictReader(f) if r.get(key)]
    except (OSError, ValueError):
        return None, None
    cur = vals[-n:]
    old = vals[-2 * n:-n]
    mean = lambda v: sum(v) / len(v) if v else None
    return mean(cur), mean(old)


def main():
    ap = argparse.ArgumentParser(description='Boucle AlphaZero pocket-zero')
    ap.add_argument('--run', default='principal')
    ap.add_argument('--iterations', type=int, default=40)
    ap.add_argument('--add-decks', default='', help='dossier de decks à ajouter au pool (ex. results\\forge\\<nom>\\decks)')
    ap.add_argument('--games', type=int, default=10000, help='parties de self-play par itération')
    ap.add_argument('--sims', type=int, default=200, help='simulations MCTS par décision en self-play')
    ap.add_argument('--temperature-moves', type=int, default=8,
                    help="(ancienne recherche PUCT seulement) décisions jouées au hasard en début de partie ; "
                         "la recherche Gumbel explore d'elle-même")
    ap.add_argument('--first-sims', type=int, default=200,
                    help="simulations de la 1re itération si pas d'échauffement (sans réseau)")
    ap.add_argument('--warmup-games', type=int, default=8000,
                    help="échauffement : parties où le réseau imite le bot e3 avant le self-play (0 = aucun)")
    ap.add_argument('--warmup-player', default='e3')
    ap.add_argument('--warmup-epochs', type=int, default=4)
    ap.add_argument('--threads', type=int, default=0, help='0 = tous les cœurs')
    ap.add_argument('--window', type=int, default=2_000_000, help='positions gardées pour l\'entraînement')
    ap.add_argument('--keep-shards', action='store_true',
                    help="garder toutes les parties d'autojeu (sinon celles sorties de la fenêtre "
                         "d'entraînement sont supprimées : ~4 Go par itération)")
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--td-lambda', type=float, default=TD_LAMBDA,
                    help="cible de valeur TD(lambda) : s'appuie sur les évaluations des positions suivantes "
                         "au lieu du seul résultat final, beaucoup moins bruité (0 = résultat final seul)")
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3,
                    help="taux d'apprentissage de départ, réduit progressivement jusqu'à lr/10 à la dernière itération")
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                    help="appareil d'entraînement")
    ap.add_argument('--selfplay-devices', default='auto',
                    help="GPU du self-play, ex. 'cuda:0,cuda:1' (auto = tous ; cpu = sans GPU)")
    ap.add_argument('--concurrency', type=int, default=2048, help='parties simultanées par GPU')
    ap.add_argument('--eval-games', type=int, default=400)
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
        try:
            print(s, flush=True)
        except UnicodeEncodeError:  # a console without some character of the line
            print(s.encode('ascii', 'replace').decode(), flush=True)
        log_f.write(s + '\n')
        log_f.flush()

    threads = a.threads or None
    info = engine.info()
    # The vocabulary travels with the weights (they follow card ids), not with the run.
    dims = {k: v for k, v in info.items() if k != 'vocab_table'}
    state_f = os.path.join(run_dir, 'state.json')
    state = json.load(open(state_f)) if os.path.exists(state_f) else {'iteration': 0, 'info': dims}
    migrating = state.get('info') != dims
    if migrating and not compatible(state.get('info', {}), dims):
        raise SystemExit('Le format des entrées a changé depuis le début de ce run : utilise un nouveau --run.')
    decks = os.path.join(run_dir, 'decks')
    if not os.path.isdir(decks):
        build_pool(decks, log=log)
    else:  # runs started before these decks existed
        if not os.path.exists(os.path.join(decks, COVERAGE_MARK)):
            extend_pool(decks, log=log)
        if not os.path.exists(os.path.join(decks, SOLO_MARK)):
            add_solo_decks(decks, log=log)

    if a.add_decks:  # e.g. forged decks: the network learns to pilot them before they are judged
        added = [f for f in os.listdir(a.add_decks) if f.endswith('.txt')]
        for f in added:
            shutil.copy(os.path.join(a.add_decks, f), os.path.join(decks, 'forge-' + f))
        log(f'{len(added)} decks ajoutés au pool d\'entraînement depuis {a.add_decks}')

    model = ZeroNet(info).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    ckpt = os.path.join(run_dir, 'weights', 'checkpoint.pt')
    if os.path.exists(ckpt):
        c = torch.load(ckpt, map_location=a.device)
        if migrating:
            migrate_checkpoint(c, model, info)
        model.load_state_dict(c['model'])
        optimizer.load_state_dict(c['optimizer'])
    if migrating:
        log('nouvelles entrées du réseau (outils, cartes sous chaque Pokémon, effets, historique '
            'public...) : le réseau continue tel quel et apprend à les utiliser')
        state['info'] = dims
        json.dump(state, open(state_f, 'w'))
    latest = os.path.join(run_dir, 'weights', 'latest.bin')
    log(f'run {a.run} : reprise à l\'itération {state["iteration"]}, appareil {a.device}, '
        f'{torch.cuda.device_count()} GPU')

    metrics_f = os.path.join(run_dir, 'metrics.csv')
    header = ['iteration', 'samples', 'policy_loss', 'value_loss', 'val_policy_loss', 'val_value_loss',
              'vs_previous', 'vs_e2', 'vs_e3', 'rmse_limitless', 'lr', 'selfplay_s', 'train_s']
    if os.path.exists(metrics_f):
        with open(metrics_f) as mf:
            old_header = mf.readline().strip().split(',')
        # Renamed after closing it: Windows refuses to move a file that is still open.
        if old_header != header:
            os.replace(metrics_f, os.path.join(run_dir, 'metrics_ancien_format.csv'))
    new_metrics = not os.path.exists(metrics_f)
    with open(metrics_f, 'a', newline='') as mf:
        w = csv.writer(mf)
        if new_metrics:
            w.writerow(header)

    shard_dir = os.path.join(run_dir, 'shards')
    games_dir = os.path.join(run_dir, 'games')
    os.makedirs(games_dir, exist_ok=True)

    def prune():
        """Deletes the self-play files that fell out of the training window (they are never
        read again; each iteration writes ~4 GB)."""
        if a.keep_shards:
            return
        freed = 0
        for p in stale_shards([os.path.join(shard_dir, f) for f in os.listdir(shard_dir)], a.window):
            try:
                size = os.path.getsize(p)
                os.remove(p)
                freed += size
            except OSError as e:
                log(f'  impossible de supprimer {os.path.basename(p)} : {e}')
        if freed:
            log(f'  anciennes parties sorties de la fenêtre d\'entraînement supprimées : {freed / 1e9:.1f} Go libérés')

    prune()
    meta = os.path.join(engine.ROOT, 'lab', 'decks', 'meta')
    if a.warmup_games and not state.get('warmup_done') and state['iteration'] == 0:
        log(f'=== échauffement : le réseau apprend en imitant {a.warmup_player} ({a.warmup_games} parties)')
        shard = os.path.join(run_dir, 'shards', '000_warmup.bin')
        t0 = time.time()
        sp = engine.imitate(decks, a.warmup_games, shard, a.warmup_player, threads)
        log(f'  {sp["games"]} parties, {sp["samples"]} positions ({(time.time() - t0) / 60:.1f} min)')
        buf = load_buffer([shard], a.window)
        m = train(model, optimizer, buf, a.device, epochs=a.warmup_epochs, batch=a.batch, q_mix=0.0, log=log)
        del buf
        export_weights(model, latest, m['value_scale'])
        shutil.copy(latest, os.path.join(run_dir, 'weights', 'warmup.bin'))
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'vocab_table': info['vocab_table']}, ckpt)
        vs_e2 = engine.arena(engine.zspec(a.eval_sims, latest), 'e2', meta, a.eval_games, threads)['a_score']
        log(f'  après échauffement : {100 * vs_e2:.1f} % contre e2')
        state['warmup_done'] = True
        json.dump(state, open(state_f, 'w'))

    if state['iteration'] >= a.iterations:
        log(f"Ce run a déjà fait {state['iteration']} itérations (--iterations {a.iterations} demandé) : "
            f"rien à faire. Pour continuer : run.bat train --iterations {state['iteration'] + 40}")
        return
    for it in range(state['iteration'], a.iterations):
        # Cosine decay from lr to lr/10 over the run: big steps early, fine-tuning at the end.
        lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / max(a.iterations - 1, 1))))
        for g in optimizer.param_groups:
            g['lr'] = lr
        log(f'=== itération {it + 1}/{a.iterations} (taux d\'apprentissage {lr:.2e})')
        prev = latest if os.path.exists(latest) else None
        shard = os.path.join(run_dir, 'shards', f'iter_{it:03d}')
        # Leftovers of an interrupted attempt at this iteration (possibly cut mid-game, or under
        # another file name) would otherwise stay in the training window.
        for f in os.listdir(shard_dir):
            if f.startswith(f'iter_{it:03d}_'):
                os.remove(os.path.join(shard_dir, f))
        t0 = time.time()
        if prev is None:
            # No network yet: CPU self-play with the engine's hand-written evaluation.
            sp = engine.selfplay(decks, a.games, a.first_sims, shard + '_w0.bin', None, threads, seed=it)
        else:
            devices = None if a.selfplay_devices == 'auto' else a.selfplay_devices.split(',')
            deck_paths = sorted(os.path.join(decks, f) for f in os.listdir(decks) if f.endswith('.txt'))
            sp, _ = gpu_selfplay.run(deck_paths, a.games, a.sims, shard, prev, info, seed=it,
                                     devices=devices, concurrency=a.concurrency,
                                     temperature_moves=a.temperature_moves, games_dir=games_dir)
        t_sp = time.time() - t0
        log(f'  self-play : {sp["games"]} parties, {sp["samples"]} positions ({t_sp / 60:.1f} min)')
        # Highlights of this iteration's games; only the latest iteration's logs are kept.
        logs = sorted(os.path.join(games_dir, f) for f in os.listdir(games_dir) if f.endswith('.csv'))
        mine = [f for f in logs if os.path.basename(f).startswith(f'iter_{it:03d}_')]
        try:
            for line in summarize(mine, deck_labels(decks), it + 1):
                log('  ' + line)
        except Exception as e:  # a summary must never stop the training
            log(f'  (résumé indisponible : {e})')
        for f in logs:
            if f not in mine:
                os.remove(f)

        shards = sorted(os.path.join(run_dir, 'shards', f) for f in os.listdir(os.path.join(run_dir, 'shards')))
        buf = load_buffer(shards, a.window)
        t0 = time.time()
        m = train(model, optimizer, buf, a.device, epochs=a.epochs, batch=a.batch, log=log, seed=it,
                  td_lambda=a.td_lambda)
        t_tr = time.time() - t0
        del buf
        prune()
        new = os.path.join(run_dir, 'weights', f'iter_{it:03d}.bin')
        export_weights(model, new, m['value_scale'])
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'vocab_table': info['vocab_table']}, ckpt)

        vs_prev = engine.arena(engine.zspec(a.eval_sims, new), engine.zspec(a.eval_sims, prev), meta,
                               a.eval_games, threads, seed=it)['a_score']
        vs_e2 = engine.arena(engine.zspec(a.eval_sims, new), 'e2', meta, a.eval_games, threads, seed=it)['a_score']
        res_e3 = engine.arena(engine.zspec(a.eval_sims, new), 'e3', meta, a.eval_games, threads, seed=it)
        vs_e3 = res_e3['a_score']
        log(f'  nouveau réseau : {100 * vs_prev:.1f} % contre le précédent (~50 % est normal : un petit pas '
            f'par itération), {100 * vs_e2:.1f} % contre e2, {100 * vs_e3:.1f} % contre e3')
        for line in records(metrics_f, vs_e3, res_e3.get('a_streak')):
            log('  ' + line)
        shutil.copy(new, latest)

        rmse = ''
        if a.validate_every and (it + 1) % a.validate_every == 0:
            log('  validation contre les vrais matchups Limitless :')
            rmse = validate(engine.zspec(a.eval_sims, latest), a.validate_games, threads, log)
            log(f'  écart moyen (RMSE) : {rmse:.1f} points (référence e3 : 11-12)')
        with open(metrics_f, 'a', newline='') as mf:
            csv.writer(mf).writerow([it + 1, m['samples'], f'{m["policy_loss"]:.4f}', f'{m["value_loss"]:.4f}',
                                     f'{m.get("val_policy_loss", 0):.4f}', f'{m.get("val_value_loss", 0):.4f}',
                                     f'{vs_prev:.3f}', f'{vs_e2:.3f}', f'{vs_e3:.3f}',
                                     f'{rmse:.1f}' if rmse != '' else '', f'{lr:.2e}',
                                     f'{t_sp:.0f}', f'{t_tr:.0f}'])
        # One iteration moves the network only a little, and 400 games measure +-5 %: the
        # trend over several iterations is what shows progress.
        cur, old = trend(metrics_f, 'vs_e3')
        if cur is not None:
            msg = f'  tendance : {100 * cur:.1f} % contre e3 en moyenne sur les 5 dernières itérations'
            if old is not None:
                msg += f' (5 précédentes : {100 * old:.1f} %)'
            log(msg)
        state['iteration'] = it + 1
        json.dump(state, open(state_f, 'w'))
    log('terminé. Réseau final :', latest)


if __name__ == '__main__':
    main()
