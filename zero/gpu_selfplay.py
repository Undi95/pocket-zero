"""Self-play with the network on GPU: one worker process per GPU, each driving ~1000 games in
parallel through the Rust engine (MCTS on all CPU cores) and evaluating their positions in
batches. Falls back to a single CPU worker when no GPU is available."""
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp


def _worker(rank, args, results):
    (deck_paths, games, sims, out, seed, weights, info, device, concurrency, threads,
     temperature_moves, noise) = args
    # The engine's thread pool is sized from this variable on first use.
    os.environ['RAYON_NUM_THREADS'] = str(threads)
    from deckgym.deckgym import BatchSelfPlay

    from .model import ZeroNet
    from .train import load_weights

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = ZeroNet(info)
    load_weights(model, weights)
    model.to(device).eval()
    env = BatchSelfPlay(deck_paths, games, sims, out, seed, concurrency, temperature_moves, noise)
    last = time.time()
    t0 = last
    evals = 0
    with torch.inference_mode():
        while not env.done():
            n, d, t, ad, at, ow = env.collect()
            if n == 0:
                break
            dense = torch.from_numpy(np.frombuffer(d, '<f4').reshape(n, -1)).to(device, non_blocking=True)
            tokens = torch.from_numpy(np.frombuffer(t, '<i4').reshape(n, -1).astype(np.int64)).to(device)
            adense = torch.from_numpy(np.frombuffer(ad, '<f4').reshape(-1, info['action_dense'])).to(device)
            atok = torch.from_numpy(np.frombuffer(at, '<i4').reshape(-1, info['action_tokens']).astype(np.int64)).to(device)
            owner = torch.from_numpy(np.frombuffer(ow, '<i4').astype(np.int64)).to(device)
            v, logits = model.forward_flat(dense, tokens, adense, atok, owner)
            env.respond(v.float().cpu().numpy().astype('<f4').tobytes(),
                        logits.float().cpu().numpy().astype('<f4').tobytes())
            evals += n
            if time.time() - last > 30:
                done, target, samples, _, _ = env.progress()
                el = time.time() - t0
                print(f'  [{device}] {done}/{target} parties, {samples} positions, '
                      f'{evals / el:.0f} évaluations/s, {done / el * 60:.1f} parties/min', flush=True)
                last = time.time()
    results[rank] = env.progress()


def run(deck_paths, games, sims, out_prefix, weights, info, seed=0, devices=None, concurrency=1024,
        temperature_moves=20, noise=0.25):
    """Returns (stats dict, list of shard paths)."""
    if devices is None:
        devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())] or ['cpu']
    cores = os.cpu_count() or 4
    threads = max(1, cores // len(devices))
    per = [games // len(devices) + (1 if i < games % len(devices) else 0) for i in range(len(devices))]
    outs = [f'{out_prefix}_w{i}.bin' for i in range(len(devices))]
    manager = mp.get_context('spawn').Manager()
    results = manager.dict()
    jobs = [(deck_paths, per[i], sims, outs[i], seed * 100 + i, weights, info, devices[i], concurrency,
             threads, temperature_moves, noise) for i in range(len(devices))]
    if len(devices) == 1:
        _worker(0, jobs[0], results)
    else:
        ctx = mp.get_context('spawn')
        procs = [ctx.Process(target=_worker, args=(i, jobs[i], results)) for i in range(len(devices))]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f'worker de self-play en échec (code {p.exitcode})')
    tot = [sum(results[i][k] for i in range(len(devices))) for k in range(5)]
    return {'games': tot[0], 'samples': tot[2], 'first_player_wins': tot[3], 'ties': tot[4]}, outs
