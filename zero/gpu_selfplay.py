"""Self-play with the network on GPU: one worker process per GPU, each driving ~1000 games in
parallel through the Rust engine (MCTS on all CPU cores) and evaluating their positions in
batches. Falls back to a single CPU worker when no GPU is available.

Each worker splits its games in two halves that take turns: while the network evaluates one
half's batch, the engine is already preparing the other half's next batch on the CPU cores
(the engine releases the GIL), so neither waits for the other."""
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.multiprocessing as mp


def shard_paths(out, games):
    """Output files of one worker: one per half (a single one for a single game)."""
    return [out] if games < 2 else [out[:-4] + 'a.bin', out[:-4] + 'b.bin']


def _worker(rank, args, results):
    (deck_paths, games, sims, out, seed, weights, info, device, concurrency, threads,
     temperature_moves, noise) = args
    # The engine's thread pool is sized from this variable on first use.
    os.environ['RAYON_NUM_THREADS'] = str(threads)
    from deckgym.deckgym import BatchSelfPlay

    from .model import ZeroNet
    from .train import load_weights

    # The batches arrive as read-only bytes; they are only read (copied to the GPU), so torch's
    # warning about non-writable arrays does not apply.
    warnings.filterwarnings('ignore', message='The given NumPy array is not writable')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = ZeroNet(info)
    load_weights(model, weights)
    model.to(device).eval()
    paths = shard_paths(out, games)
    halves = [games - games // 2, games // 2][:len(paths)]
    envs = [BatchSelfPlay(deck_paths, g, sims, p, seed * 2 + k, max(1, concurrency // len(paths)),
                          temperature_moves, noise)
            for k, (g, p) in enumerate(zip(halves, paths))]

    def evaluate(batch):
        n, d, t, ad, at, ow = batch
        dense = torch.from_numpy(np.frombuffer(d, '<f4').reshape(n, -1)).to(device, non_blocking=True)
        tokens = torch.from_numpy(np.frombuffer(t, '<i4').reshape(n, -1).astype(np.int64)).to(device)
        adense = torch.from_numpy(np.frombuffer(ad, '<f4').reshape(-1, info['action_dense'])).to(device)
        atok = torch.from_numpy(np.frombuffer(at, '<i4').reshape(-1, info['action_tokens']).astype(np.int64)).to(device)
        owner = torch.from_numpy(np.frombuffer(ow, '<i4').astype(np.int64)).to(device)
        v, logits = model.forward_flat(dense, tokens, adense, atok, owner)
        return (v.float().cpu().numpy().astype('<f4').tobytes(),
                logits.float().cpu().numpy().astype('<f4').tobytes())

    def progress():
        return [sum(x) for x in zip(*(e.progress() for e in envs))]

    last = time.time()
    t0 = last
    evals = 0
    # One background thread runs the engine (collect); the main thread runs the network and
    # hands the results back. collect calls queue up, so the engine always has a half to work on.
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=1) as engine_thread:
        pending = {k: engine_thread.submit(e.collect) for k, e in enumerate(envs)}
        while pending:
            for k in list(pending):
                batch = pending.pop(k).result()
                if batch[0] == 0:  # this half is finished
                    continue
                envs[k].respond(*evaluate(batch))
                pending[k] = engine_thread.submit(envs[k].collect)
                evals += batch[0]
            if time.time() - last > 30:
                done, target, samples, _, _ = progress()
                el = time.time() - t0
                print(f'  [{device}] {done}/{target} parties, {samples} positions, '
                      f'{evals / el:.0f} évaluations/s, {done / el * 60:.1f} parties/min', flush=True)
                last = time.time()
    results[rank] = tuple(progress())


def run(deck_paths, games, sims, out_prefix, weights, info, seed=0, devices=None, concurrency=1024,
        temperature_moves=20, noise=0.25):
    """Returns (stats dict, list of shard paths)."""
    if devices is None:
        devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())] or ['cpu']
    cores = os.cpu_count() or 4
    threads = max(1, cores // len(devices))
    per = [games // len(devices) + (1 if i < games % len(devices) else 0) for i in range(len(devices))]
    outs = [f'{out_prefix}_w{i}.bin' for i in range(len(devices))]
    shards = [p for i in range(len(devices)) for p in shard_paths(outs[i], per[i])]
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
    return {'games': tot[0], 'samples': tot[2], 'first_player_wins': tot[3], 'ties': tot[4]}, shards
