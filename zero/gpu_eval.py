"""Evaluation games on the GPUs (forge, ranked): the batched engine of the self-play, with
explicit games (deck a, deck b, seed) instead of random ones, no exploration, no training data.

One process per GPU stays up with the network loaded; each job is split between them. The
network evaluates thousands of positions per batch, so a z128 game costs about what a z8 game
costs on the CPU (where deckgym simulate evaluates one position at a time).
"""
import csv
import os
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.multiprocessing as mp


def available():
    try:
        from deckgym.deckgym import BatchSelfPlay  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available() or bool(os.environ.get('PZ_EVAL_DEVICES'))


def _play(BatchSelfPlay, model, info, device, deck_paths, pairs, sims, concurrency):
    """Plays `pairs` (list of (a, b, seed)), returns one result per pair: 1 a wins, 0 b wins, 0.5 tie."""
    tmp = tempfile.mkdtemp(prefix='pz_eval_')
    # Two halves: the engine works on one while the GPU evaluates the other.
    halves = [pairs[:len(pairs) - len(pairs) // 2], pairs[len(pairs) - len(pairs) // 2:]]
    halves = [h for h in halves if h]
    logs = [os.path.join(tmp, f'h{k}.csv') for k in range(len(halves))]
    envs = [BatchSelfPlay(deck_paths, len(h), sims, '', 0, max(1, min(concurrency // len(halves), len(h))), 0, 0.0,
                          games_log=logs[k], pairs=[tuple(map(int, p)) for p in h]) for k, h in enumerate(halves)]

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

    with torch.inference_mode(), ThreadPoolExecutor(max_workers=1) as engine_thread:
        pending = {k: engine_thread.submit(e.collect) for k, e in enumerate(envs)}
        while pending:
            for k in list(pending):
                batch = pending.pop(k).result()
                if batch[0] == 0:
                    continue
                envs[k].respond(*evaluate(batch))
                pending[k] = engine_thread.submit(envs[k].collect)
    del envs  # flushes the logs
    out = []
    for k, h in enumerate(halves):
        res = [0.5] * len(h)
        with open(logs[k], newline='') as f:
            for r in csv.DictReader(f):
                w = int(r['winner'])
                res[int(r['game'])] = 1.0 if w == 0 else 0.0 if w == 1 else 0.5
        out += res
        os.remove(logs[k])
    os.rmdir(tmp)
    return out


def _worker(device, weights, info, threads, concurrency, jobs, results):
    os.environ['RAYON_NUM_THREADS'] = str(threads)
    from deckgym.deckgym import BatchSelfPlay

    from .model import ZeroNet
    from .train import load_weights
    warnings.filterwarnings('ignore', message='The given NumPy array is not writable')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = ZeroNet(info)
    load_weights(model, weights)
    model.to(device).eval()
    while True:
        job = jobs.get()
        if job is None:
            return
        jid, deck_paths, pairs, sims = job
        try:
            results.put((jid, _play(BatchSelfPlay, model, info, device, deck_paths, pairs, sims, concurrency)))
        except Exception as e:  # reported to the caller instead of hanging it
            results.put((jid, e))


class GpuJudge:
    """Persistent GPU workers: play(deck_paths, pairs, sims) -> results, in the order of pairs."""

    def __init__(self, weights, info, devices=None, concurrency=2048):
        env = os.environ.get('PZ_EVAL_DEVICES')  # e.g. "cuda:0,cuda:1" or "cpu" (tests)
        self.devices = devices or (env.split(',') if env else None) or \
            [f'cuda:{i}' for i in range(torch.cuda.device_count())] or ['cpu']
        ctx = mp.get_context('spawn')
        threads = max(1, (os.cpu_count() or 4) // len(self.devices))
        self.jobs = [ctx.Queue() for _ in self.devices]
        self.results = ctx.Queue()
        self.procs = [ctx.Process(target=_worker, args=(d, weights, info, threads, concurrency, q, self.results),
                                  daemon=True) for d, q in zip(self.devices, self.jobs)]
        for p in self.procs:
            p.start()
        self.next_id = 0

    def play(self, deck_paths, pairs, sims):
        n = len(self.devices)
        bounds = [len(pairs) * i // n for i in range(n + 1)]
        ids = []
        for i in range(n):
            chunk = pairs[bounds[i]:bounds[i + 1]]
            if chunk:
                self.jobs[i].put((self.next_id, list(deck_paths), chunk, sims))
                ids.append((self.next_id, bounds[i]))
                self.next_id += 1
        got = {}
        while len(got) < len(ids):
            jid, res = self.results.get()
            if isinstance(res, Exception):
                raise RuntimeError(f'évaluation GPU en échec : {res}')
            got[jid] = res
        out = []
        for jid, _ in ids:
            out += got[jid]
        return out

    def close(self):
        for q in self.jobs:
            q.put(None)
        for p in self.procs:
            p.join(timeout=10)
