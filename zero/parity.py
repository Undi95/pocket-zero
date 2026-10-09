"""Checks that the Rust network (used for play) and the PyTorch one (used for training) agree.

    python -m zero.parity --shard runs/x/shards/iter_000.bin [--weights w.bin]

Without --weights a random network is exported first. Exits non-zero on mismatch."""
import argparse
import os
import sys
import tempfile

import numpy as np
import torch

from . import engine
from .data import read_shard
from .model import ZeroNet, export_weights


def check(shard, weights=None, n=16, tol=1e-3):
    info = engine.info()
    model = ZeroNet(info)
    tmp = None
    if weights is None:
        torch.manual_seed(0)
        for p in model.parameters():  # larger than default init so differences would show
            torch.nn.init.normal_(p, std=0.05)
        tmp = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
        tmp.close()
        weights = tmp.name
        export_weights(model, weights)
    else:
        from .train import load_weights
        load_weights(model, weights)
    model.eval()
    rust = engine.parity(weights, shard, n)
    data = read_shard(shard)[:len(rust)]
    with torch.no_grad():
        v, logits = model(torch.from_numpy(np.array(data['dense'])),
                          torch.from_numpy(np.array(data['tokens'])).long(),
                          torch.from_numpy(np.array(data['adense'])),
                          torch.from_numpy(np.array(data['atok'])).long())
    worst = 0.0
    for i, r in enumerate(rust):
        k = int(data['n'][i])
        worst = max(worst, abs(r['value'] - float(v[i])))
        worst = max(worst, float(np.abs(np.array(r['logits'][:k]) - logits[i, :k].numpy()).max()))
    if tmp:
        os.unlink(tmp.name)
    return worst, worst <= tol


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', required=True)
    ap.add_argument('--weights')
    a = ap.parse_args()
    worst, ok = check(a.shard, a.weights)
    print(f'écart max Rust/PyTorch : {worst:.2e} -> {"OK" if ok else "ÉCHEC"}')
    sys.exit(0 if ok else 1)
