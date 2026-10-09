"""Training step: fit the network to self-play targets (search policy + game result)."""
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F


def load_weights(model, path):
    """Loads a PZW1 file (as written by export_weights) into `model`."""
    sd = {}
    with open(path, 'rb') as f:
        assert f.read(4) == b'PZW1', f'{path}: not a pocket-zero weight file'
        (n,) = struct.unpack('<I', f.read(4))
        for _ in range(n):
            (ln,) = struct.unpack('<I', f.read(4))
            name = f.read(ln).decode()
            (nd,) = struct.unpack('<I', f.read(4))
            dims = struct.unpack(f'<{nd}I', f.read(4 * nd))
            count = int(np.prod(dims))
            sd[name] = torch.from_numpy(np.frombuffer(f.read(4 * count), dtype='<f4').reshape(dims).copy())
    model.load_state_dict(sd)


def batches(buf, batch, rng):
    idx = rng.permutation(len(buf))
    for i in range(0, len(idx) - batch + 1, batch):
        yield buf[np.sort(idx[i:i + batch])]


def to_tensors(b, device):
    t = lambda x, dt: torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=dt, non_blocking=True)
    return (t(b['dense'], torch.float32), t(b['tokens'], torch.long), t(b['adense'], torch.float32),
            t(b['atok'], torch.long), t(b['n'], torch.long), t(b['pi'], torch.float32),
            t(b['z'], torch.float32), t(b['q'], torch.float32))


def losses(model, b, device, arange, q_mix):
    dense, tokens, adense, atok, n, pi, z, q = to_tensors(b, device)
    value, logits = model(dense, tokens, adense, atok)
    mask = arange.unsqueeze(0) < n.unsqueeze(1)
    logits = logits.masked_fill(~mask, -1e9)
    policy_loss = -(pi * F.log_softmax(logits, -1)).sum(-1).mean()
    target = (1 - q_mix) * z + q_mix * q
    return policy_loss, F.mse_loss(value, target)


def evaluate_losses(model, buf, device, q_mix, batch=2048):
    model.eval()
    arange = torch.arange(buf['pi'].shape[1], device=device)
    tot_p = tot_v = 0.0
    with torch.no_grad():
        for i in range(0, len(buf), batch):
            p, v = losses(model, buf[i:i + batch], device, arange, q_mix)
            k = min(batch, len(buf) - i)
            tot_p += p.item() * k
            tot_v += v.item() * k
    return tot_p / max(len(buf), 1), tot_v / max(len(buf), 1)


def train(model, optimizer, buf, device, epochs=2, batch=1024, q_mix=0.25, log=print, seed=0,
          holdout=0.03):
    """`q_mix` blends the search value into the value target (helps early, when results are noisy).
    The most recent `holdout` fraction of positions is kept out of training to measure
    generalisation (a value loss far below the held-out one means memorised games)."""
    n_val = int(len(buf) * holdout) if len(buf) > 2000 else 0
    val, buf = (buf[len(buf) - n_val:], buf[:len(buf) - n_val]) if n_val else (None, buf)
    rng = np.random.default_rng(seed)
    ma = buf['pi'].shape[1]
    arange = torch.arange(ma, device=device)
    stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'steps': 0}
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for b in batches(buf, min(batch, len(buf)), rng):
            policy_loss, value_loss = losses(model, b, device, arange, q_mix)
            loss = policy_loss + value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():  # keep the padding embedding at zero, as Rust assumes
                model.card_emb.weight[0].zero_()
            stats['policy_loss'] += policy_loss.item()
            stats['value_loss'] += value_loss.item()
            stats['steps'] += 1
        s = max(stats['steps'], 1)
        msg = (f'  epoch {epoch + 1}/{epochs}: politique {stats["policy_loss"] / s:.3f}, '
               f'valeur {stats["value_loss"] / s:.3f}')
        if val is not None:
            vp, vv = evaluate_losses(model, val, device, q_mix)
            msg += f' | parties de contrôle : politique {vp:.3f}, valeur {vv:.3f}'
        log(msg + f' ({time.time() - t0:.0f} s)')
        stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'steps': 0} if epoch + 1 < epochs else stats
    s = max(stats['steps'], 1)
    out = {'policy_loss': stats['policy_loss'] / s, 'value_loss': stats['value_loss'] / s,
           'samples': len(buf)}
    if val is not None:
        out['val_policy_loss'], out['val_value_loss'] = evaluate_losses(model, val, device, q_mix)
    return out
