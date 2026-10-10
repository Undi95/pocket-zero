"""Training step: fit the network to self-play targets (search policy + game result)."""
import copy
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
        sel = np.sort(idx[i:i + batch])
        yield sel, buf[sel]


def value_subset(buf, frac):
    """Positions whose result trains the value head. Chosen from a hash of the position itself,
    so a position keeps the same draw across passes and iterations (the window slides)."""
    h = np.zeros(len(buf), dtype=np.uint64)
    for i in range(0, len(buf), 65536):
        d = np.ascontiguousarray(buf['dense'][i:i + 65536]).view(np.uint32).astype(np.uint64)
        h[i:i + 65536] = (d * np.arange(1, d.shape[1] + 1, dtype=np.uint64)).sum(1)
    h = (h * np.uint64(2654435761)) % np.uint64(1 << 32)
    return h.astype(np.float64) / float(1 << 32) < frac


def td_targets(buf, lam):
    """TD(lambda) value targets. Each position leans on the search values of the positions that
    follow it in the same game and ends on the result:
        G_t = s * ((1 - lam) * q_{t+1} + lam * G_{t+1}),   G_last = z,
    with s = -1 when the player to move changes. In a game decided by coin flips and draws this
    is far less noisy than the bare result. Records without a search value (actor -1:
    imitation, old shards) keep the result."""
    z, q = buf['z'].tolist(), buf['q'].tolist()
    actor, last = buf['actor'].tolist(), buf['last'].tolist()
    g = [0.0] * len(z)
    for i in range(len(z) - 1, -1, -1):
        if last[i] or actor[i] < 0 or i + 1 == len(z) or actor[i + 1] < 0:
            g[i] = z[i]
        else:
            s = 1.0 if actor[i + 1] == actor[i] else -1.0
            g[i] = s * ((1 - lam) * q[i + 1] + lam * g[i + 1])
    return np.asarray(g, dtype=np.float32)


def to_tensors(b, device):
    t = lambda x, dt: torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=dt, non_blocking=True)
    return (t(b['dense'], torch.float32), t(b['tokens'], torch.long), t(b['adense'], torch.float32),
            t(b['atok'], torch.long), t(b['n'], torch.long), t(b['pi'], torch.float32),
            t(b['z'], torch.float32), t(b['q'], torch.float32))


def losses(model, b, device, arange, q_mix, vmask=None):
    dense, tokens, adense, atok, n, pi, z, q = to_tensors(b, device)
    value, logits = model(dense, tokens, adense, atok)
    mask = arange.unsqueeze(0) < n.unsqueeze(1)
    logits = logits.masked_fill(~mask, -1e9)
    policy_loss = -(pi * F.log_softmax(logits, -1)).sum(-1).mean()
    target = (1 - q_mix) * z + q_mix * q
    if vmask is None:
        return policy_loss, F.mse_loss(value, target)
    m = torch.from_numpy(vmask).to(device=device, dtype=torch.float32)
    return policy_loss, ((value - target) ** 2 * m).sum() / m.sum().clamp(min=1.0)


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


def calibrate_value(model, val, z, device, batch=2048):
    """Temperature scaling of the value head on held-out games. A network that has seen the
    results of its training games is too sure of itself on new positions (it says +-0.9 where
    +-0.6 is right), which skews every average the search takes over coin flips and hidden
    cards. Fits v = tanh(a * logit + b) to the real results of the held-out games. Returns
    (a, b, mse_before, mse_after); the caller folds a, b into the exported weights
    (export_weights(..., value_scale=(a, b))), training goes on with the raw model."""
    model.eval()
    xs = []
    with torch.no_grad():
        for i in range(0, len(val), batch):
            dense, tokens, *_ = to_tensors(val[i:i + batch], device)
            h = model.trunk(dense, tokens)
            xs.append(model.v2(F.relu(model.v1(h))).squeeze(-1).float().cpu())
    x = torch.cat(xs)
    y = torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32))
    a = torch.ones(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], max_iter=100, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = ((torch.tanh(a * x + b) - y) ** 2).mean()
        loss.backward()
        return loss

    opt.step(closure)
    a_, b_ = float(a.item()), float(b.item())
    before = float(((torch.tanh(x) - y) ** 2).mean())
    if not (0.02 < a_ < 5.0) or not np.isfinite(b_):
        return 1.0, 0.0, before, before
    after = float(((torch.tanh(a_ * x + b_) - y) ** 2).mean())
    return a_, b_, before, after


def train(model, optimizer, buf, device, epochs=2, batch=1024, q_mix=0.25, log=print, seed=0,
          holdout=0.03, value_frac=0.2, td_lambda=0.0, calibrate=True):
    """`q_mix` blends the search value into the value target (helps early, when results are noisy).
    The most recent `holdout` fraction of positions is kept out of training to measure
    generalisation (a value loss far below the held-out one means memorised games).
    Only a fixed `value_frac` of positions train the value head: positions of one game share its
    result and look alike, so using them all teaches the network to recognise games rather than
    to judge positions (the reason AlphaGo trained its value net on one position per game).
    `td_lambda` > 0 replaces the result by TD(lambda) targets (see td_targets; q_mix is then
    ignored, the search values being already part of the target).
    `calibrate`: fit the value calibration on the held-out games (see calibrate_value), returned
    as out['value_scale'] for export_weights."""
    results = buf['z'].copy()
    if td_lambda > 0 and 'actor' in buf.dtype.names:
        buf['z'] = td_targets(buf, td_lambda)
        q_mix = 0.0
    n_val = int(len(buf) * holdout) if len(buf) > 2000 else 0
    val, buf = (buf[len(buf) - n_val:], buf[:len(buf) - n_val]) if n_val else (None, buf)
    rng = np.random.default_rng(seed)
    keep = value_subset(buf, value_frac)
    ma = buf['pi'].shape[1]
    arange = torch.arange(ma, device=device)
    stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'steps': 0}
    best = (float('inf'), 0, None)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for sel, b in batches(buf, min(batch, len(buf)), rng):
            policy_loss, value_loss = losses(model, b, device, arange, q_mix, keep[sel])
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
        msg = (f'  passe {epoch + 1}/{epochs} sur les données : politique {stats["policy_loss"] / s:.3f}, '
               f'valeur {stats["value_loss"] / s:.3f}')
        if val is not None:
            vp, vv = evaluate_losses(model, val, device, q_mix)
            msg += f' | parties de contrôle : politique {vp:.3f}, valeur {vv:.3f}'
            # Early stopping: keep the epoch that does best on games never trained on.
            if vp + vv < best[0]:
                best = (vp + vv, epoch + 1, copy.deepcopy(model.state_dict()))
        log(msg + f' ({time.time() - t0:.0f} s)')
        stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'steps': 0} if epoch + 1 < epochs else stats
    if best[2] is not None and best[1] < epochs:
        model.load_state_dict(best[2])
        log(f'  la passe {best[1]}/{epochs} généralisait mieux sur les parties de contrôle : '
            f'on garde le réseau de cette passe')
    s = max(stats['steps'], 1)
    out = {'policy_loss': stats['policy_loss'] / s, 'value_loss': stats['value_loss'] / s,
           'samples': len(buf)}
    out['value_scale'] = (1.0, 0.0)
    if val is not None and calibrate:
        a, b, before, after = calibrate_value(model, val, results[len(results) - n_val:], device)
        out.update(value_scale=(a, b), val_mse_raw=before, val_mse_cal=after)
        log(f'  calibration de la valeur sur les parties de contrôle : échelle {a:.2f} '
            f'(erreur contre les vrais résultats {before:.3f} -> {after:.3f})')
    if val is not None:
        out['val_policy_loss'], out['val_value_loss'] = evaluate_losses(model, val, device, q_mix)
    return out
