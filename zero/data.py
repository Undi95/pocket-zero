"""Reads self-play shards written by src/zero/selfplay.rs."""
import os
import struct

import numpy as np


VERSIONS = {b'PZS1': 1, b'PZS2': 2}


def shard_dtype(sd, nt, ad, at, ma, version=2):
    fields = [('dense', '<f4', (sd,)), ('tokens', '<i4', (nt,)), ('n', '<i4'),
              ('adense', '<f4', (ma, ad)), ('atok', '<i4', (ma, at)), ('pi', '<f4', (ma,)),
              ('z', '<f4'), ('q', '<f4')]
    if version >= 2:
        # actor -1 = no search value (imitation data, or a PZS1 record): no bootstrapping.
        fields += [('actor', '<i4'), ('last', '<i4')]
    return np.dtype(fields)


def read_shard(path):
    with open(path, 'rb') as f:
        head = f.read(24)
    version = VERSIONS.get(head[:4])
    if version is None:
        raise ValueError(f'{path}: not a pocket-zero shard')
    dims = struct.unpack('<5I', head[4:24])
    dt = shard_dtype(*dims, version=version)
    n = (os.path.getsize(path) - 24) // dt.itemsize
    return np.memmap(path, dtype=dt, mode='r', offset=24, shape=(n,))


def shard_len(path):
    """Records in a shard, from its header and size (no memory map, so nothing stays open)."""
    with open(path, 'rb') as f:
        head = f.read(24)
    version = VERSIONS.get(head[:4])
    if version is None or len(head) < 24:
        return 0
    dt = shard_dtype(*struct.unpack('<5I', head[4:24]), version=version)
    return (os.path.getsize(path) - 24) // dt.itemsize


def stale_shards(paths, max_samples, margin=2):
    """Shards too old to be in the training window (the newest `max_samples` positions, as
    load_buffer reads them), keeping `margin` older files in reserve."""
    paths = sorted(paths)
    total = needed = 0
    for p in reversed(paths):
        if total >= max_samples:
            break
        total += shard_len(p)
        needed += 1
    return paths[:max(0, len(paths) - needed - margin)]


def upgrade(a):
    """PZS1 records in the PZS2 layout: each one stands alone (actor -1, last 1)."""
    if 'actor' in a.dtype.names:
        return a
    out = np.zeros(len(a), dtype=np.dtype(a.dtype.descr + [('actor', '<i4'), ('last', '<i4')]))
    for name in a.dtype.names:
        out[name] = a[name]
    out['actor'] = -1
    out['last'] = 1
    return out


def widen(a, dt):
    """Records of an older input layout in the layout of `dt`: the inputs a newer layout appended
    (after the old dense features, after the old tokens) are zero, i.e. absent."""
    if a.dtype == dt:
        return a
    out = np.zeros(len(a), dtype=dt)
    for name in a.dtype.names:
        if name in ('dense', 'tokens'):
            out[name][:, :a[name].shape[1]] = a[name]
        else:
            out[name] = a[name]
    return out


def load_buffer(paths, max_samples):
    """Most recent shards first until `max_samples`, returned as one in-memory array (in the
    layout of the newest shard)."""
    chunks, total, dt = [], 0, None
    for p in reversed(paths):
        a = read_shard(p)
        take = min(len(a), max_samples - total)
        if take <= 0:
            break
        c = upgrade(np.array(a[-take:]))
        dt = dt or c.dtype
        c = widen(c, dt)
        # A shard boundary is a game boundary: an interrupted file may end mid-game, and value
        # targets must never bootstrap from another file's first position.
        c['last'][-1] = 1
        chunks.append(c)
        total += take
    if not chunks:
        raise ValueError('no samples')
    return np.concatenate(chunks[::-1])
