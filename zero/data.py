"""Reads self-play shards written by src/zero/selfplay.rs."""
import os
import struct

import numpy as np


def shard_dtype(sd, nt, ad, at, ma):
    return np.dtype([('dense', '<f4', (sd,)), ('tokens', '<i4', (nt,)), ('n', '<i4'),
                     ('adense', '<f4', (ma, ad)), ('atok', '<i4', (ma, at)), ('pi', '<f4', (ma,)),
                     ('z', '<f4'), ('q', '<f4')])


def read_shard(path):
    with open(path, 'rb') as f:
        head = f.read(24)
    if head[:4] != b'PZS1':
        raise ValueError(f'{path}: not a pocket-zero shard')
    dims = struct.unpack('<5I', head[4:24])
    dt = shard_dtype(*dims)
    n = (os.path.getsize(path) - 24) // dt.itemsize
    return np.memmap(path, dtype=dt, mode='r', offset=24, shape=(n,))


def load_buffer(paths, max_samples):
    """Most recent shards first until `max_samples`, returned as one in-memory array."""
    chunks, total = [], 0
    for p in reversed(paths):
        a = read_shard(p)
        take = min(len(a), max_samples - total)
        if take <= 0:
            break
        chunks.append(np.array(a[-take:]))
        total += take
    if not chunks:
        raise ValueError('no samples')
    return np.concatenate(chunks[::-1])
