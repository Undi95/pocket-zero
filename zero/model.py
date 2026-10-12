"""PyTorch twin of src/zero/net.rs. Layer names and shapes must match the Rust side exactly;
`python -m zero.parity` checks both networks give the same outputs."""
import os
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Network widths. 'small' is the original network; 'big' has ~3.5x the parameters (wider
# layers, 64-wide card embeddings): more capacity for 4,300 cards and technical decks.
NET_SIZES = {
    'small': {'embed_dim': 32, 'action_embed': 16, 'h1': 512, 'h2': 512, 'h': 256, 'vh': 64, 'ph': 128},
    'big': {'embed_dim': 64, 'action_embed': 32, 'h1': 1024, 'h2': 1024, 'h': 512, 'vh': 128, 'ph': 256},
}


def sized(info, sizes):
    """`info` (from pocket-zero info) with the network widths of `sizes`."""
    out = dict(info, **sizes)
    e, ae, n = out['embed_dim'], out['action_embed'], info['n_slots']
    out['trunk_in'] = info['state_dense'] + (n + 1) * e + 4 * e + n * e
    out['action_in'] = info['action_dense'] + 2 * ae + e
    return out


def weight_sizes(path):
    """The widths of a PZW1 weight file, read from its tensor shapes."""
    shapes = {}
    with open(path, 'rb') as f:
        assert f.read(4) == b'PZW1', f'{path}: not a pocket-zero weight file'
        (n,) = struct.unpack('<I', f.read(4))
        for _ in range(n):
            (ln,) = struct.unpack('<I', f.read(4))
            name = f.read(ln).decode()
            (nd,) = struct.unpack('<I', f.read(4))
            dims = struct.unpack(f'<{nd}I', f.read(4 * nd))
            shapes[name] = dims
            f.seek(4 * int(np.prod(dims)), 1)
    return {'embed_dim': shapes['card_emb.weight'][1], 'action_embed': shapes['type_emb.weight'][1],
            'h1': shapes['fc1.weight'][0], 'h2': shapes['fc2.weight'][0], 'h': shapes['fc3.weight'][0],
            'vh': shapes['v1.weight'][0], 'ph': shapes['p_state.weight'][0]}


class ZeroNet(nn.Module):
    def __init__(self, info):
        super().__init__()
        self.info = info
        e, ae = info['embed_dim'], info['action_embed']
        self.card_emb = nn.Embedding(info['vocab'], e, padding_idx=0)
        self.type_emb = nn.Embedding(info['action_type_buckets'], ae)
        self.detail_emb = nn.Embedding(info['action_detail_buckets'], ae)
        self.fc1 = nn.Linear(info['trunk_in'], info['h1'])
        self.fc2 = nn.Linear(info['h1'], info['h2'])
        self.fc3 = nn.Linear(info['h2'], info['h'])
        self.v1 = nn.Linear(info['h'], info['vh'])
        self.v2 = nn.Linear(info['vh'], 1)
        self.p_state = nn.Linear(info['h'], info['ph'])
        self.p_action = nn.Linear(info['action_in'], info['ph'])
        self.p_out = nn.Linear(info['ph'], 1)
        # Training-only regularisation (inactive in eval mode, so Rust needs no counterpart).
        self.dropout = nn.Dropout(0.15)
        n_slots = info['n_slots']
        self.slot_tokens = n_slots + 1
        self.bags = [tuple(b) for b in info['bags']]
        # Layout 2: the cards attached to each slot (Tools, cards under it), summed per slot.
        self.attached = (info['attached_start'], n_slots, info['attached_max'])

    def trunk(self, dense, tokens):
        b = dense.shape[0]
        parts = [dense, self.card_emb(tokens[:, :self.slot_tokens]).reshape(b, -1)]
        for start, length in self.bags:
            t = tokens[:, start:start + length]
            mask = (t != 0).float().unsqueeze(-1)
            s = (self.card_emb(t) * mask).sum(1)
            parts.append(s / mask.sum(1).clamp(min=1.0))
        start, n_slots, per = self.attached
        t = tokens[:, start:start + n_slots * per]
        m = (t != 0).float().unsqueeze(-1)  # empty places add nothing (as in net.rs)
        parts.append((self.card_emb(t) * m).reshape(b, n_slots, per, -1).sum(2).reshape(b, -1))
        x = torch.cat(parts, 1)
        h = self.dropout(F.relu(self.fc1(x)))
        h = self.dropout(F.relu(self.fc2(h)))
        return F.relu(self.fc3(h))

    def forward(self, dense, tokens, adense, atok):
        """dense [B,SD], tokens [B,NT], adense [B,A,AD], atok [B,A,3] -> value [B], logits [B,A]"""
        h = self.trunk(dense, tokens)
        value = torch.tanh(self.v2(F.relu(self.v1(h)))).squeeze(-1)
        a_in = torch.cat([adense, self.type_emb(atok[..., 0]), self.card_emb(atok[..., 1]),
                          self.detail_emb(atok[..., 2])], -1)
        q = F.relu(self.p_action(a_in) + self.p_state(h).unsqueeze(1))
        logits = self.p_out(q).squeeze(-1)
        return value, logits

    def forward_flat(self, dense, tokens, adense, atok, owner):
        """Ragged version for self-play batches: actions of all requests concatenated, `owner`
        gives each action's request index. Returns value [B], logits [total actions]."""
        h = self.trunk(dense, tokens)
        value = torch.tanh(self.v2(F.relu(self.v1(h)))).squeeze(-1)
        a_in = torch.cat([adense, self.type_emb(atok[:, 0]), self.card_emb(atok[:, 1]),
                          self.detail_emb(atok[:, 2])], -1)
        q = F.relu(self.p_action(a_in) + self.p_state(h)[owner])
        return value, self.p_out(q).squeeze(-1)


def migrate_state_dict(sd, info, table=None):
    """Brings weights of another vocabulary or of input layout 1 to the current ones, like
    Net::load (src/zero/net.rs): embedding rows follow their card ids (`table` is the
    "id<TAB>row" text the weights were trained with, layout-1 networks used vocab_v1.txt), and a
    layout-1 fc1 gets zero columns for the layout-2 inputs. `sd` holds tensors or arrays."""
    if table is None:
        with open(os.path.join(os.path.dirname(__file__), '..', 'src', 'zero', 'vocab_v1.txt')) as f:
            table = f.read()
    old = dict((i, int(r)) for i, r in (l.split('\t') for l in table.splitlines() if '\t' in l))
    new = dict((i, int(r)) for i, r in (l.split('\t') for l in info['vocab_table'].splitlines() if '\t' in l))
    emb = sd['card_emb.weight']
    if old != new or emb.shape[0] != info['vocab']:
        out = emb.new_zeros((info['vocab'], emb.shape[1])) if torch.is_tensor(emb) else \
            np.zeros((info['vocab'], emb.shape[1]), dtype=emb.dtype)
        out[0] = emb[0]  # padding
        done = set()
        for cid, row in new.items():
            o = old.get(cid)
            if o is not None and o < emb.shape[0] and row not in done:
                out[row] = emb[o]
                done.add(row)
        sd['card_emb.weight'] = out
    w = sd['fc1.weight']
    if w.shape[1] == info['trunk_in_v1'] != info['trunk_in']:
        d1, d2 = info['state_dense_v1'], info['state_dense']
        tail = info['trunk_in'] - (w.shape[1] + d2 - d1)
        cat = torch.cat if torch.is_tensor(w) else np.concatenate
        zeros = (lambda n: w.new_zeros((w.shape[0], n))) if torch.is_tensor(w) else \
            (lambda n: np.zeros((w.shape[0], n), dtype=w.dtype))
        sd['fc1.weight'] = cat([w[:, :d1], zeros(d2 - d1), w[:, d1:], zeros(tail)], 1)
    return sd


def export_weights(model, path, value_scale=(1.0, 0.0)):
    """Writes the PZW1 format read by Net::load (see src/zero/net.rs). `value_scale` (a, b)
    turns the value into tanh(a * logit + b) in the exported file only (calibration, see
    zero/train.py calibrate_value); the model itself is left as trained."""
    sd = {k: v.detach().float().cpu().numpy().copy() for k, v in model.state_dict().items()}
    a, b = value_scale
    sd['v2.weight'] = sd['v2.weight'] * a
    sd['v2.bias'] = sd['v2.bias'] * a + b
    sd['vocab.table'] = np.frombuffer(model.info['vocab_table'].encode(), dtype=np.uint8).astype('<f4')
    with open(path, 'wb') as f:
        f.write(b'PZW1')
        f.write(struct.pack('<I', len(sd)))
        for name, arr in sd.items():
            arr = np.ascontiguousarray(arr, dtype='<f4')
            nb = name.encode()
            f.write(struct.pack('<I', len(nb)))
            f.write(nb)
            f.write(struct.pack('<I', arr.ndim))
            f.write(struct.pack(f'<{arr.ndim}I', *arr.shape))
            f.write(arr.tobytes())
