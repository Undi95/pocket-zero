//! The value/policy network, evaluated on CPU. `zero/model.py` defines the same network in
//! PyTorch and writes its weights in the format read by `Net::load`; `pocket-zero parity`
//! checks both give the same outputs.
//!
//! Weight file: b"PZW1", u32 tensor count, then per tensor: u32 name length, name (utf-8),
//! u32 ndims, ndims x u32 dims, f32 data (row-major). All little-endian.
//!
//! Besides the layers, a file carries `vocab.table`: the text "card id<TAB>embedding row" per
//! line, one byte per f32. Loading maps every row to the current vocabulary by card id, so a
//! network survives card additions and data fixes (new cards start from a zero embedding).
//! Files without it predate layout 2 and use `vocab_v1.txt`. A layout-1 network (smaller
//! `fc1`) gets zero weights on the layout-2 inputs: it plays as before.

use std::collections::HashMap;
use std::io::{Read, Write};

use super::encode::{
    ActionFeatures, Obs, ACTION_DENSE, ATTACHED_MAX, ATTACHED_START, BAGS, EMBED_DIM, N_SLOTS,
    STATE_DENSE, STATE_DENSE_V1, VOCAB,
};

pub const ACTION_EMBED: usize = 16;
pub const H1: usize = 512;
pub const H2: usize = 512;
pub const H: usize = 256;
pub const VH: usize = 64;
pub const PH: usize = 128;
/// fc1's input: dense features, slot and stadium cards, the four bags, then (layout 2) the sum of
/// the cards attached to each slot.
pub const TRUNK_IN: usize =
    STATE_DENSE + (N_SLOTS + 1) * EMBED_DIM + 4 * EMBED_DIM + N_SLOTS * EMBED_DIM;
pub const TRUNK_IN_V1: usize = STATE_DENSE_V1 + (N_SLOTS + 1) * EMBED_DIM + 4 * EMBED_DIM;
/// Vocabulary of the layout-1 networks (they did not store theirs).
pub const VOCAB_V1: &str = include_str!("vocab_v1.txt");
pub const ACTION_IN: usize = ACTION_DENSE + ACTION_EMBED + EMBED_DIM + ACTION_EMBED;

pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

pub fn read_tensors(path: &str) -> std::io::Result<Tensors> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"PZW1" {
        return Err(std::io::Error::other("not a pocket-zero weight file"));
    }
    let mut u32buf = [0u8; 4];
    let mut read_u32 = |f: &mut dyn Read| -> std::io::Result<u32> {
        f.read_exact(&mut u32buf)?;
        Ok(u32::from_le_bytes(u32buf))
    };
    let n = read_u32(&mut f)?;
    let mut out = HashMap::new();
    for _ in 0..n {
        let len = read_u32(&mut f)? as usize;
        let mut name = vec![0u8; len];
        f.read_exact(&mut name)?;
        let nd = read_u32(&mut f)? as usize;
        let mut dims = Vec::with_capacity(nd);
        for _ in 0..nd {
            dims.push(read_u32(&mut f)? as usize);
        }
        let count: usize = dims.iter().product();
        let mut bytes = vec![0u8; count * 4];
        f.read_exact(&mut bytes)?;
        let data = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        out.insert(String::from_utf8_lossy(&name).into_owned(), (dims, data));
    }
    Ok(out)
}

pub fn write_tensors(
    path: &str,
    tensors: &[(String, Vec<usize>, Vec<f32>)],
) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"PZW1")?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    for (name, dims, data) in tensors {
        f.write_all(&(name.len() as u32).to_le_bytes())?;
        f.write_all(name.as_bytes())?;
        f.write_all(&(dims.len() as u32).to_le_bytes())?;
        for d in dims {
            f.write_all(&(*d as u32).to_le_bytes())?;
        }
        for x in data {
            f.write_all(&x.to_le_bytes())?;
        }
    }
    Ok(())
}

struct Linear {
    w: Vec<f32>, // [out, in]
    b: Vec<f32>,
    n_in: usize,
    n_out: usize,
}

impl Linear {
    fn from(t: &Tensors, name: &str, n_in: usize, n_out: usize) -> Result<Self, String> {
        let (wd, w) = t
            .get(&format!("{name}.weight"))
            .ok_or(format!("missing {name}.weight"))?;
        let (bd, b) = t
            .get(&format!("{name}.bias"))
            .ok_or(format!("missing {name}.bias"))?;
        if wd != &vec![n_out, n_in] || bd != &vec![n_out] {
            return Err(format!(
                "{name}: expected [{n_out}, {n_in}], got {wd:?} (did the encoding change? retrain)"
            ));
        }
        Ok(Linear {
            w: w.clone(),
            b: b.clone(),
            n_in,
            n_out,
        })
    }

    fn forward_into(&self, x: &[f32], out: &mut Vec<f32>) {
        out.clear();
        for o in 0..self.n_out {
            let row = &self.w[o * self.n_in..(o + 1) * self.n_in];
            let mut acc = self.b[o];
            for (a, b) in row.iter().zip(x) {
                acc += a * b;
            }
            out.push(acc);
        }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n_out);
        self.forward_into(x, &mut out);
        out
    }
}

fn relu(v: &mut [f32]) {
    for x in v {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

pub struct Net {
    card_emb: Vec<f32>,   // [V, E]
    type_emb: Vec<f32>,   // [ACTION_TYPE_BUCKETS, ACTION_EMBED]
    detail_emb: Vec<f32>, // [ACTION_DETAIL_BUCKETS, ACTION_EMBED]
    vocab: usize,
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    v1: Linear,
    v2: Linear,
    p_state: Linear,  // h -> PH (with bias)
    p_action: Linear, // action input -> PH
    p_out: Linear,    // PH -> 1
}

/// What the network says about one decision point.
pub struct Eval {
    /// Expected result for the player to move, in [-1, 1].
    pub value: f32,
    /// One logit per action, same order as given.
    pub logits: Vec<f32>,
}

fn table(t: &Tensors, name: &str, rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    let (d, w) = t.get(name).ok_or(format!("missing {name}"))?;
    if d != &vec![rows, cols] {
        return Err(format!("{name}: expected [{rows}, {cols}], got {d:?}"));
    }
    Ok(w.clone())
}

/// The current vocabulary as stored in weight files: "card id<TAB>row" per line, sorted by id.
pub fn vocab_table() -> String {
    let mut rows: Vec<(&String, &u32)> = VOCAB.by_id.iter().collect();
    rows.sort();
    rows.iter().map(|(id, i)| format!("{id}\t{i}\n")).collect()
}

fn parse_vocab_table(text: &str) -> HashMap<&str, usize> {
    text.lines()
        .filter_map(|l| l.split_once('\t'))
        .filter_map(|(id, i)| i.trim().parse().ok().map(|i| (id, i)))
        .collect()
}

/// `vocab.table` as a tensor (one byte per f32), to add to a weight file.
pub fn vocab_tensor() -> (String, Vec<usize>, Vec<f32>) {
    let bytes: Vec<f32> = vocab_table().bytes().map(|b| b as f32).collect();
    ("vocab.table".to_string(), vec![bytes.len()], bytes)
}

/// Brings tensors written for another vocabulary or for layout 1 to the current ones:
/// embedding rows follow their card ids, and a layout-1 fc1 gets zero columns for the inputs
/// layout 2 added (after the layout-1 dense block, and at the end).
pub fn migrate(t: &mut Tensors) -> Result<(), String> {
    let stored: Option<String> = t
        .remove("vocab.table")
        .map(|(_, d)| d.iter().map(|&x| x as u8 as char).collect());
    let old = parse_vocab_table(stored.as_deref().unwrap_or(VOCAB_V1));
    let (dims, emb) = t
        .get("card_emb.weight")
        .ok_or("missing card_emb.weight")?
        .clone();
    if dims.len() != 2 || dims[1] != EMBED_DIM {
        return Err(format!("card_emb.weight: unexpected shape {dims:?}"));
    }
    let old_rows = dims[0];
    let mut new = vec![0f32; VOCAB.size * EMBED_DIM];
    new[..EMBED_DIM].copy_from_slice(&emb[..EMBED_DIM]); // row 0: padding
    let mut seen = vec![false; VOCAB.size];
    for (id, &row) in &VOCAB.by_id {
        let row = row as usize;
        match old.get(id.as_str()) {
            Some(&o) if o < old_rows && !seen[row] => {
                new[row * EMBED_DIM..(row + 1) * EMBED_DIM]
                    .copy_from_slice(&emb[o * EMBED_DIM..(o + 1) * EMBED_DIM]);
                seen[row] = true;
            }
            _ => {}
        }
    }
    t.insert(
        "card_emb.weight".to_string(),
        (vec![VOCAB.size, EMBED_DIM], new),
    );
    if let Some((d, w)) = t.get("fc1.weight").cloned() {
        if d.len() == 2 && d[1] == TRUNK_IN_V1 {
            let mut out = Vec::with_capacity(d[0] * TRUNK_IN);
            for row in w.chunks(TRUNK_IN_V1) {
                out.extend_from_slice(&row[..STATE_DENSE_V1]);
                out.resize(out.len() + STATE_DENSE - STATE_DENSE_V1, 0.0);
                out.extend_from_slice(&row[STATE_DENSE_V1..]);
                out.resize(out.len() + N_SLOTS * EMBED_DIM, 0.0);
            }
            t.insert("fc1.weight".to_string(), (vec![d[0], TRUNK_IN], out));
        }
    }
    Ok(())
}

impl Net {
    pub fn load(path: &str) -> Result<Net, String> {
        let mut t = read_tensors(path).map_err(|e| format!("{path}: {e}"))?;
        migrate(&mut t).map_err(|e| format!("{path}: {e}"))?;
        let v = VOCAB.size;
        Ok(Net {
            card_emb: table(&t, "card_emb.weight", v, EMBED_DIM)?,
            type_emb: table(
                &t,
                "type_emb.weight",
                super::encode::ACTION_TYPE_BUCKETS,
                ACTION_EMBED,
            )?,
            detail_emb: table(
                &t,
                "detail_emb.weight",
                super::encode::ACTION_DETAIL_BUCKETS,
                ACTION_EMBED,
            )?,
            vocab: v,
            fc1: Linear::from(&t, "fc1", TRUNK_IN, H1)?,
            fc2: Linear::from(&t, "fc2", H1, H2)?,
            fc3: Linear::from(&t, "fc3", H2, H)?,
            v1: Linear::from(&t, "v1", H, VH)?,
            v2: Linear::from(&t, "v2", VH, 1)?,
            p_state: Linear::from(&t, "p_state", H, PH)?,
            p_action: Linear::from(&t, "p_action", ACTION_IN, PH)?,
            p_out: Linear::from(&t, "p_out", PH, 1)?,
        })
    }

    fn emb<'a>(&'a self, table: &'a [f32], idx: usize, dim: usize) -> &'a [f32] {
        &table[idx * dim..(idx + 1) * dim]
    }

    fn trunk(&self, obs: &Obs) -> Vec<f32> {
        let mut x = Vec::with_capacity(TRUNK_IN);
        x.extend_from_slice(&obs.dense);
        for &tok in &obs.tokens[..N_SLOTS + 1] {
            let i = (tok as usize).min(self.vocab - 1);
            x.extend_from_slice(self.emb(&self.card_emb, i, EMBED_DIM));
        }
        for (start, len) in BAGS {
            let mut acc = [0f32; EMBED_DIM];
            let mut n = 0;
            for &tok in &obs.tokens[start..start + len] {
                if tok != 0 {
                    let e = self.emb(
                        &self.card_emb,
                        (tok as usize).min(self.vocab - 1),
                        EMBED_DIM,
                    );
                    for k in 0..EMBED_DIM {
                        acc[k] += e[k];
                    }
                    n += 1;
                }
            }
            let n = n.max(1) as f32;
            x.extend(acc.iter().map(|a| a / n));
        }
        for slot in 0..N_SLOTS {
            let mut acc = [0f32; EMBED_DIM];
            let start = ATTACHED_START + slot * ATTACHED_MAX;
            for &tok in &obs.tokens[start..start + ATTACHED_MAX] {
                if tok != 0 {
                    let e = self.emb(
                        &self.card_emb,
                        (tok as usize).min(self.vocab - 1),
                        EMBED_DIM,
                    );
                    for k in 0..EMBED_DIM {
                        acc[k] += e[k];
                    }
                }
            }
            x.extend_from_slice(&acc);
        }
        debug_assert_eq!(x.len(), TRUNK_IN);
        let mut h = self.fc1.forward(&x);
        relu(&mut h);
        let mut h = self.fc2.forward(&h);
        relu(&mut h);
        let mut h = self.fc3.forward(&h);
        relu(&mut h);
        h
    }

    pub fn evaluate(&self, obs: &Obs, actions: &[ActionFeatures]) -> Eval {
        let h = self.trunk(obs);
        let mut v = self.v1.forward(&h);
        relu(&mut v);
        let value = self.v2.forward(&v)[0].tanh();

        let ps = self.p_state.forward(&h);
        let mut logits = Vec::with_capacity(actions.len());
        let mut a_in = Vec::with_capacity(ACTION_IN);
        let mut q = Vec::with_capacity(PH);
        for a in actions {
            a_in.clear();
            a_in.extend_from_slice(&a.dense);
            a_in.extend_from_slice(self.emb(&self.type_emb, a.tokens[0] as usize, ACTION_EMBED));
            a_in.extend_from_slice(self.emb(
                &self.card_emb,
                (a.tokens[1] as usize).min(self.vocab - 1),
                EMBED_DIM,
            ));
            a_in.extend_from_slice(self.emb(&self.detail_emb, a.tokens[2] as usize, ACTION_EMBED));
            self.p_action.forward_into(&a_in, &mut q);
            for (qi, si) in q.iter_mut().zip(&ps) {
                *qi = (*qi + si).max(0.0);
            }
            logits.push(self.p_out.forward(&q)[0]);
        }
        Eval { value, logits }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout1_fc1_gets_zero_columns_in_place() {
        // Column j of a fake layout-1 fc1 holds j + 1: after migration the layout-1 columns keep
        // their values and order, and every layout-2 input has a zero weight.
        let w: Vec<f32> = (0..2 * TRUNK_IN_V1)
            .map(|j| (j % TRUNK_IN_V1) as f32 + 1.0)
            .collect();
        let mut t = Tensors::new();
        t.insert("fc1.weight".into(), (vec![2, TRUNK_IN_V1], w));
        let emb: Vec<f32> = (0..VOCAB.size * EMBED_DIM).map(|i| i as f32).collect();
        t.insert(
            "card_emb.weight".into(),
            (vec![VOCAB.size, EMBED_DIM], emb.clone()),
        );
        migrate(&mut t).unwrap();
        let (d, w) = &t["fc1.weight"];
        assert_eq!(d, &vec![2, TRUNK_IN]);
        let row = &w[TRUNK_IN..];
        assert!((0..STATE_DENSE_V1).all(|j| row[j] == j as f32 + 1.0));
        assert!(row[STATE_DENSE_V1..STATE_DENSE].iter().all(|&x| x == 0.0));
        let rest = TRUNK_IN_V1 - STATE_DENSE_V1;
        assert!((0..rest).all(|k| row[STATE_DENSE + k] == (STATE_DENSE_V1 + k) as f32 + 1.0));
        assert!(row[STATE_DENSE + rest..].iter().all(|&x| x == 0.0));
        // The layout-1 vocabulary is the current one: embeddings untouched.
        assert_eq!(t["card_emb.weight"].1, emb);
    }

    #[test]
    fn test_embeddings_follow_card_ids_across_vocabularies() {
        // Weights trained when "A1 001" had row 5: its row comes back at its current index.
        let row = VOCAB.by_id["A1 001"] as usize;
        let mut emb = vec![0f32; 6 * EMBED_DIM];
        emb[5 * EMBED_DIM] = 7.0;
        let table: Vec<f32> = "A1 001\t5\n".bytes().map(|b| b as f32).collect();
        let mut t = Tensors::new();
        t.insert("card_emb.weight".into(), (vec![6, EMBED_DIM], emb));
        t.insert("vocab.table".into(), (vec![table.len()], table));
        migrate(&mut t).unwrap();
        let (d, e) = &t["card_emb.weight"];
        assert_eq!(d, &vec![VOCAB.size, EMBED_DIM]);
        assert_eq!(e[row * EMBED_DIM], 7.0);
        assert_eq!(e.iter().filter(|&&x| x != 0.0).count(), 1);
    }
}
