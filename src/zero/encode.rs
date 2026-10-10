//! Turns a game state, seen by one player, into network inputs.
//!
//! Only information that player could know is encoded: the opponent's hand is reduced to its
//! size, decks to their size (plus the multiset of your own remaining cards, which you know from
//! your decklist), and nothing depends on deck order. The layout is mirrored by
//! `zero/model.py`; `pocket-zero info` prints the dimensions so Python never hardcodes them.
//!
//! Actions are encoded generically from their serde representation, so new `SimpleAction`
//! variants are picked up without touching this file.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::LazyLock;

use serde_json::Value;
use strum::IntoEnumIterator;

use crate::actions::{Action, SimpleAction};
use crate::card_ids::CardId;
use crate::database::get_card_by_enum;
use crate::fxhash::{FxBuildHasher, FxHasher};
use crate::hooks::get_stage;
use crate::models::{Card, EnergyType, PlayedCard, TrainerType};
use crate::State;

// ---------------------------------------------------------------- sizes (shared with Python)

pub const EMBED_DIM: usize = 32;
pub const N_SLOTS: usize = 8; // my 4 in-play slots, then the opponent's
pub const HAND_MAX: usize = 10;
pub const DISCARD_MAX: usize = 30;
pub const DECK_MAX: usize = 20;
/// Token layout: 8 slot cards, stadium, my hand, my discard, opponent discard, my remaining deck.
pub const N_TOKENS: usize = N_SLOTS + 1 + HAND_MAX + DISCARD_MAX + DISCARD_MAX + DECK_MAX;
pub const BAGS: [(usize, usize); 4] = [
    (N_SLOTS + 1, HAND_MAX),
    (N_SLOTS + 1 + HAND_MAX, DISCARD_MAX),
    (N_SLOTS + 1 + HAND_MAX + DISCARD_MAX, DISCARD_MAX),
    (N_SLOTS + 1 + HAND_MAX + 2 * DISCARD_MAX, DECK_MAX),
];

const N_ENERGY: usize = 10;
const EFFECT_BUCKETS: usize = 8;
const TURN_EFFECT_BUCKETS: usize = 16;
pub const CARD_STATIC_DIM: usize = 40;
const SLOT_DIM: usize = 1 + 3 + N_ENERGY + 1 + 5 + 2 + 1 + EFFECT_BUCKETS + CARD_STATIC_DIM;
const GLOBAL_DIM: usize =
    1 + 2 + 2 + 2 + 2 + 2 + 4 * (N_ENERGY + 1) + 2 * N_ENERGY + 8 + 3 + 2 * TURN_EFFECT_BUCKETS;
pub const STATE_DENSE: usize = GLOBAL_DIM + N_SLOTS * SLOT_DIM;

pub const ACTION_TYPE_BUCKETS: usize = 256;
pub const ACTION_DETAIL_BUCKETS: usize = 1024;
pub const ACTION_DENSE: usize = 1 + 4 + 4 + 2 + N_ENERGY + 1 + 2 + 1;
/// Action tokens: [type bucket, card (vocab index, 0 = none), detail bucket].
pub const ACTION_TOKENS: usize = 3;

// ---------------------------------------------------------------- card vocabulary

/// Index 0 is padding. Prints with identical rules text share one index, so a reprint plays
/// exactly like the original.
pub struct Vocab {
    pub by_id: HashMap<String, u32>,
    pub size: usize,
    pub static_features: Vec<[f32; CARD_STATIC_DIM]>,
}

fn rules_key(card: &Card) -> String {
    match card {
        Card::Pokemon(p) => format!(
            "P|{}|{}|{:?}|{:?}|{}|{:?}|{:?}",
            p.name,
            p.hp,
            p.attacks
                .iter()
                .map(|a| (&a.title, a.fixed_damage, &a.effect, &a.energy_required))
                .collect::<Vec<_>>(),
            p.ability.as_ref().map(|a| (&a.title, &a.effect)),
            p.stage,
            p.weakness,
            p.retreat_cost.len()
        ),
        Card::Trainer(t) => format!("T|{}|{}", t.name, t.effect),
    }
}

fn energy_index(e: EnergyType) -> usize {
    e as usize
}

fn card_static(card: &Card) -> [f32; CARD_STATIC_DIM] {
    let mut f = [0f32; CARD_STATIC_DIM];
    match card {
        Card::Pokemon(p) => {
            f[0] = 1.0;
            f[1] = p.hp as f32 / 200.0;
            f[2 + (p.stage as usize).min(2)] = 1.0; // 2..5
            f[5] = card.is_ex() as u8 as f32;
            f[6] = card.is_mega() as u8 as f32;
            f[7 + energy_index(p.energy_type)] = 1.0; // 7..17
            if let Some(w) = p.weakness {
                f[17 + energy_index(w)] = 1.0; // 17..27
            }
            f[27] = p.retreat_cost.len() as f32 / 4.0;
            let best = p.attacks.iter().map(|a| a.fixed_damage).max().unwrap_or(0);
            f[28] = best as f32 / 200.0;
            f[29] = p
                .attacks
                .iter()
                .map(|a| a.energy_required.len())
                .min()
                .unwrap_or(0) as f32
                / 5.0;
            f[30] = p.attacks.len() as f32 / 2.0;
            f[31] = p.ability.is_some() as u8 as f32;
            f[32] = p.attacks.iter().any(|a| a.effect.is_some()) as u8 as f32;
        }
        Card::Trainer(t) => {
            f[33] = 1.0;
            let k = match t.trainer_card_type {
                TrainerType::Supporter => 0,
                TrainerType::Item => 1,
                TrainerType::Tool => 2,
                TrainerType::Fossil => 3,
                TrainerType::Stadium => 4,
            };
            f[34 + k] = 1.0; // 34..39
        }
    }
    f
}

pub static VOCAB: LazyLock<Vocab> = LazyLock::new(|| {
    let mut by_key: HashMap<String, u32> = HashMap::new();
    let mut by_id = HashMap::new();
    let mut static_features = vec![[0f32; CARD_STATIC_DIM]];
    for id in CardId::iter() {
        let card = get_card_by_enum(id);
        let key = rules_key(&card);
        let next = static_features.len() as u32;
        let idx = *by_key.entry(key).or_insert_with(|| {
            static_features.push(card_static(&card));
            next
        });
        by_id.insert(card.get_id(), idx);
    }
    Vocab {
        size: static_features.len(),
        by_id,
        static_features,
    }
});

pub fn card_token(card: &Card) -> u32 {
    VOCAB.by_id.get(&card.get_id()).copied().unwrap_or(0)
}

// ---------------------------------------------------------------- hashing

/// FNV-1a: stable across Rust versions and platforms, unlike `DefaultHasher`.
pub fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn variant_name<T: serde::Serialize>(v: &T) -> String {
    match serde_json::to_value(v) {
        Ok(Value::String(s)) => s,
        Ok(Value::Object(m)) => m.keys().next().cloned().unwrap_or_default(),
        _ => String::new(),
    }
}

// ---------------------------------------------------------------- state

#[derive(Debug, Clone)]
pub struct Obs {
    pub dense: Vec<f32>,
    pub tokens: Vec<u32>,
}

fn push_energy_onehot(out: &mut Vec<f32>, e: Option<EnergyType>) {
    let mut v = [0f32; N_ENERGY + 1];
    match e {
        Some(e) => v[energy_index(e)] = 1.0,
        None => v[N_ENERGY] = 1.0,
    }
    out.extend_from_slice(&v);
}

fn encode_slot(out: &mut Vec<f32>, slot: &Option<PlayedCard>) {
    let start = out.len();
    let Some(p) = slot else {
        out.resize(start + SLOT_DIM, 0.0);
        return;
    };
    out.push(1.0);
    let total = p.get_effective_total_hp() as f32;
    let remaining = p.get_remaining_hp() as f32;
    out.extend_from_slice(&[
        remaining / 200.0,
        total / 200.0,
        (total - remaining) / 200.0,
    ]);
    let mut energy = [0f32; N_ENERGY];
    for e in &p.attached_energy {
        energy[energy_index(*e)] += 0.25;
    }
    out.extend_from_slice(&energy);
    out.push(p.attached_tools.len() as f32);
    out.extend_from_slice(&[
        p.is_poisoned() as u8 as f32,
        p.is_burned() as u8 as f32,
        p.is_asleep() as u8 as f32,
        p.is_paralyzed() as u8 as f32,
        p.is_confused() as u8 as f32,
    ]);
    out.push(p.played_this_turn as u8 as f32);
    out.push(p.ability_used as u8 as f32);
    out.push(get_stage(p) as f32 / 2.0);
    let mut effects = [0f32; EFFECT_BUCKETS];
    for (effect, _) in p.get_effects() {
        effects[(fnv(&variant_name(effect)) as usize) % EFFECT_BUCKETS] += 1.0;
    }
    out.extend_from_slice(&effects);
    let idx = card_token(&p.card) as usize;
    out.extend_from_slice(&VOCAB.static_features[idx]);
    debug_assert_eq!(out.len() - start, SLOT_DIM);
}

fn push_bag(tokens: &mut Vec<u32>, cards: &[Card], max: usize) {
    let start = tokens.len();
    for c in cards.iter().take(max) {
        tokens.push(card_token(c));
    }
    tokens.resize(start + max, 0);
}

/// Encodes `state` from the point of view of `me`.
pub fn encode_state(state: &State, me: usize) -> Obs {
    let opp = 1 - me;
    let mut d: Vec<f32> = Vec::with_capacity(STATE_DENSE);

    d.push(state.turn_count as f32 / 30.0);
    d.extend_from_slice(&[
        state.points[me] as f32 / 3.0,
        state.points[opp] as f32 / 3.0,
    ]);
    d.extend_from_slice(&[
        state.hands[me].len() as f32 / 10.0,
        state.hands[opp].len() as f32 / 10.0,
    ]);
    d.extend_from_slice(&[
        state.decks[me].cards.len() as f32 / 20.0,
        state.decks[opp].cards.len() as f32 / 20.0,
    ]);
    d.extend_from_slice(&[
        state.discard_piles[me].len() as f32 / 20.0,
        state.discard_piles[opp].len() as f32 / 20.0,
    ]);
    d.extend_from_slice(&[
        state.discard_energies[me].len() as f32 / 10.0,
        state.discard_energies[opp].len() as f32 / 10.0,
    ]);
    for p in [me, opp] {
        push_energy_onehot(&mut d, state.energy_zone[p].current);
        push_energy_onehot(&mut d, state.energy_zone[p].next);
    }
    for p in [me, opp] {
        let mut v = [0f32; N_ENERGY];
        for e in &state.decks[p].energy_types {
            v[energy_index(*e)] = 1.0;
        }
        d.extend_from_slice(&v);
    }
    d.extend_from_slice(&[
        state.has_played_support as u8 as f32,
        state.has_retreated as u8 as f32,
        state.has_played_stadium as u8 as f32,
        state.has_used_stadium[me] as u8 as f32,
        state.is_users_first_turn() as u8 as f32,
        (state.current_player == me) as u8 as f32,
        state.end_turn_pending as u8 as f32,
        (!state.move_generation_stack.is_empty()) as u8 as f32,
    ]);
    let owner = match (state.active_stadium.is_some(), state.active_stadium_owner) {
        (false, _) => [0.0, 0.0, 1.0],
        (true, Some(o)) if o == me => [1.0, 0.0, 0.0],
        (true, _) => [0.0, 1.0, 0.0],
    };
    d.extend_from_slice(&owner);
    for turn in [state.turn_count, state.turn_count.saturating_add(1)] {
        let mut v = [0f32; TURN_EFFECT_BUCKETS];
        if let Some(effects) = state.turn_effects.get(&turn) {
            for e in effects {
                v[(fnv(&variant_name(e)) as usize) % TURN_EFFECT_BUCKETS] += 1.0;
            }
        }
        d.extend_from_slice(&v);
    }
    debug_assert_eq!(d.len(), GLOBAL_DIM);

    let mut tokens: Vec<u32> = Vec::with_capacity(N_TOKENS);
    for p in [me, opp] {
        for slot in &state.in_play_pokemon[p] {
            encode_slot(&mut d, slot);
            tokens.push(slot.as_ref().map(|c| card_token(&c.card)).unwrap_or(0));
        }
    }
    tokens.push(state.active_stadium.as_ref().map(card_token).unwrap_or(0));
    push_bag(&mut tokens, &state.hands[me], HAND_MAX);
    push_bag(&mut tokens, &state.discard_piles[me], DISCARD_MAX);
    push_bag(&mut tokens, &state.discard_piles[opp], DISCARD_MAX);
    // The multiset of your remaining deck is known; sort so the order leaks nothing.
    let mut deck: Vec<u32> = state.decks[me].cards.iter().map(card_token).collect();
    deck.sort_unstable();
    deck.truncate(DECK_MAX);
    deck.resize(DECK_MAX, 0);
    tokens.extend(deck);

    debug_assert_eq!(d.len(), STATE_DENSE);
    debug_assert_eq!(tokens.len(), N_TOKENS);
    Obs { dense: d, tokens }
}

// ---------------------------------------------------------------- actions

#[derive(Debug, Clone)]
pub struct ActionFeatures {
    pub dense: [f32; ACTION_DENSE],
    pub tokens: [u32; ACTION_TOKENS],
}

const ENERGY_NAMES: [&str; N_ENERGY] = [
    "Grass",
    "Fire",
    "Water",
    "Lightning",
    "Psychic",
    "Fighting",
    "Darkness",
    "Metal",
    "Dragon",
    "Colorless",
];

struct Walk {
    idx: Vec<usize>,
    player: Option<usize>,
    energy: [f32; N_ENERGY],
    amount: f32,
    damage: f32,
    cost: f32,
    cards: Vec<u32>,
    title: Option<String>,
}

fn walk(v: &Value, key: &str, w: &mut Walk) {
    match v {
        Value::Object(m) => {
            if let Some(Value::String(id)) = m.get("id") {
                if let Some(t) = VOCAB.by_id.get(id) {
                    w.cards.push(*t);
                    return; // a whole card: don't count its attacks' energies as the action's
                }
            }
            if let (Some(Value::String(t)), Some(dmg)) = (m.get("title"), m.get("fixed_damage")) {
                w.title = Some(t.clone());
                w.damage = dmg.as_f64().unwrap_or(0.0) as f32 / 200.0;
                if let Some(Value::Array(cost)) = m.get("energy_required") {
                    w.cost = cost.len() as f32 / 5.0;
                }
                return;
            }
            for (k, x) in m {
                walk(x, k, w);
            }
        }
        Value::Array(a) => {
            for x in a {
                walk(x, key, w);
            }
        }
        Value::Number(n) => {
            let x = n.as_f64().unwrap_or(0.0);
            let k = key.to_ascii_lowercase();
            if k.contains("idx") || k.is_empty() {
                w.idx.push(x as usize);
            } else if k == "player" {
                w.player = Some(x as usize);
            } else {
                w.amount += x as f32;
            }
        }
        Value::String(s) => {
            if let Some(i) = ENERGY_NAMES.iter().position(|e| e == s) {
                w.energy[i] += 1.0;
            }
        }
        _ => {}
    }
}

type CacheEntry = (SimpleAction, bool, usize, ActionFeatures);
type ActionCache = HashMap<u64, Vec<CacheEntry>, FxBuildHasher>;

thread_local! {
    /// encode_action is a pure function of (action, is_stack, player) and the same few hundred
    /// actions come back in every simulation: computing them (through their serde form) was a
    /// sixth of the engine's time in self-play.
    static ACTION_CACHE: RefCell<ActionCache> = RefCell::new(ActionCache::default());
}

pub fn encode_action(action: &Action, me: usize) -> ActionFeatures {
    let mut h = FxHasher::default();
    action.action.hash(&mut h);
    action.is_stack.hash(&mut h);
    me.hash(&mut h);
    let k = h.finish();
    let same = |e: &&CacheEntry| e.0 == action.action && e.1 == action.is_stack && e.2 == me;
    let hit = ACTION_CACHE.with(|c| {
        c.borrow()
            .get(&k)
            .and_then(|v| v.iter().find(same).map(|e| e.3.clone()))
    });
    if let Some(f) = hit {
        return f;
    }
    let f = encode_action_uncached(action, me);
    ACTION_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() >= 200_000 {
            c.clear();
        }
        c.entry(k)
            .or_default()
            .push((action.action.clone(), action.is_stack, me, f.clone()));
    });
    f
}

fn encode_action_uncached(action: &Action, me: usize) -> ActionFeatures {
    let json = serde_json::to_value(&action.action).unwrap_or(Value::Null);
    let (name, body) = match &json {
        Value::String(s) => (s.clone(), Value::Null),
        Value::Object(m) => m
            .iter()
            .next()
            .map(|(k, v)| (k.clone(), v.clone()))
            .unwrap_or_default(),
        _ => (String::new(), Value::Null),
    };
    let mut w = Walk {
        idx: vec![],
        player: None,
        energy: [0.0; N_ENERGY],
        amount: 0.0,
        damage: 0.0,
        cost: 0.0,
        cards: vec![],
        title: None,
    };
    walk(&body, "", &mut w);

    let mut dense = [0f32; ACTION_DENSE];
    dense[0] = action.is_stack as u8 as f32;
    if let Some(&i) = w.idx.first() {
        dense[1 + i.min(3)] = 1.0;
    }
    if let Some(&i) = w.idx.get(1) {
        dense[5 + i.min(3)] = 1.0;
    }
    match w.player {
        Some(p) if p == me => dense[9] = 1.0,
        Some(_) => dense[10] = 1.0,
        None => {}
    }
    for i in 0..N_ENERGY {
        dense[11 + i] = w.energy[i] / 4.0;
    }
    dense[21] = (w.amount / 10.0).min(3.0);
    dense[22] = w.damage;
    dense[23] = w.cost;
    dense[24] = w.cards.len() as f32 / 5.0;

    let detail = w.title.unwrap_or_default();
    let tokens = [
        (fnv(&name) as usize % ACTION_TYPE_BUCKETS) as u32,
        w.cards.first().copied().unwrap_or(0),
        if detail.is_empty() {
            0
        } else {
            (fnv(&detail) as usize % ACTION_DETAIL_BUCKETS) as u32
        },
    ];
    ActionFeatures { dense, tokens }
}

/// A stable identity for an action across determinizations (within a process), used to share
/// MCTS statistics. Cards hash by id, so the same move with the same cards gets the same key.
pub fn action_key(action: &SimpleAction) -> u64 {
    let mut h = FxHasher::default();
    action.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::get_initialized_game;

    #[test]
    fn test_encoding_sizes_and_hidden_information() {
        let game = get_initialized_game(3);
        let state = game.get_state_clone();
        let me = state.current_player;
        let obs = encode_state(&state, me);
        assert_eq!(obs.dense.len(), STATE_DENSE);
        assert_eq!(obs.tokens.len(), N_TOKENS);
        assert!(obs.tokens.iter().all(|&t| (t as usize) < VOCAB.size));

        // Changing the opponent's hand contents (same size) or any deck order changes nothing.
        let mut other = state.clone();
        let opp = 1 - me;
        let mut pool = other.hands[opp].clone();
        pool.append(&mut other.decks[opp].cards);
        pool.reverse();
        other.decks[opp].cards = pool.split_off(state.hands[opp].len());
        other.hands[opp] = pool;
        other.decks[me].cards.reverse();
        let obs2 = encode_state(&other, me);
        assert_eq!(obs.dense, obs2.dense);
        assert_eq!(obs.tokens, obs2.tokens);

        let (actor, actions) = state.generate_possible_actions();
        for a in &actions {
            let f = encode_action(a, actor);
            assert!(f.tokens[0] < ACTION_TYPE_BUCKETS as u32);
            assert!(f.dense.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn test_reprints_share_a_vocab_index() {
        let a = card_token(&get_card_by_enum(CardId::A1225Sabrina));
        let b = card_token(&get_card_by_enum(CardId::A4b338Sabrina));
        assert_eq!(a, b);
        assert_ne!(a, 0);
    }
}
