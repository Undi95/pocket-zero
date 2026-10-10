//! Self-play: the network plays both sides, every searched decision becomes a training sample.
//!
//! Shard file (read by `zero/data.py`): b"PZS2", then u32 STATE_DENSE, N_TOKENS, ACTION_DENSE,
//! ACTION_TOKENS, MAX_ACTIONS, then fixed-size records, all little-endian:
//!   f32 dense[STATE_DENSE] | i32 tokens[N_TOKENS] | i32 n_actions
//!   | f32 action_dense[MAX_ACTIONS * ACTION_DENSE] | i32 action_tokens[MAX_ACTIONS * ACTION_TOKENS]
//!   | f32 policy[MAX_ACTIONS] | f32 z (final result for the player to move) | f32 q (search value)
//!   | i32 actor | i32 last (1 on the last decision of its game)
//! The records of a game are contiguous and in order, so value targets can bootstrap from the
//! following positions (TD(lambda) in zero/train.py). PZS1 is the same without actor/last.

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::encode::{
    encode_action, encode_state, ActionFeatures, Obs, ACTION_DENSE, ACTION_TOKENS, N_TOKENS,
    STATE_DENSE,
};
use super::mcts::{pick, search, Evaluator, MctsConfig};
use crate::actions::apply_action;
use crate::players::Player;
use crate::state::GameOutcome;
use crate::{Deck, State};

pub const MAX_ACTIONS: usize = 48;

pub struct Sample {
    pub obs: Obs,
    pub actions: Vec<ActionFeatures>,
    pub policy: Vec<f32>,
    pub actor: usize,
    /// Search value for the player to move; NaN when there was no search (imitation).
    pub q: f32,
}

#[derive(Clone)]
pub struct SelfPlayConfig {
    pub mcts: MctsConfig,
    /// PUCT root only: decisions per game played proportionally to visit counts before switching
    /// to the most visited one.
    pub temperature_moves: usize,
}

/// Plays one game; returns its samples and the outcome.
pub fn play_game<E: Evaluator>(
    eval: &E,
    cfg: &SelfPlayConfig,
    deck_a: &Deck,
    deck_b: &Deck,
    rng: &mut StdRng,
) -> (Vec<Sample>, Option<GameOutcome>) {
    let mut state = State::initialize(deck_a, deck_b, rng);
    let mut samples = Vec::new();
    let mut decisions = 0usize;
    let mut steps = 0usize;
    while !state.is_game_over() && steps < 5000 {
        steps += 1;
        let (actor, actions) = state.generate_possible_actions();
        if actions.len() == 1 {
            apply_action(rng, &mut state, &actions[0]);
            continue;
        }
        let res = search(eval, &cfg.mcts, &state, actor, &actions, rng);
        samples.push(Sample {
            obs: encode_state(&state, actor),
            actions: actions.iter().map(|a| encode_action(a, actor)).collect(),
            policy: res.policy.clone(),
            actor,
            q: res.value,
        });
        // The Gumbel root already explores (its noise makes the pick a sample of the improved
        // policy); the PUCT root plays proportionally to visits for the first decisions.
        let i = if cfg.mcts.gumbel {
            res.best
        } else {
            let temperature = if decisions < cfg.temperature_moves {
                1.0
            } else {
                0.0
            };
            pick(&res.visits, temperature, rng)
        };
        decisions += 1;
        apply_action(rng, &mut state, &actions[i]);
    }
    (samples, state.winner)
}

fn z_for(outcome: Option<GameOutcome>, actor: usize) -> f32 {
    match outcome {
        Some(GameOutcome::Win(p)) if p == actor => 1.0,
        Some(GameOutcome::Win(_)) => -1.0,
        _ => 0.0,
    }
}

pub fn write_header(w: &mut impl Write) -> std::io::Result<()> {
    w.write_all(b"PZS2")?;
    for v in [
        STATE_DENSE,
        N_TOKENS,
        ACTION_DENSE,
        ACTION_TOKENS,
        MAX_ACTIONS,
    ] {
        w.write_all(&(v as u32).to_le_bytes())?;
    }
    Ok(())
}

pub fn write_sample(w: &mut impl Write, s: &Sample, z: f32, last: bool) -> std::io::Result<()> {
    // Keep the most visited actions if there are more than MAX_ACTIONS.
    let mut order: Vec<usize> = (0..s.actions.len()).collect();
    order.sort_by(|&a, &b| s.policy[b].total_cmp(&s.policy[a]));
    order.truncate(MAX_ACTIONS);
    let kept: f32 = order.iter().map(|&i| s.policy[i]).sum::<f32>().max(1e-9);

    let mut buf: Vec<u8> = Vec::with_capacity(
        4 * (STATE_DENSE + N_TOKENS + 1 + MAX_ACTIONS * (ACTION_DENSE + ACTION_TOKENS + 1) + 4),
    );
    for x in &s.obs.dense {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    for t in &s.obs.tokens {
        buf.extend_from_slice(&(*t as i32).to_le_bytes());
    }
    buf.extend_from_slice(&(order.len() as i32).to_le_bytes());
    for k in 0..MAX_ACTIONS {
        match order.get(k) {
            Some(&i) => {
                for x in &s.actions[i].dense {
                    buf.extend_from_slice(&x.to_le_bytes());
                }
            }
            None => buf.extend(std::iter::repeat_n(0u8, 4 * ACTION_DENSE)),
        }
    }
    for k in 0..MAX_ACTIONS {
        match order.get(k) {
            Some(&i) => {
                for t in &s.actions[i].tokens {
                    buf.extend_from_slice(&(*t as i32).to_le_bytes());
                }
            }
            None => buf.extend(std::iter::repeat_n(0u8, 4 * ACTION_TOKENS)),
        }
    }
    for k in 0..MAX_ACTIONS {
        let p = order.get(k).map_or(0.0, |&i| s.policy[i] / kept);
        buf.extend_from_slice(&p.to_le_bytes());
    }
    // No search value (imitation): q 0 and actor -1, so value targets keep the game result.
    let (q, actor) = if s.q.is_nan() {
        (0.0f32, -1i32)
    } else {
        (s.q, s.actor as i32)
    };
    buf.extend_from_slice(&z.to_le_bytes());
    buf.extend_from_slice(&q.to_le_bytes());
    buf.extend_from_slice(&actor.to_le_bytes());
    buf.extend_from_slice(&(last as i32).to_le_bytes());
    w.write_all(&buf)
}

pub struct SelfPlayStats {
    pub games: usize,
    pub samples: usize,
    pub wins_first_deck: usize,
    pub ties: usize,
}

/// Self-play: `games` games in parallel between random pairs from `decks`, written to `out`.
pub fn run<E: Evaluator>(
    eval: &E,
    cfg: &SelfPlayConfig,
    decks: &[Deck],
    games: usize,
    seed: u64,
    out: &str,
    progress: bool,
) -> std::io::Result<SelfPlayStats> {
    run_games(
        |a, b, rng| play_game(eval, cfg, a, b, rng),
        decks,
        games,
        seed,
        out,
        progress,
    )
}

/// Imitation data: an existing player (e.g. e3, or the y expert) plays both sides; each of its
/// decisions becomes a sample. The policy target is the player's own probabilities when it gives
/// them (`last_scores`, normalised), otherwise the move it chose.
pub fn play_imitation<M>(
    make: &M,
    deck_a: &Deck,
    deck_b: &Deck,
    rng: &mut StdRng,
) -> (Vec<Sample>, Option<GameOutcome>)
where
    M: Fn(Deck) -> Box<dyn Player>,
{
    let mut players = [make(deck_a.clone()), make(deck_b.clone())];
    let mut state = State::initialize(deck_a, deck_b, rng);
    let mut samples = Vec::new();
    let mut steps = 0usize;
    while !state.is_game_over() && steps < 5000 {
        steps += 1;
        let (actor, actions) = state.generate_possible_actions();
        if actions.len() == 1 {
            apply_action(rng, &mut state, &actions[0]);
            continue;
        }
        let chosen = players[actor].decision_fn(rng, &state, &actions);
        let i = actions.iter().position(|a| *a == chosen).unwrap_or(0);
        let policy = match players[actor].last_scores() {
            Some(s)
                if s.len() == actions.len()
                    && s.iter().all(|x| *x >= 0.0)
                    && s.iter().sum::<f64>() > 0.0 =>
            {
                let sum: f64 = s.iter().sum();
                s.iter().map(|x| (*x / sum) as f32).collect()
            }
            _ => {
                let mut p = vec![0.0; actions.len()];
                p[i] = 1.0;
                p
            }
        };
        samples.push(Sample {
            obs: encode_state(&state, actor),
            actions: actions.iter().map(|a| encode_action(a, actor)).collect(),
            policy,
            actor,
            q: f32::NAN, // no search value
        });
        apply_action(rng, &mut state, &actions[i]);
    }
    (samples, state.winner)
}

/// One decision of a value check: what the hand-written evaluation and the network think of the
/// position, and how the game ended (all for the player to move).
pub struct ValueRow {
    pub game: usize,
    pub turn: u8,
    pub actor: usize,
    pub baseline: f64,
    pub net: f32,
    pub z: f32,
}

/// Plays `games` games with `make` players and records, at every decision, the hand-written
/// evaluation, the network value and the final result: which one predicts the winner better?
pub fn value_check<M>(
    make: M,
    net: &super::net::Net,
    decks: &[Deck],
    games: usize,
    seed: u64,
) -> Vec<ValueRow>
where
    M: Fn(Deck) -> Box<dyn Player> + Sync,
{
    (0..games)
        .into_par_iter()
        .flat_map_iter(|g| {
            let mut rng =
                StdRng::seed_from_u64(seed.wrapping_mul(1_000_003).wrapping_add(g as u64));
            let a = &decks[rng.gen_range(0..decks.len())];
            let b = &decks[rng.gen_range(0..decks.len())];
            let mut players = [make(a.clone()), make(b.clone())];
            let mut state = State::initialize(a, b, &mut rng);
            let mut rows = Vec::new();
            let mut steps = 0usize;
            while !state.is_game_over() && steps < 5000 {
                steps += 1;
                let (actor, actions) = state.generate_possible_actions();
                if actions.len() > 1 {
                    rows.push(ValueRow {
                        game: g,
                        turn: state.turn_count,
                        actor,
                        baseline: crate::players::baseline_value_function(&state, actor),
                        net: net.evaluate(&encode_state(&state, actor), &[]).value,
                        z: 0.0,
                    });
                }
                let action = if actions.len() == 1 {
                    actions[0].clone()
                } else {
                    players[actor].decision_fn(&mut rng, &state, &actions)
                };
                apply_action(&mut rng, &mut state, &action);
            }
            for r in rows.iter_mut() {
                r.z = z_for(state.winner, r.actor);
            }
            rows
        })
        .collect()
}

pub fn run_imitation<M>(
    make: M,
    decks: &[Deck],
    games: usize,
    seed: u64,
    out: &str,
    progress: bool,
) -> std::io::Result<SelfPlayStats>
where
    M: Fn(Deck) -> Box<dyn Player> + Sync,
{
    run_games(
        |a, b, rng| play_imitation(&make, a, b, rng),
        decks,
        games,
        seed,
        out,
        progress,
    )
}

fn run_games<F>(
    play: F,
    decks: &[Deck],
    games: usize,
    seed: u64,
    out: &str,
    progress: bool,
) -> std::io::Result<SelfPlayStats>
where
    F: Fn(&Deck, &Deck, &mut StdRng) -> (Vec<Sample>, Option<GameOutcome>) + Sync,
{
    let file = std::fs::File::create(out)?;
    let writer = Mutex::new(std::io::BufWriter::new(file));
    write_header(&mut *writer.lock().unwrap())?;
    let done = AtomicUsize::new(0);
    let samples = AtomicUsize::new(0);
    let wins = AtomicUsize::new(0);
    let ties = AtomicUsize::new(0);
    let start = std::time::Instant::now();
    (0..games)
        .into_par_iter()
        .try_for_each(|g| -> std::io::Result<()> {
            let mut rng =
                StdRng::seed_from_u64(seed.wrapping_mul(1_000_003).wrapping_add(g as u64));
            let a = &decks[rng.gen_range(0..decks.len())];
            let b = &decks[rng.gen_range(0..decks.len())];
            let (game_samples, outcome) = play(a, b, &mut rng);
            match outcome {
                Some(GameOutcome::Win(0)) => {
                    wins.fetch_add(1, Ordering::Relaxed);
                }
                Some(GameOutcome::Win(_)) => {}
                _ => {
                    ties.fetch_add(1, Ordering::Relaxed);
                }
            }
            {
                let mut w = writer.lock().unwrap();
                for (k, s) in game_samples.iter().enumerate() {
                    let last = k + 1 == game_samples.len();
                    write_sample(&mut *w, s, z_for(outcome, s.actor), last)?;
                }
            }
            samples.fetch_add(game_samples.len(), Ordering::Relaxed);
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if progress && (n.is_multiple_of(10) || n == games) {
                let secs = start.elapsed().as_secs_f64();
                eprintln!(
                    "selfplay {n}/{games} games, {} samples, {:.2} games/s",
                    samples.load(Ordering::Relaxed),
                    n as f64 / secs
                );
            }
            Ok(())
        })?;
    writer.lock().unwrap().flush()?;
    Ok(SelfPlayStats {
        games,
        samples: samples.load(Ordering::Relaxed),
        wins_first_deck: wins.load(Ordering::Relaxed),
        ties: ties.load(Ordering::Relaxed),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zero::mcts::Bootstrap;

    #[test]
    fn test_selfplay_game_finishes_and_writes_fixed_size_records() {
        let deck = Deck::from_file("example_decks/venusaur-exeggutor.txt").unwrap();
        let cfg = SelfPlayConfig {
            mcts: MctsConfig {
                simulations: 8,
                noise_frac: 0.25,
                gumbel_noise: true,
                ..MctsConfig::default()
            },
            temperature_moves: 10,
        };
        let mut rng = StdRng::seed_from_u64(1);
        let (samples, outcome) = play_game(&Bootstrap, &cfg, &deck, &deck, &mut rng);
        assert!(outcome.is_some(), "game should end");
        assert!(!samples.is_empty());
        let mut buf = Vec::new();
        write_sample(&mut buf, &samples[0], 1.0, false).unwrap();
        let expected =
            4 * (STATE_DENSE + N_TOKENS + 1 + MAX_ACTIONS * (ACTION_DENSE + ACTION_TOKENS + 1) + 4);
        assert_eq!(buf.len(), expected);
        let p: f32 = samples[0].policy.iter().sum();
        assert!((p - 1.0).abs() < 1e-4);
    }
}
