//! Monte Carlo tree search for a game with hidden information and chance.
//!
//! - Hidden information: every simulation starts from a fresh *determinization* of the root
//!   (both decks shuffled, the opponent's hand redealt from their unseen cards), so the search
//!   never relies on cards the searching player can't know.
//! - Chance (coin flips, draws, random effects): the engine samples each outcome when an action
//!   is applied, so the tree is "open loop": nodes are reached by sequences of *decisions*, and
//!   their statistics average over everything random in between.
//! - Forced moves (a single legal action) are applied without creating a decision.
//!
//! Values are stored from the root player's point of view and flipped when the opponent acts.
//!
//! Root (default): Gumbel search (Danihelka et al. 2022, "Policy improvement by planning with
//! Gumbel"). Sequential halving over the most promising actions; every candidate of a phase is
//! searched on the *same* determinizations and random streams, so they are compared on the same
//! hidden cards and the same coin flips (paired comparisons: the luck cancels out instead of
//! drowning the difference between moves). The policy target is the prior improved by the
//! completed Q-values, which stays meaningful with few simulations, unlike visit counts.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
pub(crate) use rand_distr_gamma::sample_gamma;

use super::encode::{action_key, encode_action, encode_state};
use super::net::Net;
use crate::actions::{apply_action, Action};
use crate::players::{baseline_value_function, determinize};
use crate::state::GameOutcome;
use crate::State;

/// Evaluates a decision point: value for the player to move in [-1, 1] and a prior per action.
pub trait Evaluator: Sync {
    fn evaluate(&self, state: &State, actor: usize, actions: &[Action]) -> (f32, Vec<f32>);
}

/// Used before any network exists: uniform priors and the engine's hand-written evaluation.
pub struct Bootstrap;

impl Evaluator for Bootstrap {
    fn evaluate(&self, state: &State, actor: usize, actions: &[Action]) -> (f32, Vec<f32>) {
        let v = (baseline_value_function(state, actor) / 15_000.0).tanh() as f32;
        (v, vec![1.0 / actions.len() as f32; actions.len()])
    }
}

impl Evaluator for Net {
    fn evaluate(&self, state: &State, actor: usize, actions: &[Action]) -> (f32, Vec<f32>) {
        let obs = encode_state(state, actor);
        let feats: Vec<_> = actions.iter().map(|a| encode_action(a, actor)).collect();
        let e = Net::evaluate(self, &obs, &feats);
        (e.value, softmax(&e.logits))
    }
}

pub fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

#[derive(Clone, Debug)]
pub struct MctsConfig {
    pub simulations: usize,
    pub c_puct: f32,
    /// Dirichlet noise on the root priors (self-play exploration); 0 disables it.
    pub noise_frac: f32,
    pub dirichlet_alpha: f32,
    /// Safety cap on decisions per simulation (games end within 30 turns anyway).
    pub max_depth: usize,
    /// Gumbel root with paired determinizations (see the module docs); false = plain PUCT root
    /// with visit-count targets.
    pub gumbel: bool,
    /// Gumbel noise on the root (self-play exploration); off for evaluation play.
    pub gumbel_noise: bool,
    /// Most root actions entering sequential halving.
    pub max_considered: usize,
    /// Scale of the Q-values against the prior logits: sigma(q) = (c_visit + max N) * c_scale * q,
    /// q mapped to [0, 1].
    pub c_visit: f32,
    pub c_scale: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            simulations: 128,
            c_puct: 1.5,
            noise_frac: 0.0,
            dirichlet_alpha: 0.5,
            max_depth: 400,
            gumbel: true,
            gumbel_noise: false,
            max_considered: 16,
            c_visit: 50.0,
            c_scale: 0.5,
        }
    }
}

fn sample_gumbel(rng: &mut StdRng) -> f32 {
    let u: f32 = rng.gen::<f32>().clamp(1e-7, 1.0 - 1e-7);
    -(-u.ln()).ln()
}

/// Root of a Gumbel search. Simulations are handed out as (root action, determinization seed):
/// the candidates still in contention all get the same seeds within a phase.
pub(crate) struct GumbelRoot {
    pub(crate) keys: Vec<u64>,
    logits: Vec<f32>,
    gumbel: Vec<f32>,
    remaining: Vec<usize>,
    phases: usize,
    budget: usize,
    queue: Vec<(usize, u64)>,
    started: bool,
}

impl GumbelRoot {
    pub(crate) fn new(
        cfg: &MctsConfig,
        actions: &[Action],
        priors: &[f32],
        rng: &mut StdRng,
    ) -> Self {
        let keys: Vec<u64> = actions.iter().map(|a| action_key(&a.action)).collect();
        let logits: Vec<f32> = priors.iter().map(|p| p.max(1e-12).ln()).collect();
        let gumbel: Vec<f32> = (0..actions.len())
            .map(|_| {
                if cfg.gumbel_noise {
                    sample_gumbel(rng)
                } else {
                    0.0
                }
            })
            .collect();
        let m = cfg.max_considered.max(1).min(actions.len());
        let mut order: Vec<usize> = (0..actions.len()).collect();
        order.sort_by(|&a, &b| (gumbel[b] + logits[b]).total_cmp(&(gumbel[a] + logits[a])));
        order.truncate(m);
        GumbelRoot {
            keys,
            logits,
            gumbel,
            remaining: order,
            phases: (m as f32).log2().ceil().max(1.0) as usize,
            budget: cfg.simulations,
            queue: Vec::new(),
            started: false,
        }
    }

    fn sigma(cfg: &MctsConfig, q: f32, max_n: u32) -> f32 {
        (cfg.c_visit + max_n as f32) * cfg.c_scale * (q + 1.0) * 0.5
    }

    /// Visits and summed value of each root action (root player's point of view).
    fn stats(&self, root: &Node) -> Vec<(u32, f32)> {
        self.keys
            .iter()
            .map(|k| root.children.get(k).map_or((0, 0.0), |c| (c.n, c.w)))
            .collect()
    }

    /// The next simulation to run, or None when the search is over.
    pub(crate) fn next(
        &mut self,
        root: &Node,
        cfg: &MctsConfig,
        rng: &mut StdRng,
    ) -> Option<(usize, u64)> {
        if let Some(x) = self.queue.pop() {
            return Some(x);
        }
        if self.budget == 0 || self.remaining.len() <= 1 {
            return None;
        }
        if self.started {
            // A phase just ended: the better half goes on.
            let stats = self.stats(root);
            let max_n = stats.iter().map(|s| s.0).max().unwrap_or(0);
            let score = |a: usize| {
                let (n, w) = stats[a];
                let q = if n > 0 { w / n as f32 } else { -1.0 };
                self.gumbel[a] + self.logits[a] + Self::sigma(cfg, q, max_n)
            };
            self.remaining
                .sort_by(|&a, &b| score(b).total_cmp(&score(a)));
            self.remaining.truncate(self.remaining.len().div_ceil(2));
            if self.remaining.len() <= 1 {
                return None;
            }
        }
        self.started = true;
        let per = (self.budget / (self.phases * self.remaining.len())).max(1);
        for _ in 0..per {
            let seed: u64 = rng.gen();
            for &a in self.remaining.iter().rev() {
                self.queue.push((a, seed));
            }
        }
        self.queue.pop()
    }

    /// (action to play, improved policy over all actions, value for the player to move).
    pub(crate) fn finish(&self, root: &Node, cfg: &MctsConfig, root_value: f32) -> (usize, Vec<f32>, f32) {
        let stats = self.stats(root);
        let max_n = stats.iter().map(|s| s.0).max().unwrap_or(0);
        let sum_n: u32 = stats.iter().map(|s| s.0).sum();
        // Unvisited actions get the value of the prior policy, estimated from the root value and
        // the visited actions' Q-values ("completed Q", v_mix in the paper).
        let (pq, pw) = stats
            .iter()
            .zip(&self.logits)
            .filter(|(s, _)| s.0 > 0)
            .fold((0.0f32, 0.0f32), |(pq, pw), (s, l)| {
                let p = l.exp();
                (pq + p * s.1 / s.0 as f32, pw + p)
            });
        let v_mix = if pw > 0.0 {
            (root_value + sum_n as f32 * pq / pw) / (1.0 + sum_n as f32)
        } else {
            root_value
        };
        let completed: Vec<f32> = stats
            .iter()
            .map(|s| if s.0 > 0 { s.1 / s.0 as f32 } else { v_mix })
            .collect();
        let scores: Vec<f32> = (0..self.keys.len())
            .map(|a| self.logits[a] + Self::sigma(cfg, completed[a], max_n))
            .collect();
        let policy = softmax(&scores);
        let value = policy.iter().zip(&completed).map(|(p, q)| p * q).sum();
        let best = self
            .remaining
            .iter()
            .copied()
            .max_by(|&a, &b| (self.gumbel[a] + scores[a]).total_cmp(&(self.gumbel[b] + scores[b])))
            .unwrap_or(0);
        (best, policy, value)
    }
}

#[derive(Default)]
pub(crate) struct Child {
    pub(crate) n: u32,
    pub(crate) w: f32, // sum of values, root player's point of view
    pub(crate) node: Node,
}

#[derive(Default)]
pub(crate) struct Node {
    pub(crate) expanded: bool,
    pub(crate) priors: HashMap<u64, f32>,
    pub(crate) children: HashMap<u64, Child>,
    pub(crate) n: u32,
}

pub(crate) fn terminal_value(state: &State, root: usize) -> f32 {
    match state.winner {
        Some(GameOutcome::Win(p)) if p == root => 1.0,
        Some(GameOutcome::Win(_)) => -1.0,
        _ => 0.0,
    }
}

struct Search<'a, E: Evaluator> {
    eval: &'a E,
    cfg: &'a MctsConfig,
    root: usize,
}

impl<E: Evaluator> Search<'_, E> {
    fn expand(&self, node: &mut Node, state: &State, actor: usize, actions: &[Action]) -> f32 {
        let (v, priors) = self.eval.evaluate(state, actor, actions);
        for (a, p) in actions.iter().zip(priors) {
            node.priors.insert(action_key(&a.action), p);
        }
        node.expanded = true;
        if actor == self.root {
            v
        } else {
            -v
        }
    }

    /// Plays forced moves, then either expands a new decision or descends one level.
    fn descend(&self, node: &mut Node, state: &mut State, rng: &mut StdRng, depth: usize) -> f32 {
        let (actor, actions) = loop {
            if state.is_game_over() {
                return terminal_value(state, self.root);
            }
            let (actor, actions) = state.generate_possible_actions();
            if actions.len() == 1 {
                apply_action(rng, state, &actions[0]);
                continue;
            }
            break (actor, actions);
        };
        if !node.expanded || depth >= self.cfg.max_depth {
            let v = self.expand(node, state, actor, &actions);
            node.n += 1;
            return v;
        }

        let sign = if actor == self.root { 1.0 } else { -1.0 };
        let sqrt_n = (node.n.max(1) as f32).sqrt();
        let fallback = 1.0 / actions.len() as f32;
        let mut best = (f32::NEG_INFINITY, 0usize, 0u64);
        for (i, a) in actions.iter().enumerate() {
            let key = action_key(&a.action);
            let p = node.priors.get(&key).copied().unwrap_or(fallback);
            let (n, q) = match node.children.get(&key) {
                Some(c) if c.n > 0 => (c.n, sign * c.w / c.n as f32),
                _ => (0, 0.0),
            };
            let u = q + self.cfg.c_puct * p * sqrt_n / (1.0 + n as f32);
            if u > best.0 {
                best = (u, i, key);
            }
        }
        let (_, i, key) = best;
        apply_action(rng, state, &actions[i]);
        let child = node.children.entry(key).or_default();
        let v = self.descend(&mut child.node, state, rng, depth + 1);
        child.n += 1;
        child.w += v;
        node.n += 1;
        v
    }
}

/// Result of a search at a decision point.
pub struct SearchResult {
    /// Visit count per legal action, in the order of `actions`.
    pub visits: Vec<u32>,
    /// Policy target (sums to 1), in the order of `actions`.
    pub policy: Vec<f32>,
    /// The action the search picks.
    pub best: usize,
    /// Root value estimate for the player to move.
    pub value: f32,
}

/// Searches `state` for `actor`, whose legal actions are `actions` (as generated from `state`).
pub fn search<E: Evaluator>(
    eval: &E,
    cfg: &MctsConfig,
    state: &State,
    actor: usize,
    actions: &[Action],
    rng: &mut StdRng,
) -> SearchResult {
    let s = Search {
        eval,
        cfg,
        root: actor,
    };
    let mut root = Node::default();
    if cfg.gumbel {
        let (root_v, priors) = eval.evaluate(state, actor, actions);
        for (a, p) in actions.iter().zip(&priors) {
            root.priors.insert(action_key(&a.action), *p);
        }
        root.expanded = true;
        root.n = 1;
        let mut plan = GumbelRoot::new(cfg, actions, &priors, rng);
        while let Some((a, seed)) = plan.next(&root, cfg, rng) {
            let mut srng = StdRng::seed_from_u64(seed);
            let mut sim = determinize(state, actor, &mut srng);
            apply_action(&mut srng, &mut sim, &actions[a]);
            let child = root.children.entry(plan.keys[a]).or_default();
            let v = s.descend(&mut child.node, &mut sim, &mut srng, 1);
            child.n += 1;
            child.w += v;
            root.n += 1;
        }
        let (best, policy, value) = plan.finish(&root, cfg, root_v);
        let visits = plan
            .keys
            .iter()
            .map(|k| root.children.get(k).map_or(0, |c| c.n))
            .collect();
        return SearchResult {
            visits,
            policy,
            best,
            value,
        };
    }
    let root_v = s.expand(&mut root, state, actor, actions);
    if cfg.noise_frac > 0.0 {
        let noise: Vec<f32> = (0..actions.len())
            .map(|_| sample_gamma(rng, cfg.dirichlet_alpha))
            .collect();
        let total: f32 = noise.iter().sum::<f32>().max(1e-9);
        for (a, n) in actions.iter().zip(noise) {
            let p = root.priors.entry(action_key(&a.action)).or_insert(0.0);
            *p = (1.0 - cfg.noise_frac) * *p + cfg.noise_frac * n / total;
        }
    }
    root.n = 1;
    for _ in 0..cfg.simulations {
        let mut sim = determinize(state, actor, rng);
        // The root decision itself is taken on the determinized copy (same legal actions for
        // the root player, whose information is unchanged).
        s.descend(&mut root, &mut sim, rng, 0);
    }
    let visits: Vec<u32> = actions
        .iter()
        .map(|a| root.children.get(&action_key(&a.action)).map_or(0, |c| c.n))
        .collect();
    let total_w: f32 = root.children.values().map(|c| c.w).sum();
    let total_n: u32 = root.children.values().map(|c| c.n).sum();
    let total = visits.iter().sum::<u32>().max(1) as f32;
    SearchResult {
        policy: visits.iter().map(|&n| n as f32 / total).collect(),
        best: pick(&visits, 0.0, rng),
        visits,
        value: if total_n > 0 {
            total_w / total_n as f32
        } else {
            root_v
        },
    }
}

/// Picks an action index from visit counts: proportional to N^(1/temperature), or the most
/// visited one when `temperature` is 0.
pub fn pick(visits: &[u32], temperature: f32, rng: &mut StdRng) -> usize {
    if temperature <= 0.0 {
        return visits
            .iter()
            .enumerate()
            .max_by_key(|(_, n)| **n)
            .map(|(i, _)| i)
            .unwrap_or(0);
    }
    let w: Vec<f64> = visits
        .iter()
        .map(|&n| (n as f64).powf(1.0 / temperature as f64))
        .collect();
    let total: f64 = w.iter().sum();
    if total <= 0.0 {
        return rng.gen_range(0..visits.len());
    }
    let mut r = rng.gen::<f64>() * total;
    for (i, x) in w.iter().enumerate() {
        r -= x;
        if r <= 0.0 {
            return i;
        }
    }
    visits.len() - 1
}

/// Gamma sampling for Dirichlet noise (Marsaglia-Tsang), so no extra dependency is needed.
mod rand_distr_gamma {
    use rand::rngs::StdRng;
    use rand::Rng;

    fn normal(rng: &mut StdRng) -> f32 {
        // Box-Muller
        let u1: f32 = rng.gen::<f32>().max(1e-12);
        let u2: f32 = rng.gen();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    pub fn sample_gamma(rng: &mut StdRng, alpha: f32) -> f32 {
        if alpha < 1.0 {
            let u: f32 = rng.gen::<f32>().max(1e-12);
            return sample_gamma(rng, alpha + 1.0) * u.powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = normal(rng);
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u: f32 = rng.gen();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }
}
