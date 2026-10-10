use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use rand::rngs::StdRng;

use super::encode::encode_state;
use super::mcts::{search, Bootstrap, Evaluator, MctsConfig};
use super::net::Net;
use crate::actions::Action;
use crate::players::{baseline_value_function, ExpectiMiniMaxPlayer, Player, ValueFunction};
use crate::state::GameOutcome;
use crate::{Deck, State};

static NETS: LazyLock<Mutex<HashMap<String, Arc<Net>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Loads a weight file once per process and shares it between players and threads.
pub fn load_net(path: &str) -> Result<Arc<Net>, String> {
    let mut cache = NETS.lock().unwrap();
    if let Some(net) = cache.get(path) {
        return Ok(net.clone());
    }
    let net = Arc::new(Net::load(path)?);
    cache.insert(path.to_string(), net.clone());
    Ok(net)
}

pub struct ZeroPlayer {
    deck: Deck,
    net: Option<Arc<Net>>,
    cfg: MctsConfig,
    last_scores: Option<Vec<f64>>,
}

impl std::fmt::Debug for ZeroPlayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Zero {{ sims: {}, net: {} }}",
            self.cfg.simulations,
            self.net.is_some()
        )
    }
}

impl ZeroPlayer {
    pub fn new(deck: Deck, sims: usize, net: Option<Arc<Net>>) -> Self {
        Self::with_config(
            deck,
            net,
            MctsConfig {
                simulations: sims,
                ..MctsConfig::default()
            },
        )
    }

    pub fn with_config(deck: Deck, net: Option<Arc<Net>>, cfg: MctsConfig) -> Self {
        ZeroPlayer {
            deck,
            net,
            cfg,
            last_scores: None,
        }
    }

    /// Weights from POCKET_ZERO_WEIGHTS; without it the player searches with the bootstrap
    /// evaluator (handy before the first training run).
    pub fn from_env(deck: Deck, sims: usize) -> Self {
        let net = std::env::var("POCKET_ZERO_WEIGHTS")
            .ok()
            .filter(|p| !p.is_empty())
            .map(|p| load_net(&p).unwrap_or_else(|e| panic!("POCKET_ZERO_WEIGHTS: {e}")));
        Self::new(deck, sims, net)
    }

    fn choose<E: Evaluator>(
        &mut self,
        eval: &E,
        rng: &mut StdRng,
        state: &State,
        actions: &[Action],
    ) -> usize {
        let actor = actions[0].actor;
        if self.cfg.simulations == 0 {
            let (_, priors) = eval.evaluate(state, actor, actions);
            self.last_scores = Some(priors.iter().map(|p| *p as f64).collect());
            return priors
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
        let res = search(eval, &self.cfg, state, actor, actions, rng);
        self.last_scores = Some(res.policy.iter().map(|p| *p as f64).collect());
        res.best
    }
}

impl Player for ZeroPlayer {
    fn get_deck(&self) -> Deck {
        self.deck.clone()
    }

    fn decision_fn(&mut self, rng: &mut StdRng, state: &State, actions: &[Action]) -> Action {
        if actions.len() == 1 {
            return actions[0].clone();
        }
        let i = match self.net.clone() {
            Some(net) => self.choose(net.as_ref(), rng, state, actions),
            None => self.choose(&Bootstrap, rng, state, actions),
        };
        actions[i].clone()
    }

    fn last_scores(&self) -> Option<Vec<f64>> {
        self.last_scores.clone()
    }
}

/// The engine's evaluation replaced by the network's: win probability (as a value in [-1, 1])
/// from `me`'s point of view. The network is always asked from the side about to act, the
/// viewpoint it was trained on.
pub fn net_value_function(net: Arc<Net>) -> ValueFunction {
    Box::new(move |state: &State, me: usize| match state.winner {
        Some(GameOutcome::Win(p)) => {
            if p == me {
                1.0
            } else {
                -1.0
            }
        }
        Some(GameOutcome::Tie) => 0.0,
        None => {
            let actor = state.current_player;
            let v = net.evaluate(&encode_state(state, actor), &[]).value as f64;
            if actor == me {
                v
            } else {
                -v
            }
        }
    })
}

/// Expert for expert iteration: the exact chance-aware search of the `e` players (every coin
/// flip and draw weighted by its real probability) with the network as evaluation. Its scores,
/// turned into probabilities, become the policy targets of the next network.
pub struct NetSearchPlayer {
    inner: ExpectiMiniMaxPlayer,
    temperature: f64,
    last_scores: Option<Vec<f64>>,
}

impl std::fmt::Debug for NetSearchPlayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NetSearch {{ depth: {} }}", self.inner.max_depth)
    }
}

impl NetSearchPlayer {
    /// Without a network, falls back to the engine's hand-written evaluation (scaled like the
    /// bootstrap evaluator), i.e. an `e<depth>` player with soft scores.
    pub fn new(deck: Deck, depth: usize, net: Option<Arc<Net>>) -> Self {
        let value_function: ValueFunction = match net {
            Some(net) => net_value_function(net),
            None => {
                Box::new(|s: &State, me: usize| (baseline_value_function(s, me) / 15_000.0).tanh())
            }
        };
        NetSearchPlayer {
            inner: ExpectiMiniMaxPlayer {
                deck,
                max_depth: depth.max(1),
                write_debug_trees: false,
                value_function,
            },
            temperature: 0.05,
            last_scores: None,
        }
    }

    pub fn from_env(deck: Deck, depth: usize) -> Self {
        let net = std::env::var("POCKET_ZERO_WEIGHTS")
            .ok()
            .filter(|p| !p.is_empty())
            .map(|p| load_net(&p).unwrap_or_else(|e| panic!("POCKET_ZERO_WEIGHTS: {e}")));
        Self::new(deck, depth, net)
    }
}

impl Player for NetSearchPlayer {
    fn get_deck(&self) -> Deck {
        self.inner.deck.clone()
    }

    fn decision_fn(&mut self, rng: &mut StdRng, state: &State, actions: &[Action]) -> Action {
        if actions.len() == 1 {
            self.last_scores = Some(vec![1.0]);
            return actions[0].clone();
        }
        let scores = self.inner.score_actions(rng, state, actions);
        let best = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        // Softmax over expected values: near-equal moves share the probability mass.
        let top = scores[best];
        let w: Vec<f64> = scores
            .iter()
            .map(|s| ((s - top) / self.temperature).exp())
            .collect();
        let sum: f64 = w.iter().sum();
        self.last_scores = Some(w.iter().map(|x| x / sum).collect());
        actions[best].clone()
    }

    /// Probabilities (sum to 1), usable directly as a policy target.
    fn last_scores(&self) -> Option<Vec<f64>> {
        self.last_scores.clone()
    }
}
