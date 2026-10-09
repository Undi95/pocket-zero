use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use rand::rngs::StdRng;

use super::mcts::{pick, search, Bootstrap, Evaluator, MctsConfig};
use super::net::Net;
use crate::actions::Action;
use crate::players::Player;
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
        ZeroPlayer {
            deck,
            net,
            cfg: MctsConfig {
                simulations: sims,
                ..MctsConfig::default()
            },
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
        self.last_scores = Some(res.visits.iter().map(|n| *n as f64).collect());
        pick(&res.visits, 0.0, rng)
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
