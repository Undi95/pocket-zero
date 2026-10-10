//! pocket-zero command line: self-play data, arena matches and helpers for the Python trainer.
//!
//!   pocket-zero info
//!   pocket-zero selfplay --decks DIR --games N --sims S --out FILE [--weights W]
//!   pocket-zero arena --a z64:W --b e2 --decks DIR --games N
//!   pocket-zero parity --weights W --shard FILE --n 16

use std::io::Read;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use clap::{Parser, Subcommand};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde_json::json;

use deckgym::players::{create_players, parse_player_code, Player, PlayerCode};
use deckgym::state::GameOutcome;
use deckgym::zero::encode::{self, ActionFeatures, Obs};
use deckgym::zero::mcts::{Bootstrap, MctsConfig};
use deckgym::zero::net::{self, Net};
use deckgym::zero::selfplay::{self, SelfPlayConfig, MAX_ACTIONS};
use deckgym::zero::{load_net, NetSearchPlayer, ZeroPlayer};
use deckgym::{Deck, Game};

#[derive(Parser)]
#[command(name = "pocket-zero")]
struct Cli {
    /// Worker threads (default: all cores)
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the tensor layout as JSON (read by the Python trainer)
    Info,
    /// Generate training samples by self-play
    Selfplay {
        #[arg(long)]
        decks: String,
        #[arg(long)]
        games: usize,
        #[arg(long, default_value_t = 128)]
        sims: usize,
        #[arg(long)]
        out: String,
        /// Weight file; omit for the untrained bootstrap evaluator
        #[arg(long)]
        weights: Option<String>,
        #[arg(long, default_value_t = 20)]
        temperature_moves: usize,
        #[arg(long, default_value_t = 0.25)]
        noise: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Imitation data: an engine player (default e3) plays both sides, its moves are the targets
    Imitate {
        #[arg(long)]
        decks: String,
        #[arg(long)]
        games: usize,
        #[arg(long)]
        out: String,
        #[arg(long, default_value = "e3")]
        player: String,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Play two players against each other on random deck pairs (each pair played both ways)
    Arena {
        /// Player spec: z<sims>[:weights], e<depth>, r, ...
        #[arg(long)]
        a: String,
        #[arg(long)]
        b: String,
        #[arg(long)]
        decks: String,
        #[arg(long, default_value_t = 100)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// One matchup with the same player on both sides: score of deck A (validation vs real data)
    Match {
        #[arg(long)]
        deck_a: String,
        #[arg(long)]
        deck_b: String,
        /// Player spec used by both sides: z<sims>[:weights], e<depth>, ...
        #[arg(long)]
        player: String,
        #[arg(long, default_value_t = 100)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Evaluate the network on the first records of a shard (Python/Rust parity check)
    Parity {
        #[arg(long)]
        weights: String,
        #[arg(long)]
        shard: String,
        #[arg(long, default_value_t = 16)]
        n: usize,
    },
}

fn load_decks(dir: &str) -> Vec<Deck> {
    let path = std::path::Path::new(dir);
    let mut files: Vec<_> = if path.is_dir() {
        std::fs::read_dir(path)
            .unwrap_or_else(|e| panic!("{dir}: {e}"))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "txt"))
            .collect()
    } else {
        vec![path.to_path_buf()]
    };
    files.sort();
    let decks: Vec<Deck> = files
        .iter()
        .filter_map(|p| {
            let d = Deck::from_file(p.to_str()?).ok()?;
            (d.is_valid()).then_some(d)
        })
        .collect();
    assert!(!decks.is_empty(), "no valid deck in {dir}");
    decks
}

fn make_player(spec: &str, deck: Deck) -> Box<dyn Player> {
    let (code, weights) = match spec.split_once(':') {
        Some((c, w)) => (c, Some(w)),
        None => (spec, None),
    };
    // zp<sims>: the plain PUCT root (visit counts), kept for comparisons.
    if let Some(sims) = code
        .strip_prefix("zp")
        .and_then(|r| r.parse::<usize>().ok())
    {
        let net = weights.map(|w| load_net(w).unwrap_or_else(|e| panic!("{e}")));
        let cfg = MctsConfig {
            simulations: sims,
            gumbel: false,
            ..MctsConfig::default()
        };
        return Box::new(ZeroPlayer::with_config(deck, net, cfg));
    }
    match parse_player_code(code).unwrap_or_else(|e| panic!("{e}")) {
        PlayerCode::Z { sims } => {
            let net = weights.map(|w| load_net(w).unwrap_or_else(|e| panic!("{e}")));
            let mut cfg = MctsConfig {
                simulations: sims,
                ..MctsConfig::default()
            };
            // Search experiments: PZ_CSCALE / PZ_CONSIDER override the Gumbel settings.
            if let Some(x) = std::env::var("PZ_CSCALE").ok().and_then(|v| v.parse().ok()) {
                cfg.c_scale = x;
            }
            if let Some(x) = std::env::var("PZ_CONSIDER").ok().and_then(|v| v.parse().ok()) {
                cfg.max_considered = x;
            }
            Box::new(ZeroPlayer::with_config(deck, net, cfg))
        }
        PlayerCode::Y { depth } => {
            let net = weights.map(|w| load_net(w).unwrap_or_else(|e| panic!("{e}")));
            Box::new(NetSearchPlayer::new(deck, depth, net))
        }
        other => create_players(deck.clone(), deck, vec![other, PlayerCode::R])
            .into_iter()
            .next()
            .unwrap(),
    }
}

fn read_shard_obs(path: &str, n: usize) -> Vec<(Obs, Vec<ActionFeatures>)> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .and_then(|mut f| f.read_to_end(&mut bytes))
        .unwrap_or_else(|e| panic!("{path}: {e}"));
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as usize;
    let extra = match &bytes[..4] {
        b"PZS1" => 2,
        b"PZS2" => 4,
        _ => panic!("{path}: not a pocket-zero shard"),
    };
    let (sd, nt, ad, at, ma) = (u32_at(4), u32_at(8), u32_at(12), u32_at(16), u32_at(20));
    let rec = 4 * (sd + nt + 1 + ma * (ad + at + 1) + extra);
    let f32_at = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let mut out = Vec::new();
    let mut off = 24;
    while off + rec <= bytes.len() && out.len() < n {
        let dense = (0..sd).map(|i| f32_at(off + 4 * i)).collect();
        let tokens = (0..nt).map(|i| u32_at(off + 4 * (sd + i)) as u32).collect();
        let na = u32_at(off + 4 * (sd + nt));
        let base_d = off + 4 * (sd + nt + 1);
        let base_t = base_d + 4 * ma * ad;
        let actions = (0..na)
            .map(|k| {
                let mut f = ActionFeatures {
                    dense: [0.0; encode::ACTION_DENSE],
                    tokens: [0; encode::ACTION_TOKENS],
                };
                for i in 0..ad {
                    f.dense[i] = f32_at(base_d + 4 * (k * ad + i));
                }
                for i in 0..at {
                    f.tokens[i] = u32_at(base_t + 4 * (k * at + i)) as u32;
                }
                f
            })
            .collect();
        out.push((Obs { dense, tokens }, actions));
        off += rec;
    }
    out
}

fn main() {
    let cli = Cli::parse();
    if let Some(t) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .expect("thread pool");
    }
    match cli.cmd {
        Cmd::Info => {
            let info = json!({
                "state_dense": encode::STATE_DENSE,
                "n_tokens": encode::N_TOKENS,
                "n_slots": encode::N_SLOTS,
                "bags": encode::BAGS,
                "action_dense": encode::ACTION_DENSE,
                "action_tokens": encode::ACTION_TOKENS,
                "action_type_buckets": encode::ACTION_TYPE_BUCKETS,
                "action_detail_buckets": encode::ACTION_DETAIL_BUCKETS,
                "max_actions": MAX_ACTIONS,
                "vocab": encode::VOCAB.size,
                "embed_dim": encode::EMBED_DIM,
                "action_embed": net::ACTION_EMBED,
                "h1": net::H1, "h2": net::H2, "h": net::H, "vh": net::VH, "ph": net::PH,
                "trunk_in": net::TRUNK_IN,
                "action_in": net::ACTION_IN,
            });
            println!("{}", serde_json::to_string_pretty(&info).unwrap());
        }
        Cmd::Selfplay {
            decks,
            games,
            sims,
            out,
            weights,
            temperature_moves,
            noise,
            seed,
        } => {
            let decks = load_decks(&decks);
            let cfg = SelfPlayConfig {
                mcts: MctsConfig {
                    simulations: sims,
                    noise_frac: noise,
                    gumbel_noise: noise > 0.0,
                    ..MctsConfig::default()
                },
                temperature_moves,
            };
            let stats = match weights {
                Some(w) => {
                    let net = Net::load(&w).unwrap_or_else(|e| panic!("{e}"));
                    selfplay::run(&net, &cfg, &decks, games, seed, &out, true)
                }
                None => selfplay::run(&Bootstrap, &cfg, &decks, games, seed, &out, true),
            }
            .unwrap_or_else(|e| panic!("{out}: {e}"));
            println!(
                "{}",
                json!({"games": stats.games, "samples": stats.samples,
                       "first_player_wins": stats.wins_first_deck, "ties": stats.ties})
            );
        }
        Cmd::Imitate {
            decks,
            games,
            out,
            player,
            seed,
        } => {
            let decks = load_decks(&decks);
            let stats = selfplay::run_imitation(
                |d| make_player(&player, d),
                &decks,
                games,
                seed,
                &out,
                true,
            )
            .unwrap_or_else(|e| panic!("{out}: {e}"));
            println!(
                "{}",
                json!({"games": stats.games, "samples": stats.samples,
                       "first_player_wins": stats.wins_first_deck, "ties": stats.ties})
            );
        }
        Cmd::Arena {
            a,
            b,
            decks,
            games,
            seed,
        } => {
            let decks = Arc::new(load_decks(&decks));
            let a_wins = AtomicUsize::new(0);
            let b_wins = AtomicUsize::new(0);
            let ties = AtomicUsize::new(0);
            let done = AtomicUsize::new(0);
            (0..games).into_par_iter().for_each(|g| {
                // Pair 2k / 2k+1 use the same decks and seed with the seats swapped.
                let pair = g / 2;
                let mut rng =
                    StdRng::seed_from_u64(seed.wrapping_mul(7919).wrapping_add(pair as u64));
                let d1 = decks[rng.gen_range(0..decks.len())].clone();
                let d2 = decks[rng.gen_range(0..decks.len())].clone();
                let game_seed: u64 = rng.gen();
                let a_first = g % 2 == 0;
                let (da, db) = if a_first { (d1, d2) } else { (d2, d1) };
                let pa = make_player(&a, da);
                let pb = make_player(&b, db);
                let players = if a_first { vec![pa, pb] } else { vec![pb, pa] };
                let mut game = Game::new(players, game_seed);
                let a_seat = if a_first { 0 } else { 1 };
                match game.play() {
                    Some(GameOutcome::Win(p)) if p == a_seat => {
                        a_wins.fetch_add(1, Ordering::Relaxed)
                    }
                    Some(GameOutcome::Win(_)) => b_wins.fetch_add(1, Ordering::Relaxed),
                    _ => ties.fetch_add(1, Ordering::Relaxed),
                };
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_multiple_of(20) {
                    eprintln!("arena {n}/{games}");
                }
            });
            let (aw, bw, t) = (
                a_wins.load(Ordering::Relaxed),
                b_wins.load(Ordering::Relaxed),
                ties.load(Ordering::Relaxed),
            );
            println!(
                "{}",
                json!({"a": a, "b": b, "a_wins": aw, "b_wins": bw, "ties": t,
                       "a_score": (aw as f64 + t as f64 / 2.0) / games.max(1) as f64})
            );
        }
        Cmd::Match {
            deck_a,
            deck_b,
            player,
            games,
            seed,
        } => {
            let da = Deck::from_file(&deck_a).unwrap_or_else(|e| panic!("{deck_a}: {e}"));
            let db = Deck::from_file(&deck_b).unwrap_or_else(|e| panic!("{deck_b}: {e}"));
            let results: Vec<f64> = (0..games)
                .into_par_iter()
                .map(|g| {
                    let mut rng =
                        StdRng::seed_from_u64(seed.wrapping_mul(104_729).wrapping_add(g as u64));
                    let players = vec![
                        make_player(&player, da.clone()),
                        make_player(&player, db.clone()),
                    ];
                    match Game::new(players, rng.gen()).play() {
                        Some(GameOutcome::Win(0)) => 1.0,
                        Some(GameOutcome::Win(_)) => 0.0,
                        _ => 0.5,
                    }
                })
                .collect();
            let score = results.iter().sum::<f64>() / games.max(1) as f64;
            println!(
                "{}",
                json!({"deck_a": deck_a, "deck_b": deck_b, "games": games, "score_a": score})
            );
        }
        Cmd::Parity { weights, shard, n } => {
            let net = Net::load(&weights).unwrap_or_else(|e| panic!("{e}"));
            let out: Vec<_> = read_shard_obs(&shard, n)
                .iter()
                .map(|(obs, acts)| {
                    let e = net.evaluate(obs, acts);
                    json!({"value": e.value, "logits": e.logits})
                })
                .collect();
            println!("{}", serde_json::to_string(&out).unwrap());
        }
    }
}
