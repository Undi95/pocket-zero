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
        /// Plain PUCT root (visit-count targets) instead of the Gumbel root
        #[arg(long)]
        puct: bool,
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
        /// Another player for deck B (who misplays a matchup?); default: the same as --player
        #[arg(long)]
        player_b: Option<String>,
        #[arg(long, default_value_t = 100)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// League: every deck of a folder plays about `games_per_deck` games against random others
    /// of the folder, the same player on both sides; wins per deck as JSON
    League {
        #[arg(long)]
        decks: String,
        #[arg(long)]
        player: String,
        #[arg(long, default_value_t = 40)]
        games_per_deck: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Value diagnostics: at every decision of games played by `player`, the hand-written
    /// evaluation, the network value and the final result, as CSV
    Valuecheck {
        #[arg(long)]
        decks: String,
        #[arg(long)]
        games: usize,
        #[arg(long, default_value = "e3")]
        player: String,
        #[arg(long)]
        weights: String,
        #[arg(long)]
        out: String,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Engine speed: the GPU self-play loop with a dummy evaluator (no network), evaluations/s
    Bench {
        #[arg(long, default_value = "lab/decks/meta")]
        decks: String,
        #[arg(long, default_value_t = 200)]
        games: usize,
        #[arg(long, default_value_t = 64)]
        sims: usize,
        #[arg(long, default_value_t = 256)]
        concurrency: usize,
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
    load_named_decks(dir).into_iter().map(|(_, d)| d).collect()
}

/// Valid decks of a folder (or a single file), with their file stems.
fn load_named_decks(dir: &str) -> Vec<(String, Deck)> {
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
    let decks: Vec<(String, Deck)> = files
        .iter()
        .filter_map(|p| {
            let d = Deck::from_file(p.to_str()?).ok()?;
            let stem = p.file_stem()?.to_string_lossy().into_owned();
            (d.is_valid()).then_some((stem, d))
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
            if let Some(x) = std::env::var("PZ_CONSIDER")
                .ok()
                .and_then(|v| v.parse().ok())
            {
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
                "layout": encode::LAYOUT,
                "state_dense_v1": encode::STATE_DENSE_V1,
                "trunk_in_v1": net::TRUNK_IN_V1,
                "attached_start": encode::ATTACHED_START,
                "attached_max": encode::ATTACHED_MAX,
                "vocab_table": net::vocab_table(),
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
            puct,
        } => {
            let decks = load_decks(&decks);
            let cfg = SelfPlayConfig {
                mcts: MctsConfig {
                    simulations: sims,
                    noise_frac: noise,
                    gumbel: !puct,
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
            let done = AtomicUsize::new(0);
            // +1 a wins, -1 b wins, 0 tie; in game order, for the win streaks.
            let results: Vec<i8> = (0..games)
                .into_par_iter()
                .map(|g| {
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
                    let r = match game.play() {
                        Some(GameOutcome::Win(p)) if p == a_seat => 1,
                        Some(GameOutcome::Win(_)) => -1,
                        _ => 0,
                    };
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if n.is_multiple_of(20) {
                        eprintln!("arena {n}/{games}");
                    }
                    r
                })
                .collect();
            let count = |v: i8| results.iter().filter(|&&r| r == v).count();
            let (aw, bw, t) = (count(1), count(-1), count(0));
            let streak = |v: i8| {
                let (mut best, mut cur) = (0usize, 0usize);
                for &r in &results {
                    cur = if r == v { cur + 1 } else { 0 };
                    best = best.max(cur);
                }
                best
            };
            println!(
                "{}",
                json!({"a": a, "b": b, "a_wins": aw, "b_wins": bw, "ties": t,
                       "a_score": (aw as f64 + t as f64 / 2.0) / games.max(1) as f64,
                       "a_streak": streak(1), "b_streak": streak(-1)})
            );
        }
        Cmd::Match {
            deck_a,
            deck_b,
            player,
            player_b,
            games,
            seed,
        } => {
            let player_b = player_b.unwrap_or_else(|| player.clone());
            let da = Deck::from_file(&deck_a).unwrap_or_else(|e| panic!("{deck_a}: {e}"));
            let db = Deck::from_file(&deck_b).unwrap_or_else(|e| panic!("{deck_b}: {e}"));
            let results: Vec<f64> = (0..games)
                .into_par_iter()
                .map(|g| {
                    let mut rng =
                        StdRng::seed_from_u64(seed.wrapping_mul(104_729).wrapping_add(g as u64));
                    let players = vec![
                        make_player(&player, da.clone()),
                        make_player(&player_b, db.clone()),
                    ];
                    match Game::new(players, rng.gen()).play() {
                        Some(GameOutcome::Win(0)) => 1.0,
                        Some(GameOutcome::Win(_)) => 0.0,
                        _ => 0.5,
                    }
                })
                .collect();
            let score = results.iter().sum::<f64>() / games.max(1) as f64;
            let ties = results.iter().filter(|&&r| r == 0.5).count() as f64 / games.max(1) as f64;
            println!(
                "{}",
                json!({"deck_a": deck_a, "deck_b": deck_b, "games": games, "score_a": score, "ties": ties})
            );
        }
        Cmd::League {
            decks,
            player,
            games_per_deck,
            seed,
        } => {
            let decks = load_named_decks(&decks);
            let n = decks.len();
            assert!(n >= 2, "a league needs at least two decks");
            // Game g: deck g % n against a random other one, seats random (Game::new flips).
            let games = (games_per_deck * n).div_ceil(2);
            let done = AtomicUsize::new(0);
            let results: Vec<(usize, usize, f64)> = (0..games)
                .into_par_iter()
                .map(|g| {
                    let mut rng =
                        StdRng::seed_from_u64(seed.wrapping_mul(31_337).wrapping_add(g as u64));
                    let i = g % n;
                    let j = (i + rng.gen_range(1..n)) % n;
                    let players = vec![
                        make_player(&player, decks[i].1.clone()),
                        make_player(&player, decks[j].1.clone()),
                    ];
                    let score = match Game::new(players, rng.gen()).play() {
                        Some(GameOutcome::Win(0)) => 1.0,
                        Some(GameOutcome::Win(_)) => 0.0,
                        _ => 0.5,
                    };
                    let k = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if k.is_multiple_of(100) {
                        eprintln!("league {k}/{games}");
                    }
                    (i, j, score)
                })
                .collect();
            let mut table = vec![(0.0f64, 0usize); n];
            for (i, j, s) in results {
                table[i].0 += s;
                table[i].1 += 1;
                table[j].0 += 1.0 - s;
                table[j].1 += 1;
            }
            let per: serde_json::Map<String, serde_json::Value> = decks
                .iter()
                .zip(&table)
                .map(|((name, _), (w, g))| (name.clone(), json!([w, g])))
                .collect();
            println!("{}", json!({"games": games, "decks": per}));
        }
        Cmd::Valuecheck {
            decks,
            games,
            player,
            weights,
            out,
            seed,
        } => {
            let decks = load_decks(&decks);
            let net = Net::load(&weights).unwrap_or_else(|e| panic!("{e}"));
            let rows =
                selfplay::value_check(|d| make_player(&player, d), &net, &decks, games, seed);
            let mut csv = String::from("game,turn,actor,baseline,net,z\n");
            for r in &rows {
                csv.push_str(&format!(
                    "{},{},{},{},{},{}\n",
                    r.game, r.turn, r.actor, r.baseline, r.net, r.z
                ));
            }
            std::fs::write(&out, csv).unwrap_or_else(|e| panic!("{out}: {e}"));
            println!("{}", json!({"games": games, "rows": rows.len()}));
        }
        Cmd::Bench {
            decks,
            games,
            sims,
            concurrency,
        } => {
            let decks = load_decks(&decks);
            let cfg = SelfPlayConfig {
                mcts: MctsConfig {
                    simulations: sims,
                    noise_frac: 0.25,
                    gumbel_noise: true,
                    ..MctsConfig::default()
                },
                temperature_moves: 8,
            };
            let out = std::env::temp_dir().join("pz_bench.bin");
            let names = (0..decks.len()).map(|i| format!("deck{i}")).collect();
            let mut bsp = deckgym::zero::batch::BatchSelfPlay::new(
                cfg,
                decks,
                names,
                games,
                concurrency,
                1,
                out.to_str().unwrap(),
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("{e}"));
            let start = std::time::Instant::now();
            let (mut evals, mut rounds) = (0usize, 0usize);
            let mut t_collect = std::time::Duration::ZERO;
            let mut t_respond = std::time::Duration::ZERO;
            while !bsp.done() {
                let t0 = std::time::Instant::now();
                let b = bsp.collect().unwrap_or_else(|e| panic!("{e}"));
                t_collect += t0.elapsed();
                if b.n == 0 {
                    break;
                }
                evals += b.n;
                rounds += 1;
                let t1 = std::time::Instant::now();
                bsp.respond(&vec![0.0; b.n], &vec![0.0; b.owner.len()]);
                t_respond += t1.elapsed();
            }
            let secs = start.elapsed().as_secs_f64();
            let (done, _, samples, _, _) = bsp.progress();
            let _ = std::fs::remove_file(&out);
            println!(
                "{}",
                json!({"games": done, "samples": samples, "evals": evals, "rounds": rounds,
                       "seconds": secs, "evals_per_s": evals as f64 / secs,
                       "collect_s": t_collect.as_secs_f64(), "respond_s": t_respond.as_secs_f64()})
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
