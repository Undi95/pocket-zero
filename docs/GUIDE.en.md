# pocket-zero guide

*[Version française : GUIDE.fr.md](GUIDE.fr.md)*

This guide explains every tool in detail: what it is for, how to run it, how to read its
results. Every command runs from the project folder, in a command prompt (`cmd`) or PowerShell
(`.\run.bat ...`). `run.bat help` lists every command and its options.

> The console output of the tools (training log, reports) is in French; the web interface,
> the README and this guide are in both languages.

## Contents

1. [What pocket-zero is](#1-what-pocket-zero-is)
2. [Install](#2-install)
3. [Update](#3-update)
4. [The network: use the published one or train your own](#4-the-network-use-the-published-one-or-train-your-own)
5. [Follow the training live](#5-follow-the-training-live)
6. [The forge: improve decks, with proof](#6-the-forge-improve-decks-with-proof)
7. [Homemade decks](#7-homemade-decks)
8. [Deck search (genetic algorithm)](#8-deck-search-genetic-algorithm)
9. [Simulated ranked](#9-simulated-ranked)
10. [Solo mode decks](#10-solo-mode-decks)
11. [The interface: decklists and official images](#11-the-interface-decklists-and-official-images)
12. [The meta and its automatic updates](#12-the-meta-and-its-automatic-updates)
13. [Deck format](#13-deck-format)
14. [The bots](#14-the-bots)
15. [What the bot sees](#15-what-the-bot-sees)
16. [How it works](#how-it-works)
17. [For maintainers and forks](#17-for-maintainers-and-forks)
18. [Troubleshooting](#18-troubleshooting)
19. [Credits and license](#19-credits-and-license)

## 1. What pocket-zero is

pocket-zero has three layers:

1. **The engine**: a Pokémon TCG Pocket simulator written in Rust, from
   [deckgym-core](https://github.com/bcollazo/deckgym-core) (by bcollazo, AGPL-3.0). It knows the
   rules and every card up to the Deluxe Pack Mega (B4b). This fork audited the rules and every
   trainer card, and fixed what did not match the official text (details in `lab/README.md`, in
   French).
2. **The bot**: a neural network that learns to play the AlphaZero way, by playing against
   itself, with a search that handles hidden information and chance.
3. **The tools**: the forge (improve decks), deck search, simulated ranked, the web interface,
   the meta updates. They all use the bot to play their games.

The goal: objectively good decks, measured over thousands of games played by a bot that pilots
better than a human, rather than gut feelings.

## 2. Install

**Requirements**

- Windows 10 or 11.
- [Git](https://git-scm.com/).
- [Python 3.10 or later](https://www.python.org/): tick "py launcher" during the installation.
- For training: an NVIDIA GPU (RTX 20xx to 50xx) with a recent driver. The deck tools also run
  without a GPU (on the CPU, slower).

**Installation**

```
git clone https://github.com/Undi95/pocket-zero
cd pocket-zero
setup.bat
```

`setup.bat` (which runs `setup.ps1`) does everything **inside the folder**, nothing system-wide:

1. finds Python 3.10+;
2. creates the `.venv` Python environment;
3. installs PyTorch with CUDA 12.8 (works from RTX 20xx to RTX 50xx) and numpy;
4. downloads the engine binaries (`bin\pocket-zero.exe`, `bin\deckgym.exe`) and the engine
   Python module, built by GitHub Actions (`windows-bin` branch): no Rust or Visual Studio needed;
5. downloads the published pretrained network (`weights\pocket-zero.bin`, from the repository's
   `weights` branch; its `VERSION` file says which one it is), if there is one;
6. checks that everything loads and lists the GPUs it found.

**Other systems (Linux, macOS)**: no setup script, but everything builds:
`cargo build --release --bin pocket-zero --bin deckgym` (stable Rust), then in a Python
environment: `pip install torch numpy maturin` and `maturin develop --release --features python`.
The commands become `python -m zero.loop`, `python -m zero.forge`, etc. (see `run.bat` for the
mapping).

## 3. Update

```
run.bat update
```

Runs `git pull` (code, documentation, **new meta lists**), then `setup.ps1` again: up-to-date
engine binaries and the new published network if there is one (downloaded only when it changed).

- A Python-only change only needs `git pull`.
- An engine change (`src\` folder) is compiled by GitHub Actions ("Windows build" workflow,
  ~8-10 minutes): wait for the workflow to finish before `run.bat update`.
- If `run.bat` itself stops working: `git pull`, then `run.bat update`.

## 4. The network: use the published one or train your own

Every tool needs a trained network. Without a `--weights` option, they use:

1. the last run you trained (`runs\<run>\weights\latest.bin`, the most recent one);
2. otherwise, the published network downloaded by the setup (`weights\pocket-zero.bin`).
   Currently: iteration 103 of 150 of the big network, a preview until the final network.

To force a network: `--weights weights\pocket-zero.bin` (or any `.bin` file).

### Training

```
run.bat train --run main --iterations 150
run.bat train --run big1 --net big --iterations 150
```

- `--run`: the run name (folder `runs\<name>\`, `principal` by default). Running the same
  command again **resumes where the run stopped**: Ctrl+C at any time, at worst you lose the
  current iteration.
- `--iterations`: the total you aim for (not the number to add).
- `--net big`: the big network (~3.5× the parameters of `small`, better at piloting technical
  decks). The size is set when the run starts.
- `--focus 0.5` (default): half of the games involve at least one real deck (meta, other
  sources, forged decks): the network becomes an expert of the decks that matter. `--focus 0`:
  uniform pool.
- `--add-decks FOLDER`: adds decks to the pool (for example the forge's decks, ×10).
- `--no-ui`: does not open the live page in the browser.
- Advanced: `--games 10000` (self-play games per iteration), `--sims 200` (simulations per
  move), `--selfplay-devices cuda:0,cuda:1`, `--concurrency`, `--lr`, `--window`. See
  `run.bat help`.

**What one iteration does**

1. **Self-play**: 10,000 games of the network against itself, on every GPU (one process per
   card; the search runs in Rust on every core, the network evaluates thousands of positions per
   batch on the GPU). Decks are drawn from the training pool: meta lists, other sources, lines
   and pairs of every Pokémon, solo mode decks, "discovery" decks that cover every card.
2. **Training** on the recent positions (`cuda:0`), keeping the pass that generalizes best on
   held-out games.
3. **Evaluation**: 400 games against the previous network, against `e2` and against `e3` (two
   hand-written bots).
4. Every 5 iterations: **validation** against 10 real tournament matchups (Limitless) between 5
   meta decks: the mean gap (RMSE) tells whether the bot pilots the decks like real players.

On the reference machine (2× RTX 5090, big network), one iteration takes ~12 minutes:
150 iterations ≈ 30 hours.

**Reading the results** (`runs\<run>\log.txt`, `metrics.csv`, or better: the live page)

- "vs previous" stays around 50 %: each iteration is a small step, that is normal.
- **`vs_e3`**: the progress. Read it as a 5-iteration mean (the log prints it): 400 games
  measure within ±5 %. The score has a ceiling (luck exists: even a perfect player does not win
  100 % against a weak one), so +1 point at 65 % is worth much more than +1 point at 50 %.
- **RMSE**: the gap to real matchups, lower is better. `e3` is at ~11-12 points.
- **Held-out losses**: as long as the policy loss goes down, the network is still learning.
- The highlights after each self-play (deck of the iteration, gems outside the meta, streaks,
  upsets) are for fun and for spotting decks.

**The bot learns against itself, not against `e3`**: `e3` is only a thermometer. The network can
go far beyond it.

**Files of a run**: `log.txt`, `metrics.csv`, `weights\latest.bin` (the current bot),
`weights\iter_NNN.bin` (every iteration, numbered from 0), `checkpoint.pt` (resume),
`victoires.json` (per-deck results for the live page), `decks\` (the pool).

**Compare with an older version** (the best judge of progress):

```
bin\pocket-zero.exe arena --a z32:runs\big1\weights\latest.bin --b z32:runs\big1\weights\iter_039.bin --decks lab\decks\meta --games 400
```

(`iter_039` = iteration 40.) Above ~55 %, the network has really improved.

## 5. Follow the training live

The page opens by itself when `run.bat train` starts (otherwise `run.bat ui --train`):
`http://localhost:8765/train`. It refreshes every 20 seconds.

- **Top**: iteration, current step (self-play, training, validation), mean iteration time,
  estimated end, score against `e3`. Pick the run in the list.
- **Charts**: strength (vs `e3` raw and as a 5-iteration mean, vs `e2`), RMSE with the `e3`
  reference line, held-out losses.
- **Latest validation**: the 10 real matchups, gap in color (green < 5, orange < 10, red).
- **Decks that win the most**: self-play results per deck (copies of the same deck merged), over
  the last iteration, the last 10 or the whole run; all decks, homemade decks or real decks.
  Ranked by a corrected rate (5 wins and 5 losses added, so that a 3 out of 3 does not beat a 60
  out of 100); click a header to sort differently, click a deck to see its cards. It is a
  **scouting tool** (against the whole pool, weak decks included): confirm an interesting deck
  with the forge.
- **Highlights** and **log**: the last lines of the training log (in French).

From another PC or a phone on the same network: the address is printed at startup
(`http://192.168.x.x:8765/train`). 🌐 button at the top right: French or English.

## 6. The forge: improve decks, with proof

The recommended tool to get good decks.

```
run.bat forge --name forge1                     # the 10 most played meta lists
run.bat forge --name forge1 --deck my_deck.txt  # your deck (repeatable option)
run.bat forge --name forge1 --lock "B1a 42"     # keep some cards no matter what
```

**How it works**: the forge starts from a list and tries changes one by one: replace a trainer,
add or remove a copy, change an evolution line, change the energy. The most likely changes are
tried first (weighted by what real lists play). Each variant plays against the **weighted real
meta** (DotGG shares), on **the same games** as the current list (same seeds: same hands, same
coin flips): luck cancels out, only the effect of the change remains.

- A variant is kept only if it wins **clearly** (2 standard errors), then it is **confirmed** on
  new games.
- Weak variants are dropped early (progressive elimination): the forge spends its time on the
  serious candidates.
- At the end, a stronger **final measurement** (`z128`) compares the starting list and the forged
  list; if the gain is negative, the changes are cancelled.
- A gain measured between two versions of the same deck is nearly insensitive to the bot's
  biases: that is why the forge is more reliable than comparing two different decks.

**Results**: `results\forge\<name>\FORGE.md` (score, gain ✓ proven or ≈, every change,
matchups) and `results\forge\<name>\decks\` (the lists, visible in `run.bat ui`). Running again
with the same name resumes where it stopped.

**Options**: `--top 10` (number of meta lists), `--climb z32` (bot during the improvement),
`--final z128` (bot of the final measurement), `--steps 12` (maximum changes per deck),
`--cands 30` (variants per step), `--engine auto|gpu|cpu` (games run on the GPUs by default),
`--reeval` (only redoes the final measurement, useful after more training), `--weights`.

## 7. Homemade decks

```
run.bat forge --name homemade1 --new 12
```

Homemade mode draws 12 archetypes from **the whole card pool** (an attacker, its line, coherent
trainers), then forges them as above. The result is a deck nobody published, improved card by
card against the real meta.

**The full loop** so the bot pilots these new decks well:

1. `run.bat forge --name homemade1 --new 12`
2. `run.bat train --iterations 170 --add-decks results\forge\homemade1\decks`: the bot also
   trains on these decks (×10 in the pool) and learns to pilot them;
3. `run.bat forge --name homemade1 --reeval`: a new, less biased measurement;
4. `run.bat ranked --deck results\forge\homemade1\decks\<deck>.txt`, then for real in the game.

## 8. Deck search (genetic algorithm)

```
run.bat search --name search1
run.bat search --name around_pikachu --lock "A1 96"
```

A wider search than the forge, over the whole card pool: screening of every attacker line (`z0`,
very fast), best pairs, then a genetic algorithm with any card (`z8`), and a final validation of
the best ones (`z32`). Score: half against the real meta, half against a field of archetypes
found by the search (`--field-share 0.2` to change the field's part). Validity rules: 20 cards,
2 copies, an evolution needs its pre-evolution (or Rare Candy and the Basic Pokémon for a Stage
2), at least 4 Pokémon (`--min-pokemon`).

Result: `results\<name>\RESULTATS_D.md`. Options: `--gens 25` (generations; more = more
thorough), `--search z32` (stronger, slower), `--lock` (required cards, comma separated). The
forge is usually more reliable; the search is for exploring.

## 9. Simulated ranked

```
run.bat ranked
run.bat ranked --deck results\forge\forge1\decks\my_deck.txt
```

1. **Population**: the meta archetypes with their real share (DotGG), played with their known
   lists (Limitless, Game8…), plus your `--deck` files.
2. **Matchup matrix** played by the bot (`z64`, 20 games per pair, on the GPUs; cached in
   `results\ranked\matrix.json`: running again is instant).
3. **Season**: 5,000 players, 120 games each, against opponents of their rank. Official rules:
   +10 per win; loss 0 (Beginner), −5 (Poké Ball, Great Ball), −7 (Ultra Ball), −10 (Master
   Ball); win-streak bonus +3/+6/+9/+12 up to Great Ball; no demotion below the start of the
   current tier.

Result `results\ranked\CLASSEMENT.md`: wins per deck and per rank, share of players reaching
Master Ball, games needed. Options: `--player z32`, `--games 40`, `--players`, `--matches`,
`--weights`.

## 10. Solo mode decks

`lab\decks\solo\` holds the 466 decks of the solo mode battles (Poké Ball to Master Ball, from
Serebii). They are part of the training pool, and:

```
run.bat solo
```

makes them play each other: a ranking per rank, then a tournament of the best against the meta
(`results\solo\CLASSEMENT.md`).

## 11. The interface: decklists and official images

```
run.bat ui                  # decklists
run.bat ui --train          # straight to the training page
run.bat ui --host 127.0.0.1 # visible from this PC only
```

`http://localhost:8765`: every deck of the project (meta, other sources, solo mode, forge and
search results) or a pasted list, shown as cards with the game's checks (20 cards, 2 copies, a
Basic Pokémon).

- French images (Poképédia) or English images (TCGdex, Limitless), loaded by the browser.
- **Click a card**: zoom (arrows for the next one, Esc to close).
- Editing the text updates the list. "Copy text" for the engine, "Print / PDF".
- 🌐 button: interface in French or English (remembered; the browser language by default).
- Reachable from the local network by default (the address is printed at startup). On the first
  start, Windows asks for network access for Python: tick "Private networks".

## 12. The meta and its automatic updates

**The lists**

| Folder | Content | Updated |
|---|---|---|
| `lab\decks\meta\` | 17 reference Limitless lists (validation, evaluation, training pool) | by hand: fixed so the measurements stay comparable |
| `lab\decks\sources\limitless\` | the list of the best player of each archetype ≥ 1 % in the tournaments of the last 14 days (≥ 32 players), with share and win rate | automatically |
| `lab\decks\sources\game8\` | the lists of the Game8 tier list (tiers S and A) | automatically |
| `lab\decks\sources\dotgg\shares.json` | the meta share of each archetype (ptcgpocket.gg) | automatically |
| `lab\decks\sources\meta_share.json` | the shares spread over the lists: the ranked population and the forge's meta | automatically |

Every file keeps at its top (`#` lines, ignored by the engine) the site, the source URL, the
dates, the tier or the share.

**Automatic update**: the "Meta update" GitHub workflow (`.github/workflows/meta-update.yml`)
runs on Mondays and Thursdays (06:17 UTC). It runs `lab/tools/meta_sources.py` and commits
whatever is new to the repository. On your PC, `run.bat update` brings the new lists. The
workflow summary (GitHub Actions tab) says what changed.

- A file is rewritten only when its cards change (no commit for a mere date).
- A deck that leaves a source's list is removed.
- **New expansion**: a list with a card the engine does not know yet is skipped and reported
  ("cartes inconnues du moteur", unknown cards) until the engine is updated (`database.json` and
  the card implementations, see the original deckgym-core repository and
  `.claude/skills/implement-cards`).
- A source whose page changes layout prints "échec" (failed): its previous files stay.

**Manual update**, right now: `run.bat meta` (or `run.bat meta limitless`).

**Effect on the tools**: the ranked simulation and the forge read `meta_share.json` every time
they start; training adds the source lists to the pool (×10) **when a run starts**.

**Adding a source**: a function in `lab/tools/meta_sources.py` that writes its decks with
`write_deck(...)`, declared in `SOURCES`.

## 13. Deck format

```
# source: https://...      (comment line, ignored)
Energy: Lightning
2 Oricorio A3 66
2 Dedenne ex B3b 24
```

One card per line: count, name (English), set, number. 20 cards, at most 2 copies per name, at
least one Basic Pokémon. `Energy:`: one to three types.

## 14. The bots

Usable wherever a tool takes a player (`--player`, `--a`, `--b`, `--climb`…):

| Code | Bot |
|---|---|
| `z0` | network alone (its policy), very fast |
| `z8`, `z32`, `z64`, `z128` | network + search (8 to 128 simulations per move): more simulations = stronger, slower |
| `e2`, `e3` | hand-written expectiminimax, depth 2 or 3 (fixed yardsticks) |
| `r` | random |

With the binary directly: `z32:path\to\network.bin`. Diagnosing a matchup (who pilots which deck
badly?):

```
bin\pocket-zero.exe match --deck-a A.txt --deck-b B.txt --player z32:weights\pocket-zero.bin --player-b e3 --games 200
```

The `z` player also works in the deckgym simulator:

```
set POCKET_ZERO_WEIGHTS=weights\pocket-zero.bin
bin\deckgym.exe simulate deck1.txt deck2.txt --num 100 --players z64,e3
```

## 15. What the bot sees

Everything a player sees on screen, nothing more: the cards in play and their HP, energies,
tools, cards under each Pokémon, statuses, effects (with their amounts), actual retreat cost,
energy zone (current and next), stadium, its hand, the discard piles, the composition of its
remaining deck (not its order), the size of the opponent's hand and deck, the points, the KOs
and the attacks of the last turns. Never the opponent's hand or the order of the decks.

## How it works

**The engine** (`src/`): the deckgym simulator: game state, legal move generation, card effects.
Random effects are described as distributions of outcomes (`Outcomes`), which lets the bots
reason about chance.

**The encoding** (`src/zero/encode.rs`): what a player sees, turned into tensors: a dense part
(HP, energies, statuses, effects, counters) and card tokens (a learned representation per card).
Every legal move is encoded too: the network scores moves one by one.

**The network** (`src/zero/net.rs`, `zero/model.py`): a shared trunk, a value head (who is going
to win?) and a policy head (which move to play?). The PyTorch twin is used for training, the
Rust version for playing; `python -m zero.parity` checks that they give the same results.

**The search** (`src/zero/mcts.rs`, `batch.rs`): before each move, the bot simulates what comes
next. Hidden information is handled by **determinization** (it imagines opponent hands and deck
orders consistent with what it knows), chance in open loop. At the root, the **Gumbel** search
compares the promising moves by successive halving, every candidate played on **the same** hidden
cards and the same coin flips (paired comparisons: luck cancels out instead of drowning the
difference between two moves). The batched version advances thousands of games at once and
sends their positions to the GPU together.

**Learning** (`zero/loop.py`, `zero/train.py`): the policy learns the search-improved policy
(sharper than visit counts); the value learns a TD(λ) target (each position leans on the
evaluations of the following positions rather than only on the final result, very noisy in a
coin-flip game), calibrated at every iteration on held-out games.

**The tools** (`zero/`): `forge.py` (paired games, progressive elimination), `search.py` and
`lab/search/pipeline.py` (genetic search), `ranked.py`, `solo.py`, `ui.py`, `gpu_selfplay.py` and
`gpu_eval.py` (batched games on the GPUs).

## 17. For maintainers and forks

**Publishing a network** (so everyone gets it with `run.bat update`):

```
run.bat publish runs\big1\weights\latest.bin "big1, iteration 150/150"
```

The network goes to the repository's `weights` branch, alone, with a `VERSION` line (your note,
the date, a checksum). The branch is overwritten at every publication: only the latest network
stays there, the repository's history does not grow. The setup takes your repository's network,
otherwise the original repository's. You need write access to the repository (Git asks you to
sign in to GitHub the first time).

**Windows binaries**: the "Windows build" workflow compiles on every engine change on `main` and
publishes to the `windows-bin` branch, which `setup.ps1` downloads. On a fork, enable GitHub
Actions (*Actions* tab) so it runs.

**Automatic meta**: the "Meta update" workflow needs Actions enabled and write permission
(*Settings* → *Actions* → *General* → *Workflow permissions* → *Read and write*). GitHub pauses
scheduled workflows of a public repository with no activity for 60 days: *Enable workflow*
restarts them. Manual run: *Actions* → *Meta update* → *Run workflow*.

**Tests**: `cargo test --release --features "tui test-utils"` (engine), `cargo clippy --features
"tui test-utils" -- -D warnings`, `python -m zero.parity` (PyTorch twin).

## 18. Troubleshooting

| Problem | Solution |
|---|---|
| `'o' is not recognized...` when running `run.bat` | broken line endings: `git pull` (the repository forces Windows line endings for `.bat` files) |
| "Réseau introuvable / network not found" | `run.bat update` (published network) or `run.bat train`, or `--weights file.bin` |
| "windows-bin branch missing" | the "Windows build" workflow has not run yet: wait for it (or enable Actions on your fork) |
| `unexpected shape` when loading a network | binaries older than the code: wait for the "Windows build" workflow, then `run.bat update` |
| The page does not open from another PC | Windows network set to "Public", or access refused on the first start: set the network to "Private", or in an administrator terminal `netsh advfirewall firewall add rule name="pocket-zero ui" dir=in action=allow protocol=TCP localport=8765` |
| Disk full during training | old games are deleted automatically; `--keep-shards` keeps them |
| "Le format des entrées a changé" (input format changed) | an incompatible network change: new `--run` |
| A meta list "ignoré : cartes inconnues du moteur" (unknown cards) | new expansion not yet in the engine: the list comes back once the engine is updated |
| NumPy "not writable" warning | harmless |

## 19. Credits and license

- [deckgym-core](https://github.com/bcollazo/deckgym-core) by **bcollazo** and its contributors:
  the simulator everything is built on.
- Meta: [Limitless](https://play.limitlesstcg.com/), [Game8](https://game8.co/games/Pokemon-TCG-Pocket),
  [ptcgpocket.gg / DotGG](https://ptcgpocket.gg/); solo mode and ranked rules:
  [Serebii](https://www.serebii.net/).
- Images: [Poképédia](https://www.pokepedia.fr/), [TCGdex](https://tcgdex.dev/),
  [Limitless](https://pocket.limitlesstcg.com/) (loaded when displayed, not stored).
- License [AGPL-3.0](../LICENSE.txt). Fan project, not affiliated with Nintendo, Creatures, GAME
  FREAK, The Pokémon Company or DeNA.
