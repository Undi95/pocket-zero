# pocket-zero

Bot AlphaZero et recherche de decks pour Pokémon TCG Pocket, construits sur le moteur
[deckgym-core](https://github.com/bcollazo/deckgym-core) (AGPL-3.0), avec les règles et les cartes
auditées dans Pocket Deck Lab (voir `lab/README.md`).

## Installation (Windows)

1. Clone le dépôt, puis double-clique sur **`setup.bat`**.
   Tout s'installe *dans le dossier* : environnement Python `.venv` (PyTorch CUDA 12.8 pour les
   RTX 50xx), binaires du moteur `bin\` (compilés par GitHub Actions, workflow « Windows build »),
   module Python du moteur. Rien n'est installé dans le système.
2. Prérequis : Git, Python 3.10+ (avec le lanceur `py`), un pilote NVIDIA récent.

## Entraîner le bot

```
run.bat train
```

Chaque itération : self-play (recherche en Rust sur tous les cœurs, réseau évalué par lots sur
**chaque GPU**, un processus par carte), entraînement sur `cuda:0`, puis évaluation contre
l'ancien réseau et contre les bots `e2` et `e3`. Toutes les 5 itérations, validation contre les
vrais matchups Limitless (écart moyen, RMSE ; le bot `e3` est à ~10-12 points).
Tout se trouve dans `runs\principal\` : `log.txt`, `metrics.csv`, `weights\latest.bin`.
L'entraînement reprend où il s'est arrêté : relance simplement la même commande.

Après chaque autojeu, un petit résumé des moments forts : deck de l'itération, classement de la
méta contre tout le pool, pépites hors méta, plus longue série, victoire éclair, exploit,
marathon ; après l'évaluation, les records contre `e3`.

Lire les résultats : « contre le précédent » reste vers 50 % (chaque itération n'est qu'un petit
pas, et 400 parties mesurent à ±5 %). La progression se lit sur `vs_e3` dans la durée : le log
affiche la moyenne des 5 dernières itérations à côté des 5 précédentes.

Options utiles : `--iterations 60`, `--games 3000`, `--sims 200`, `--concurrency 2048`,
`--selfplay-devices cuda:0,cuda:1`, `--run autre_nom` (nouveau run depuis zéro).
`run.bat help` les liste toutes.

## Chercher des decks (toutes les cartes)

```
run.bat search --name recherche1
```

Recherche sur **tout le pool de cartes**, jouée par le réseau entraîné : criblage de chaque
lignée d'attaquant, paires, algorithme génétique avec n'importe quelle carte (lignées
incomplètes, Rare Candy, 1 à 3 énergies), score = 50 % méta réel + 50 % terrain d'archétypes
hors méta. Résultat : `results\recherche1\RESULTATS_D.md`. Options : `--lock "B1a 42"` (construire
autour d'une carte), `--gens 40`. Joueurs par défaut : criblage `z0` (réseau seul), algorithme
génétique `z8` (8 simulations, déjà plus fort que `e3`), validation finale `z32` ; `--search z32`
pour un joueur encore plus fort mais plus lent. Lance-la entraînement arrêté : les deux se
partagent le processeur.

Toutes les cartes ont un deck dans le pool d'entraînement (`cover-*.txt`, ajoutés au démarrage
d'un run) : le bot a vu jouer chaque Pokémon et chaque dresseur que la recherche peut choisir.

## Les decks du mode solo

`lab/decks/solo/` contient les decks des développeurs des combats progressifs du mode solo
(466 decks, de `pokeball` à `masterball`, d'après les listes de Serebii), retirés du jeu fin
octobre 2026. Ils font partie du pool d'entraînement (`solo-*.txt`), et :

```
run.bat solo
```

les fait s'affronter avec le bot entraîné : un classement par rang, puis un grand tournoi des
meilleurs de chaque rang contre les 17 listes de la méta (`results\solo\CLASSEMENT.md`, une
dizaine de minutes, entraînement arrêté).

## Utiliser le bot ailleurs

Le joueur `z<simulations>` marche partout où le moteur prend un joueur (`z0` = politique du
réseau seule, rapide) :

```
set POCKET_ZERO_WEIGHTS=runs\principal\weights\latest.bin
bin\deckgym.exe simulate deck1.txt deck2.txt --num 100 --players z64,e3
```

## Comment ça marche

- `src/zero/encode.rs` : ce qu'un joueur voit (taille de la main adverse mais pas son contenu,
  composition de son propre deck mais pas son ordre), encodé en tenseurs.
- `src/zero/mcts.rs`, `src/zero/batch.rs` : recherche avec déterminisation (information cachée)
  et hasard en boucle ouverte ; version par lots pour le GPU. Racine « Gumbel » : halving
  séquentiel sur les coups les plus prometteurs, chaque candidat joué sur *les mêmes* cartes
  cachées et les mêmes pile ou face (comparaisons appariées : la chance s'annule au lieu de
  noyer la différence entre deux coups), et cible de politique = politique améliorée par les
  valeurs Q (bien plus nette que les nombres de visites, qui restaient presque plats).
  `zp<sims>` garde l'ancienne racine PUCT pour comparer.
- Cible de valeur TD(λ) (`--td-lambda 0.9`) : chaque position s'appuie sur les évaluations des
  positions suivantes de la même partie, au lieu du seul résultat final (très bruité dans un jeu
  de pile ou face). Valeur calibrée à chaque itération sur des parties jamais vues (le réseau
  avait tendance à être trop sûr de lui).
- `pocket-zero valuecheck` : compare, sur des parties jouées, l'évaluation du réseau, celle écrite
  à la main et le vrai résultat.
- `zero/` : modèle PyTorch (jumeau exact du réseau Rust, vérifié par `python -m zero.parity`),
  entraînement, boucle AlphaZero, pool de decks, recherche.

---

<img src="./images/logo.svg" alt="Logo" width="100" height="100">

# deckgym-core: Pokémon TCG Pocket Simulator

![Card Implemented](https://img.shields.io/badge/Cards_Implemented-3340_%2F_3761_%2888.8%25%29-green)

**deckgym-core** is a high-performance Rust library designed for simulating Pokémon TCG Pocket games. It features a command-line interface (CLI) capable of running 10,000 simulations in approximately 3 seconds. This is the library that powers https://www.deckgym.com.

Its mission is to elevate the competitive TCG Pocket scene by helping players optimize their decks through large-scale simulations.

Join our Discord: https://discord.gg/ymxXHrzhak!

## Usage

The CLI runs simulations between two decks in DeckGym Format. To create these files, build your decks in https://www.deckgym.com/builder, select **Share** > **Copy as Text**, and save the content as a text file.

We already provide several example decks in the repo you can use to get started. For example, to face off a VenusaurEx-ExeggutorEx deck with a Weezing-Arbok deck 1,000 times, run:

```bash
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --num 1000 -v
```

You can also simulate one deck against multiple decks in a folder. The total games will be distributed evenly across all decks:

```bash
# Simulate your deck against all decks in example_decks folder (1000 games total)
cargo run simulate my_deck.txt example_decks/ --num 1000 -v
```

## Terminal User Interface (TUI)

The TUI provides an interactive way to view and replay games with a visual representation of the game state.

To use the TUI, you need to enable the `tui` feature:

```bash
cargo run --bin tui --features tui -- example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --players e,e
```

### Controls

- **↑/↓/Space**: Navigate between game states (forward/backward)
- **PageUp/PageDown**: Scroll through battle log
- **A/D**: Scroll through your hand cards
- **Shift+A/Shift+D**: Scroll through opponent's hand cards
- **Q/Esc**: Quit

## Data Export

The simulator can export (state, action) pairs during gameplay for machine learning and data analysis applications. This feature outputs portable JSON files that are easy to consume from any programming language.

### Usage

Use the `--data-output` flag to specify an output folder:

```bash
cargo run -- simulate example_decks/fire.txt example_decks/flareon.txt \
  -n 1000 \
  --players r,r \
  --data-output ./exported_data
```

### Output Structure

The data is organized as:
```
exported_data/
├── <game_id_1>/
│   ├── ply_0000.json
│   ├── ply_0001.json
│   └── ...
├── <game_id_2>/
│   ├── ply_0000.json
│   └── ...
└── ...
```

Each JSON file contains:
- `game_id`: UUID of the game
- `ply`: Sequential decision point number
- `actor`: Which player (0 or 1) made the decision
- `state`: Complete game state (hands, decks, in-play Pokemon, etc.)
- `playable_actions`: All available actions at this decision point
- `chosen_action`: The action that was actually taken

### Processing the Data

A Python example is included to demonstrate data loading and feature extraction:

```bash
python examples/load_exported_data.py ./exported_data
```

The JSON format makes it easy to:
- Train policy networks (predict actions from states)
- Train value networks (estimate win probability)
- Perform imitation learning
- Build custom machine learning models
- Analyze gameplay patterns and statistics

## Replay Recording (Web Viewer)

The simulator can record one JSON replay file per game: a full board snapshot at every decision
point plus the legal options and the action that was chosen. These replays are meant to be opened
in the web viewer under `viewer/` (see `viewer/README.md`), which steps through the game with
animated transitions.

```bash
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt \
  -n 3 --players r,r --replay-dir replays/
```

Use `--replay-sample N` to only write replays for the first N games of a run (useful when running
many simulations but only wanting a few to inspect):

```bash
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt \
  -n 1000 --players r,r --replay-dir replays/ --replay-sample 10
```

### Play one game and open it in the viewer

`play` is a shortcut for the "run one game, then go look at it" loop: it plays exactly one game,
writes its replay, prints a winner/points summary, and prints a `viewer` URL as its last line.

```bash
cd viewer && npm run dev        # leave this running in one terminal
```
```bash
# in another terminal, from the repo root
cargo run -- play example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --players e2,r
```

Ctrl+Click (or Cmd+Click, depending on your terminal) the printed link to open that game in the
viewer — see [`viewer/README.md`](./viewer/README.md#playing-one-game-and-opening-it-straight-from-the-terminal)
for how that link resolves and its other flags (`--seed`, `--bot-a`/`--bot-b`, `--replay-dir`,
`--viewer-url`).

## External Bots

A bot written in any language can play as either seat, talking to the engine over stdin/stdout in
JSON lines. Give that seat the player code `x` and point `--bot-a`/`--bot-b` at the command to run:

```bash
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt \
  -n 1 --players r,x --bot-b "python3 examples/bots/random_bot.py"
```

`examples/bots/random_bot.py` is a minimal, stdlib-only example bot (picks a random legal action
and attaches a short note explaining its pick). See [`docs/bot-protocol.md`](./docs/bot-protocol.md)
for the full wire protocol, including the message schemas and how failures (timeouts, invalid
replies, a crashed process) are handled. `--bot-timeout-ms` (default 10000) controls how long the
engine waits for one decision. A bot's notes, when set, are stored in `--replay-dir` replays and
shown in the [web viewer](./viewer/README.md)'s options panel.

## Contributing

New to Open Source? See [CONTRIBUTING.md](./CONTRIBUTING.md).

The main contribution is to implement more cards, basically their attack and abilities logic. This makes the cards eligible for simulation and thus available for use in https://www.deckgym.com.

See the Claude [SKILL.md](./.claude/skills/implement-cards/SKILL.md) describing how to implement cards.
It's good documentation for humans and AIs alike.


## Appendix: Useful Commands

Once you have Rust installed (see https://www.rust-lang.org/tools/install) you should be able to use the following commands from the root of the repo:

**Running Automated Test Suite**

```bash
cargo test --features "tui test-utils"
```

**Running Benchmarks**

```bash
cargo bench --features test-utils
```

**Running Main Script**

```bash
# Simulate between two specific decks
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --num 1000 --players r,r
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --num 1 --players r,r -vv
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --num 1 --players r,r -vvvv

# Simulate one deck against all decks in a folder (games distributed evenly)
cargo run simulate example_decks/venusaur-exeggutor.txt example_decks/ --num 1000 --players r,r -v

# Optimize incomplete decks
cargo run optimize example_decks/incomplete-chari.txt A2147,A2148 example_decks/ --num 10 --players e,e -v
cargo run optimize example_decks/incomplete-chari.txt A2147,A2147,A2148,A2148 example_decks/ --num 1000 --players r,r -v --parallel
```

**Card Search Tool**

The repository includes a search utility that's particularly useful for agentic AI applications, as reading the complete `database.json` file (which contains all card data) often exceeds context limits.

```bash
# Search for cards by name
cargo run --bin search "Charizard"

# Search for cards with specific attacks
cargo run --bin search "Venusaur" --attack "Giant Bloom"
```

**Card Implementation Status Tool**

Check which cards are fully implemented versus which are missing attack effects, abilities, or trainer logic. This tool helps contributors identify cards that need implementation work.

```bash
# Show all cards with their implementation status
cargo run --bin card_status

# Show only incomplete cards
cargo run --bin card_status -- --incomplete-only

# Get the first incomplete card (useful for automation)
cargo run --bin card_status -- --first-incomplete
```

The tool displays a summary showing total cards, completion percentage, and a breakdown of missing implementations by type (attacks, abilities, tools, trainer logic).

**Temporary Deck Generator**

Generate a valid temporary test deck for a specific card id (it considers the evolution chain of a card and the required energy types if its a pokemon card).

```bash
cargo run --bin temp_deck_generator -- "A1 035"
```

**Card Test Command**

Generate a temporary test deck and run 10,000 games against all decks in `example_decks/` (games distributed evenly) using random players.

```bash
cargo run --bin card_test -- "A1 035"
```

**Setting Up Git Hooks (Optional)**

The repository includes a pre-commit hook that ensures code quality by automatically fixing issues and running tests before each commit. To enable it:

```bash
git config core.hooksPath .githooks
```

The pre-commit hook runs:
1. `cargo clippy --fix --allow-dirty --features tui -- -D warnings` - Auto-fixes linting issues
2. `cargo fmt` - Auto-formats code
3. `git add -u` - Adds clippy and formatting fixes to the commit
4. `cargo test --features "tui test-utils"` - Runs the full test suite (fails commit if tests fail)

This helps maintain code quality and prevents broken commits, but it's optional and each developer can choose whether to enable it.

**Generating database.rs**

Ensure database.json is up-to-date with latest data. Mock the `get_card_by_enum` in `database.rs` with a `_ => panic` so that
it compiles mid-way through the generation.

```bash
cargo run --bin card_enum_generator > tmp.rs && mv tmp.rs src/card_ids.rs && cargo fmt
```

Then temporarily edit `database.rs` for `_` to match Bulbasaur (this is so that the next code can compile-run).

```bash
cargo run --bin card_enum_generator -- --database > tmp.rs && mv tmp.rs src/database.rs && cargo fmt
```

To generate attacks do (first time):
```bash
cargo run --bin card_enum_generator -- --attack-map > tmp.rs && mv tmp.rs src/actions/effect_mechanic_map.rs && cargo fmt
```

then with each new set of new mechanics, use:
```bash
cargo run --bin card_enum_generator -- --incremental-attack-map
```
and manually copy-paste into the ever changing `src/actions/effect_mechanic_map.rs`.

For abilities incremental updates, use:
```bash
cargo run --bin card_enum_generator -- --incremental-ability-map
```
and manually copy-paste into `src/actions/effect_ability_mechanic_map.rs`.

**Profiling Main Script**
sudo cargo flamegraph --root --dev -- simulate example_decks/venusaur-exeggutor.txt example_decks/weezing-arbok.txt --num 1000 && open flamegraph.svg
```
