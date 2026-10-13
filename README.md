# pocket-zero

**An AlphaZero bot and deck-building tools for Pokémon TCG Pocket.**
*[Version française : README.fr.md](README.fr.md)*

pocket-zero teaches a neural network to play Pokémon TCG Pocket by playing millions of games
against itself, then uses that network to find, test and improve decks: meta lists, your own
lists, or 100 % homemade decks built from every card in the game.

It is built on **[deckgym-core](https://github.com/bcollazo/deckgym-core)** by
[bcollazo](https://github.com/bcollazo), a fast Pokémon TCG Pocket simulator written in Rust
(AGPL-3.0). pocket-zero is a fork of it: the engine, the card database and most of the card
implementations come from deckgym; this project adds the rule and card audit, the AlphaZero bot,
the GPU training, the deck tools and the web interface (see [Credits](#credits)).

## What it does

| | |
|---|---|
| **Plays like a player** | The bot only sees what a player sees on screen: its hand, the board, tools, statuses, effects, discard piles, the *size* of the opponent's hand and deck, never their content or order. It searches with hidden information (determinization) and chance (coin flips). |
| **Learns by itself** | AlphaZero-style training: self-play on every GPU, a network trained on the results, evaluated against hand-written bots and against real tournament matchups from Limitless. |
| **Improves decks with proof** | The *forge* tries one change at a time (a trainer, a copy, an evolution line, an energy) against the real meta, on the *same* games as the current list, and keeps it only if it wins clearly. Homemade mode builds new archetypes from the whole card pool. |
| **Tests in real conditions** | A simulated ranked season with the official ladder rules (Beginner to Master Ball) and the real meta shares. |
| **Stays up to date** | The meta decklists (Limitless tournaments, Game8, DotGG shares) refresh automatically twice a week. |
| **Shows everything** | A web interface in French and English: decklists with the official card images (click to zoom), and a live page that follows the training and the decks that win the most. |

## Quick start (Windows)

Requirements: Windows 10/11, [Git](https://git-scm.com/), [Python 3.10+](https://www.python.org/)
(tick "py launcher"), an NVIDIA GPU with a recent driver for training (RTX 20xx to 50xx; the
deck tools also run on the CPU, slower).

```
git clone https://github.com/Undi95/pocket-zero
cd pocket-zero
setup.bat
```

`setup.bat` installs everything **inside the folder** (Python environment, PyTorch CUDA, engine
binaries built by GitHub Actions, the published network if there is one). Nothing is installed
system-wide, no Rust or Visual Studio needed.

Then:

```
run.bat ui                                  decklists + live training page (http://localhost:8765)
run.bat forge --name test --deck my.txt     improve your deck against the meta
run.bat forge --name homemade --new 12      12 homemade archetypes from the whole card pool
run.bat ranked --deck my.txt                simulated ranked season
run.bat train --run mine --net big          train your own network
run.bat update                              latest code, binaries, network and meta lists
run.bat help                                every command and option
```

## Pretrained network

`setup.bat` and `run.bat update` download the published network into `weights\pocket-zero.bin`,
so the deck tools work right away, without training.

**Currently published: iteration 103 of 150 of the big network (preview, not the final
network)**, about 68 % against the hand-written `e3` bot. The final network (iteration 150) will
replace it; `run.bat update` brings it. The `VERSION` file in `weights\` says which one you have.

## Documentation

- **[Full guide (English)](docs/GUIDE.en.md)**: every tool explained in detail: installation,
  training and how to read it, the forge, homemade decks, deck search, ranked simulation, the
  web interface, the automatic meta updates, troubleshooting, publishing a network.
- **[Guide complet (français)](docs/GUIDE.fr.md)**
- [`lab/README.md`](lab/README.md): engine audit notes (rules, cards, accuracy), in French.
- [`lab/decks/sources/README.md`](lab/decks/sources/README.md): where the meta lists come from.

## How it works, in short

- `src/` — the deckgym engine (rules, 4,300+ cards, move generation), audited and fixed here.
- `src/zero/` — what a player sees encoded as tensors (`encode.rs`), the network (`net.rs`),
  the search with hidden information and paired comparisons (`mcts.rs`, `batch.rs`), self-play
  and the `pocket-zero` command line.
- `zero/` — Python: the PyTorch twin of the network, the AlphaZero loop, the GPU self-play and
  evaluation, the forge, deck search, ranked simulation and web interface.
- `lab/` — the meta decklists, the solo mode decks, the audit and meta tools.

The [guide](docs/GUIDE.en.md#how-it-works) goes into the details.

## Credits

- **[deckgym-core](https://github.com/bcollazo/deckgym-core)** by **bcollazo** and its
  contributors: the Rust simulator this project is built on (engine, card database, card
  implementations, Python bindings). License AGPL-3.0.
- Meta data: [Limitless](https://play.limitlesstcg.com/) (tournament results and decklists),
  [Game8](https://game8.co/games/Pokemon-TCG-Pocket) (tier list),
  [ptcgpocket.gg / DotGG](https://ptcgpocket.gg/) (meta shares),
  [Serebii](https://www.serebii.net/) (solo mode decks and ranked rules). Every list keeps its
  source URL and date at the top of its file.
- Card images, loaded by your browser when you view a deck (none are stored in this repository):
  [Poképédia](https://www.pokepedia.fr/) (French), [TCGdex](https://tcgdex.dev/) and
  [Limitless](https://pocket.limitlesstcg.com/) (English).
- pocket-zero by [Undi95](https://github.com/Undi95).

## License

[AGPL-3.0](LICENSE.txt), like deckgym-core. You can use, modify and share this project; if you
distribute a modified version or run it as a network service, you must publish its source under
the same license.

*Pokémon, Pokémon TCG Pocket and the card names and images are trademarks of Nintendo,
Creatures, GAME FREAK and The Pokémon Company. This is a fan project, not affiliated with or
endorsed by them or by DeNA.*
