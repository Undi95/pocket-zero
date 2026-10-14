# Using the pretrained network

This page explains how to get the published pocket-zero network and use it in every tool:
no training needed. The [full guide](GUIDE.en.md) explains each tool in more detail.

> **Current network: iteration 150 of the big network.** It beats the hand-written `e3` bot about
> 73 % of the time and `e2` about 81 %. Ten more iterations with the latest engine fixes will
> replace it (iteration 160); `run.bat update` brings it.

## 1. Get it

### With the setup (recommended)

Requirements: Windows 10/11, [Git](https://git-scm.com/), [Python 3.10+](https://www.python.org/)
(tick "py launcher"). An NVIDIA GPU makes the tools much faster, but they also run on the CPU.

```
git clone https://github.com/Undi95/pocket-zero
cd pocket-zero
setup.bat
```

The setup installs everything inside the folder and downloads the network to
**`weights\pocket-zero.bin`**. The file `weights\VERSION` says which network you have.

### Get the newest network later

```
run.bat update
```

It updates the code, the engine binaries, the meta decklists and the network. Run it from time to
time, and once the final network is out.

### Download it by hand

The network lives alone on the repository's `weights` branch:

- direct link: `https://github.com/Undi95/pocket-zero/raw/weights/pocket-zero.bin`
- its description: `https://github.com/Undi95/pocket-zero/blob/weights/VERSION`

Put it anywhere and pass it with `--weights` (see below), or save it as `weights\pocket-zero.bin`.

## 2. Which network the tools use

Without `--weights`, every tool uses, in this order:

1. the most recent network you trained yourself (`runs\<run>\weights\latest.bin`), if any;
2. otherwise the published network, `weights\pocket-zero.bin`.

To choose explicitly, add `--weights` to any command:

```
run.bat forge --name test --deck my_deck.txt --weights weights\pocket-zero.bin
```

## 3. Use it in each tool

### Improve your deck: the forge

```
run.bat forge --name mydeck --deck my_deck.txt
```

The bot tries one change at a time (a trainer, a copy, an evolution line, the energy) against the
real meta, on the same games as your current list, and keeps a change only when it clearly wins.
Result: `results\forge\mydeck\FORGE.md` (every change, the proven gain, the matchups) and the
improved list in `results\forge\mydeck\decks\`.

- `--deck` can be repeated to forge several decks.
- `--lock "B1a 42"` keeps cards you want no matter what.
- No `--deck`: the 10 most played meta lists are forged.

### Build new decks: homemade mode

```
run.bat forge --name homemade --new 12
```

Draws 12 archetypes from the whole card pool, then forges each one against the real meta.

### Test a deck on the ladder: simulated ranked

```
run.bat ranked --deck my_deck.txt
```

A full ranked season (Beginner to Master Ball, official point rules, real meta shares) with your
deck among 5,000 players. Result: `results\ranked\CLASSEMENT.md` (win rate per rank, share of
players reaching Master Ball, games needed).

### Explore: deck search

```
run.bat search --name search1
run.bat search --name around_pikachu --lock "A1 96"
```

A genetic search over every card; `--lock` builds around the cards you give. Result:
`results\search1\RESULTATS_D.md`.

### Solo mode decks

```
run.bat solo
```

The 466 solo mode decks against each other, then the best ones against the meta.

### See decks with the official card images

```
run.bat ui
```

Opens `http://localhost:8765`: every deck of the project and the results of the tools, shown as
cards (French or English images, click to zoom), French or English interface. This one does not
need the network.

### One matchup, in detail

Who wins with both decks played by the network, then with deck B played by the hand-written
`e3` bot (if the result moves a lot, one of the two pilots misplays that deck):

```
bin\pocket-zero.exe match --deck-a A.txt --deck-b B.txt --player z32:weights\pocket-zero.bin --games 200
bin\pocket-zero.exe match --deck-a A.txt --deck-b B.txt --player z32:weights\pocket-zero.bin --player-b e3 --games 200
```

### The network against a deck folder

```
bin\pocket-zero.exe arena --a z32:weights\pocket-zero.bin --b e3 --decks lab\decks\meta --games 400
```

### In the deckgym simulator

```
set POCKET_ZERO_WEIGHTS=weights\pocket-zero.bin
bin\deckgym.exe simulate deck1.txt deck2.txt --num 100 --players z64,e3
```

## 4. Strength and speed: the `z` players

Every tool takes a player such as `--player z64` (ranked), `--climb z32` / `--final z128`
(forge), `--search z8` (deck search).

| Player | What it is | Use |
|---|---|---|
| `z0` | the network alone (its first instinct) | very fast screening |
| `z8` | network + 8 simulations per move | already stronger than `e3` |
| `z32` | network + 32 simulations | good default for measurements |
| `z64`, `z128` | network + 64 / 128 simulations | strongest, slower |
| `e2`, `e3` | hand-written bots (depth 2 / 3) | fixed yardsticks |

With the binaries directly, add the network file: `z32:weights\pocket-zero.bin`.

## 5. Deck files

```
# comment lines start with #
Energy: Lightning
2 Oricorio A3 66
2 Dedenne ex B3b 24
```

One card per line: count, English name, set, number. 20 cards, at most 2 copies per name, at
least one Basic Pokémon. `run.bat ui` checks a list and shows it as cards.

## 6. If something goes wrong

| Message | Fix |
|---|---|
| "Réseau introuvable / network not found" | `run.bat update`, or pass `--weights path\to\pocket-zero.bin` |
| `unexpected shape` when loading the network | the engine binaries are older than the network: `run.bat update` |
| Very slow tools | no NVIDIA GPU found: everything runs on the CPU; use fewer games (`--games`) or a lighter player (`z8`) |
| Other problems | [troubleshooting](GUIDE.en.md#18-troubleshooting) |
