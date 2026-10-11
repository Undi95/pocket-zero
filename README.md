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

