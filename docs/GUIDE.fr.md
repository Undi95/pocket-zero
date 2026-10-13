# Guide de pocket-zero

*[English version: GUIDE.en.md](GUIDE.en.md)*

Ce guide explique chaque outil en détail : à quoi il sert, comment le lancer, comment lire ses
résultats. Toutes les commandes se lancent depuis le dossier du projet, dans une invite de
commandes (`cmd`) ou PowerShell (`.\run.bat ...`). `run.bat help` liste toutes les commandes et
leurs options.

## Sommaire

1. [Ce qu'est pocket-zero](#1-ce-quest-pocket-zero)
2. [Installer](#2-installer)
3. [Mettre à jour](#3-mettre-à-jour)
4. [Le réseau : utiliser le réseau publié ou entraîner le tien](#4-le-réseau--utiliser-le-réseau-publié-ou-entraîner-le-tien)
5. [Suivre l'entraînement en direct](#5-suivre-lentraînement-en-direct)
6. [La forge : améliorer des decks, preuve à l'appui](#6-la-forge--améliorer-des-decks-preuve-à-lappui)
7. [Les decks maison](#7-les-decks-maison)
8. [La recherche de decks (algorithme génétique)](#8-la-recherche-de-decks-algorithme-génétique)
9. [Le ranked simulé](#9-le-ranked-simulé)
10. [Les decks du mode solo](#10-les-decks-du-mode-solo)
11. [L'interface : decklists et images officielles](#11-linterface--decklists-et-images-officielles)
12. [La méta et ses mises à jour automatiques](#12-la-méta-et-ses-mises-à-jour-automatiques)
13. [Format des decks](#13-format-des-decks)
14. [Les bots](#14-les-bots)
15. [Ce que le bot voit](#15-ce-que-le-bot-voit)
16. [Comment ça marche](#comment-ça-marche)
17. [Pour les mainteneurs et les forks](#17-pour-les-mainteneurs-et-les-forks)
18. [Dépannage](#18-dépannage)
19. [Crédits et licence](#19-crédits-et-licence)

## 1. Ce qu'est pocket-zero

pocket-zero a trois étages :

1. **Le moteur** : un simulateur de Pokémon TCG Pocket écrit en Rust, issu de
   [deckgym-core](https://github.com/bcollazo/deckgym-core) (de bcollazo, AGPL-3.0). Il connaît les
   règles et toutes les cartes jusqu'au Deluxe Pack Mega (B4b). Ce fork a audité les règles et
   chaque dresseur, corrigé ce qui ne collait pas au texte officiel (détails dans
   `lab/README.md`).
2. **Le bot** : un réseau de neurones qui apprend à jouer façon AlphaZero, en jouant contre
   lui-même, avec une recherche qui gère l'information cachée et le hasard.
3. **Les outils** : la forge (améliorer des decks), la recherche de decks, le ranked simulé,
   l'interface web, la mise à jour de la méta. Ils se servent tous du bot pour jouer les parties.

Le but : des decks objectivement bons, mesurés sur des milliers de parties jouées par un bot qui
pilote mieux qu'un humain, plutôt que des intuitions.

## 2. Installer

**Prérequis**

- Windows 10 ou 11.
- [Git](https://git-scm.com/).
- [Python 3.10 ou plus](https://www.python.org/) : coche « py launcher » à l'installation.
- Pour l'entraînement : une carte NVIDIA (RTX 20xx à 50xx) avec un pilote récent. Les outils de
  decks marchent aussi sans GPU (sur le processeur, plus lentement).

**Installation**

```
git clone https://github.com/Undi95/pocket-zero
cd pocket-zero
setup.bat
```

`setup.bat` (qui lance `setup.ps1`) fait tout **dans le dossier**, rien dans le système :

1. trouve Python 3.10+ ;
2. crée l'environnement Python `.venv` ;
3. installe PyTorch avec CUDA 12.8 (marche des RTX 20xx aux RTX 50xx) et numpy ;
4. télécharge les binaires du moteur (`bin\pocket-zero.exe`, `bin\deckgym.exe`) et le module
   Python du moteur, compilés par GitHub Actions (branche `windows-bin`) : ni Rust ni Visual
   Studio ne sont nécessaires ;
5. télécharge le réseau pré-entraîné publié (`weights\pocket-zero.bin`), s'il y en a un ;
6. vérifie que tout se charge et affiche les GPU trouvés.

**Autres systèmes (Linux, macOS)** : pas de script d'installation, mais tout se compile :
`cargo build --release --bin pocket-zero --bin deckgym` (Rust stable), puis dans un environnement
Python : `pip install torch numpy maturin` et `maturin develop --release --features python`.
Les commandes deviennent `python -m zero.loop`, `python -m zero.forge`, etc. (voir `run.bat`
pour la correspondance).

## 3. Mettre à jour

```
run.bat update
```

Fait `git pull` (code, documentation, **nouvelles listes méta**), puis relance `setup.ps1` :
binaires du moteur à jour et nouveau réseau publié s'il y en a un (il n'est téléchargé que s'il
a changé).

- Un changement purement Python n'a besoin que de `git pull`.
- Un changement du moteur (dossier `src\`) est compilé par GitHub Actions (workflow « Windows
  build », ~8-10 minutes) : attends la fin du workflow avant `run.bat update`.
- Si `run.bat` lui-même ne marche plus : `git pull` puis `run.bat update`.

## 4. Le réseau : utiliser le réseau publié ou entraîner le tien

Tous les outils ont besoin d'un réseau entraîné. Sans option `--weights`, ils prennent :

1. le dernier run que tu as entraîné (`runs\<run>\weights\latest.bin`, le plus récent) ;
2. sinon, le réseau publié téléchargé par l'installation (`weights\pocket-zero.bin`).

Pour forcer un réseau : `--weights weights\pocket-zero.bin` (ou n'importe quel fichier `.bin`).

### Entraîner

```
run.bat train --run principal --iterations 150
run.bat train --run grand --net big --iterations 150
```

- `--run` : le nom du run (dossier `runs\<nom>\`, `principal` par défaut). Relancer la même
  commande **reprend là où le run s'est arrêté** : Ctrl+C à tout moment, au pire tu perds
  l'itération en cours.
- `--iterations` : le total visé (pas le nombre à ajouter).
- `--net big` : le gros réseau (~3,5× plus de paramètres que `small`, meilleur pilotage des decks
  techniques). La taille est fixée au premier lancement du run.
- `--focus 0.5` (défaut) : la moitié des parties implique au moins un vrai deck (méta, autres
  sources, decks forgés) : le réseau devient expert des decks qui comptent. `--focus 0` : pool
  uniforme.
- `--add-decks DOSSIER` : ajoute des decks au pool (par exemple les decks de la forge, ×10).
- `--no-ui` : n'ouvre pas la page de suivi dans le navigateur.
- Avancé : `--games 10000` (parties d'auto-jeu par itération), `--sims 200` (simulations par
  coup), `--selfplay-devices cuda:0,cuda:1`, `--concurrency`, `--lr`, `--window`. Voir
  `run.bat help`.

**Ce que fait une itération**

1. **Auto-jeu** : 10 000 parties du réseau contre lui-même, sur tous les GPU (un processus par
   carte ; la recherche tourne en Rust sur tous les cœurs, le réseau évalue des milliers de
   positions par lot sur le GPU). Decks tirés du pool d'entraînement : listes méta, autres
   sources, lignées et paires de chaque Pokémon, decks du mode solo, decks « découverte » qui
   couvrent chaque carte.
2. **Entraînement** sur les positions récentes (`cuda:0`), en gardant la passe qui généralise le
   mieux sur des parties de contrôle jamais vues.
3. **Évaluation** : 400 parties contre le réseau précédent, contre `e2` et contre `e3` (deux bots
   écrits à la main).
4. Toutes les 5 itérations : **validation** contre 10 vrais matchups de tournoi (Limitless) entre
   5 decks de la méta : l'écart moyen (RMSE) dit si le bot pilote les decks comme les vrais
   joueurs.

Sur la machine de référence (2× RTX 5090, gros réseau), une itération prend ~12 minutes :
150 itérations ≈ 30 heures.

**Lire les résultats** (`runs\<run>\log.txt`, `metrics.csv`, ou mieux : la page en direct)

- « contre le précédent » reste vers 50 % : chaque itération est un petit pas, c'est normal.
- **`vs_e3`** : la progression. Lis-la en moyenne sur 5 itérations (le log l'affiche) : 400
  parties mesurent à ±5 %. Le score plafonne (la chance existe : même un joueur parfait ne gagne
  pas 100 % contre un faible), donc +1 point à 65 % vaut bien plus que +1 point à 50 %.
- **RMSE** : l'écart aux vrais matchups, plus bas = mieux. `e3` est à ~11-12 points.
- **Pertes sur les parties de contrôle** : tant que la politique baisse, le réseau apprend encore.
- Les moments forts après chaque auto-jeu (deck de l'itération, pépites hors méta, séries,
  exploits) sont pour le plaisir et pour repérer des decks.

**Le bot apprend contre lui-même, pas contre `e3`** : `e3` n'est qu'un thermomètre. Le réseau peut
largement le dépasser.

**Fichiers d'un run** : `log.txt`, `metrics.csv`, `weights\latest.bin` (le bot à jour),
`weights\iter_NNN.bin` (chaque itération, numérotées à partir de 0), `checkpoint.pt` (reprise),
`victoires.json` (résultats des decks pour la page en direct), `decks\` (le pool).

**Comparer à une ancienne version** (le meilleur juge du progrès) :

```
bin\pocket-zero.exe arena --a z32:runs\grand\weights\latest.bin --b z32:runs\grand\weights\iter_039.bin --decks lab\decks\meta --games 400
```

(`iter_039` = itération 40.) Au-dessus de ~55 %, le réseau a vraiment progressé.

## 5. Suivre l'entraînement en direct

La page s'ouvre toute seule au lancement de `run.bat train` (sinon `run.bat ui --train`) :
`http://localhost:8765/train`. Elle se rafraîchit toutes les 20 secondes.

- **En haut** : itération, étape en cours (auto-jeu, entraînement, validation), durée moyenne
  d'une itération, heure de fin estimée, score contre `e3`. Choix du run dans la liste.
- **Courbes** : force (contre `e3` brut et en moyenne sur 5, contre `e2`), RMSE avec la ligne de
  référence d'`e3`, pertes sur les parties de contrôle.
- **Dernière validation** : les 10 vrais matchups, l'écart coloré (vert < 5, orange < 10, rouge).
- **Decks qui gagnent le plus** : résultats de l'auto-jeu par deck (copies d'un même deck
  regroupées), sur la dernière itération, les 10 dernières ou tout le run ; tous les decks, les
  decks maison ou les vrais decks. Classement par taux corrigé (5 victoires et 5 défaites
  fictives ajoutées, pour qu'un 3 sur 3 ne passe pas devant un 60 sur 100) ; clic sur un en-tête
  pour trier autrement, clic sur un deck pour voir ses cartes. C'est un **repérage** (contre tout
  le pool, decks faibles compris) : confirme un deck intéressant avec la forge.
- **Moments forts** et **journal** : les dernières lignes du log (en français).

Depuis un autre PC ou un téléphone du même réseau : l'adresse s'affiche au lancement
(`http://192.168.x.x:8765/train`). Bouton 🌐 en haut à droite : français ou anglais.

## 6. La forge : améliorer des decks, preuve à l'appui

L'outil recommandé pour obtenir de bons decks.

```
run.bat forge --name forge1                      # les 10 listes méta les plus jouées
run.bat forge --name forge1 --deck mon_deck.txt  # ton deck (option répétable)
run.bat forge --name forge1 --lock "B1a 42"      # garde certaines cartes quoi qu'il arrive
```

**Principe** : la forge part d'une liste et essaie des changements un par un : remplacer un
dresseur, ajouter ou retirer une copie, changer une lignée d'évolution, changer d'énergie. Les
changements les plus probables sont essayés en premier (pondérés par ce que jouent les vraies
listes). Chaque variante est jouée contre la **méta réelle pondérée** (parts DotGG), sur **les
mêmes parties** que la liste actuelle (mêmes graines : mêmes mains, mêmes pile ou face) : la
chance s'annule, il ne reste que l'effet du changement.

- Une variante n'est gardée que si elle gagne **nettement** (2 écarts-types), puis elle est
  **confirmée** sur de nouvelles parties.
- Les variantes faibles sont éliminées tôt (élimination progressive) : la forge passe son temps
  sur les candidats sérieux.
- À la fin, une **mesure finale** plus forte (`z128`) compare la liste de départ et la liste
  forgée ; si le gain est négatif, les changements sont annulés.
- Un gain mesuré entre deux versions du même deck est quasi insensible aux biais du bot : c'est
  pour ça que la forge est plus fiable que de comparer deux decks différents.

**Résultats** : `results\forge\<nom>\FORGE.md` (score, gain ✓ prouvé ou ≈, chaque changement,
matchups) et `results\forge\<nom>\decks\` (les listes, visibles dans `run.bat ui`). Relancer
avec le même nom reprend où elle s'était arrêtée.

**Options** : `--top 10` (nombre de listes méta), `--climb z32` (bot pendant l'amélioration),
`--final z128` (bot de la mesure finale), `--steps 12` (changements maximum par deck),
`--cands 30` (variantes par étape), `--engine auto|gpu|cpu` (les parties se jouent sur les GPU
par défaut), `--reeval` (refait seulement la mesure finale, utile après un nouvel entraînement),
`--weights`.

## 7. Les decks maison

```
run.bat forge --name maison1 --new 12
```

Le mode maison tire 12 archétypes de **tout le pool de cartes** (un attaquant, sa lignée, des
dresseurs cohérents), puis les forge comme ci-dessus. Le résultat est un deck que personne n'a
publié, amélioré carte par carte contre la vraie méta.

**La boucle complète** pour que le bot pilote bien ces nouveaux decks :

1. `run.bat forge --name maison1 --new 12`
2. `run.bat train --iterations 170 --add-decks results\forge\maison1\decks` : le bot s'entraîne
   en plus sur ces decks (×10 dans le pool) et apprend à les piloter ;
3. `run.bat forge --name maison1 --reeval` : nouvelle mesure, moins biaisée ;
4. `run.bat ranked --deck results\forge\maison1\decks\<deck>.txt`, puis en vrai dans le jeu.

## 8. La recherche de decks (algorithme génétique)

```
run.bat search --name recherche1
run.bat search --name autour_de_pikachu --lock "A1 96"
```

Une recherche plus large que la forge, sur tout le pool de cartes : criblage de chaque lignée
d'attaquant (`z0`, très rapide), meilleures paires, puis algorithme génétique avec n'importe
quelle carte (`z8`), et validation finale des meilleurs (`z32`). Score : moitié contre la méta
réelle, moitié contre un terrain d'archétypes trouvés par la recherche (`--field-share 0.2` pour
changer la part du terrain). Règles de validité : 20 cartes, 2 exemplaires, une évolution a besoin
de sa pré-évolution (ou de Super Bonbon et du Pokémon de base pour un Stade 2), au moins 4
Pokémon (`--min-pokemon`).

Résultat : `results\<nom>\RESULTATS_D.md`. Options : `--gens 25` (générations ; plus = plus
fouillé), `--search z32` (plus fort, plus lent), `--lock` (cartes imposées, séparées par des
virgules). La forge est en général plus fiable ; la recherche sert à explorer.

## 9. Le ranked simulé

```
run.bat ranked
run.bat ranked --deck results\forge\forge1\decks\mon_deck.txt
```

1. **Population** : les archétypes de la méta avec leur vraie part (DotGG), joués avec leurs
   listes connues (Limitless, Game8…), plus tes decks `--deck`.
2. **Matrice des matchups** jouée par le bot (`z64`, 20 parties par paire, sur les GPU ; mise en
   cache dans `results\ranked\matrix.json` : relancer est instantané).
3. **Saison** : 5 000 joueurs, 120 parties chacun, contre des adversaires de leur rang. Règles
   officielles : +10 par victoire ; défaite 0 (Débutant), −5 (Poké Ball, Super Ball), −7 (Hyper
   Ball), −10 (Master Ball) ; bonus de série +3/+6/+9/+12 jusqu'à Super Ball ; pas de descente
   sous le début du rang.

Résultat `results\ranked\CLASSEMENT.md` : victoires par deck et par rang, part des joueurs qui
atteignent Master Ball, parties nécessaires. Options : `--player z32`, `--games 40`, `--players`,
`--matches`, `--weights`.

## 10. Les decks du mode solo

`lab\decks\solo\` contient les 466 decks des combats du mode solo (de Poké Ball à Master Ball,
d'après Serebii). Ils font partie du pool d'entraînement, et :

```
run.bat solo
```

les fait s'affronter : un classement par rang, puis un tournoi des meilleurs contre la méta
(`results\solo\CLASSEMENT.md`).

## 11. L'interface : decklists et images officielles

```
run.bat ui                  # decklists
run.bat ui --train          # directement la page de l'entraînement
run.bat ui --host 127.0.0.1 # visible de ce PC seulement
```

`http://localhost:8765` : tous les decks du projet (méta, autres sources, mode solo, résultats
de la forge et de la recherche) ou une liste collée, affichés en cartes avec les vérifications du
jeu (20 cartes, 2 exemplaires, un Pokémon de base).

- Images en français (Poképédia) ou en anglais (TCGdex, Limitless), chargées par le navigateur.
- **Clic sur une carte** : zoom (flèches pour passer à la suivante, Échap pour fermer).
- Modifier le texte met la liste à jour. « Copier le texte » pour le moteur, « Imprimer / PDF ».
- Bouton 🌐 : interface en français ou en anglais (mémorisé ; par défaut, la langue du
  navigateur).
- Accessible depuis le réseau local par défaut (l'adresse s'affiche au lancement). Au premier
  lancement, Windows demande l'accès réseau pour Python : coche « Réseaux privés ».

## 12. La méta et ses mises à jour automatiques

**Les listes**

| Dossier | Contenu | Mise à jour |
|---|---|---|
| `lab\decks\meta\` | 17 listes Limitless de référence (validation, évaluation, pool d'entraînement) | à la main : fixes pour garder les mesures comparables |
| `lab\decks\sources\limitless\` | la liste du meilleur joueur de chaque archétype ≥ 1 % sur les tournois des 14 derniers jours (≥ 32 joueurs), avec part et taux de victoire | automatique |
| `lab\decks\sources\game8\` | les listes de la tier list Game8 (tiers S et A) | automatique |
| `lab\decks\sources\dotgg\shares.json` | les parts de méta de chaque archétype (ptcgpocket.gg) | automatique |
| `lab\decks\sources\meta_share.json` | les parts réparties sur les listes : la population du ranked et la méta de la forge | automatique |

Chaque fichier garde en tête (lignes `#`, ignorées par le moteur) le site, l'URL de la source,
les dates, le tier ou la part.

**Mise à jour automatique** : le workflow GitHub « Meta update » (`.github/workflows/meta-update.yml`)
tourne le lundi et le jeudi (06:17 UTC). Il relance `lab/tools/meta_sources.py`, et s'il y a du
nouveau, le commite dans le dépôt. Chez toi, `run.bat update` récupère les nouvelles listes. Le
résumé du workflow (onglet Actions de GitHub) dit ce qui a changé.

- Un fichier n'est réécrit que si ses cartes changent (pas de commit pour une simple date).
- Un deck qui sort de la liste d'une source est supprimé.
- **Nouvelle extension** : une liste qui contient une carte que le moteur ne connaît pas encore
  est ignorée et signalée (« cartes inconnues du moteur ») jusqu'à ce que le moteur soit mis à jour
  (`database.json` et implémentation des cartes, voir le dépôt deckgym-core d'origine et
  `.claude/skills/implement-cards`).
- Une source dont la page change de forme affiche « échec » : ses anciens fichiers restent.

**Mise à jour manuelle**, tout de suite : `run.bat meta` (ou `run.bat meta limitless`).

**Effet sur les outils** : le ranked et la forge lisent `meta_share.json` à chaque lancement ;
l'entraînement ajoute les listes des sources au pool (×10) **au démarrage d'un run**.

**Ajouter une source** : une fonction dans `lab/tools/meta_sources.py` qui écrit ses decks avec
`write_deck(...)`, déclarée dans `SOURCES`.

## 13. Format des decks

```
# source: https://...      (ligne de commentaire, ignorée)
Energy: Lightning
2 Oricorio A3 66
2 Dedenne ex B3b 24
```

Une carte par ligne : nombre, nom (en anglais), extension, numéro. 20 cartes, 2 exemplaires
maximum par nom, au moins un Pokémon de base. `Energy:` : un à trois types.

## 14. Les bots

Utilisables partout où un outil prend un joueur (`--player`, `--a`, `--b`, `--climb`…) :

| Code | Bot |
|---|---|
| `z0` | réseau seul (sa politique), très rapide |
| `z8`, `z32`, `z64`, `z128` | réseau + recherche (8 à 128 simulations par coup) : plus de simulations = plus fort, plus lent |
| `e2`, `e3` | expectiminimax écrit à la main, profondeur 2 ou 3 (repères fixes) |
| `r` | aléatoire |

Avec le binaire directement : `z32:chemin\vers\reseau.bin`. Diagnostic d'un matchup (qui pilote
mal quel deck ?) :

```
bin\pocket-zero.exe match --deck-a A.txt --deck-b B.txt --player z32:weights\pocket-zero.bin --player-b e3 --games 200
```

Le joueur `z` marche aussi dans le simulateur deckgym :

```
set POCKET_ZERO_WEIGHTS=weights\pocket-zero.bin
bin\deckgym.exe simulate deck1.txt deck2.txt --num 100 --players z64,e3
```

## 15. Ce que le bot voit

Tout ce qu'un joueur voit à l'écran, rien de plus : les cartes en jeu et leurs PV, énergies,
outils, cartes sous chaque Pokémon, statuts, effets (avec leurs montants), coût de retraite réel,
zone d'énergie (actuelle et suivante), stade, sa main, les défausses, la composition de son deck
restant (pas son ordre), la taille de la main et du deck adverses, les points, les KO et les
attaques des derniers tours. Jamais la main adverse ni l'ordre des decks.

## Comment ça marche

**Le moteur** (`src/`) : le simulateur deckgym : état de la partie, génération des coups légaux,
effets des cartes. Les effets aléatoires sont décrits comme des distributions de résultats
(`Outcomes`), ce qui permet aux bots de raisonner sur le hasard.

**L'encodage** (`src/zero/encode.rs`) : ce qu'un joueur voit, transformé en tenseurs : une partie
dense (PV, énergies, statuts, effets, compteurs) et des jetons de cartes (une représentation
apprise par carte). Chaque coup légal est encodé aussi : le réseau note les coups un par un.

**Le réseau** (`src/zero/net.rs`, `zero/model.py`) : un tronc commun, une tête de valeur (qui va
gagner ?) et une tête de politique (quel coup jouer ?). Le jumeau PyTorch sert à l'entraînement,
la version Rust à jouer ; `python -m zero.parity` vérifie qu'ils donnent les mêmes résultats.

**La recherche** (`src/zero/mcts.rs`, `batch.rs`) : avant chaque coup, le bot simule la suite.
L'information cachée est gérée par **déterminisation** (il imagine des mains adverses et des
ordres de deck compatibles avec ce qu'il sait), le hasard en boucle ouverte. À la racine, la
recherche **Gumbel** compare les coups prometteurs par élimination successive, chaque candidat
joué sur **les mêmes** cartes cachées et les mêmes pile ou face (comparaisons appariées : la
chance s'annule au lieu de noyer la différence entre deux coups). La version par lots fait
avancer des milliers de parties en même temps et envoie leurs positions au GPU d'un coup.

**L'apprentissage** (`zero/loop.py`, `zero/train.py`) : la politique apprend la politique
améliorée par la recherche (plus nette que les nombres de visites) ; la valeur apprend une cible
TD(λ) (chaque position s'appuie sur les évaluations des positions suivantes plutôt que sur le
seul résultat final, très bruité dans un jeu de pile ou face), calibrée à chaque itération sur
des parties jamais vues.

**Les outils** (`zero/`) : `forge.py` (parties appariées, élimination progressive), `search.py`
et `lab/search/pipeline.py` (recherche génétique), `ranked.py`, `solo.py`, `ui.py`,
`gpu_selfplay.py` et `gpu_eval.py` (parties par lots sur les GPU).

## 17. Pour les mainteneurs et les forks

**Publier un réseau** (pour que tout le monde l'ait avec `run.bat update`) :

1. sur GitHub : *Releases* → *Draft a new release* ;
2. un tag, par exemple `net-2026-10-15`, et un titre ;
3. joins le fichier du réseau renommé **`pocket-zero.bin`** (par exemple
   `runs\grand\weights\latest.bin`) ;
4. *Publish release*. L'installation prend le réseau de la dernière release de ton dépôt, sinon
   celui du dépôt d'origine.

**Binaires Windows** : le workflow « Windows build » compile à chaque changement du moteur sur
`main` et publie sur la branche `windows-bin`, que `setup.ps1` télécharge. Sur un fork, active
GitHub Actions (onglet *Actions*) pour qu'il tourne.

**Méta automatique** : le workflow « Meta update » a besoin des Actions activées et des droits
d'écriture (*Settings* → *Actions* → *General* → *Workflow permissions* → *Read and write*).
GitHub met en pause les workflows planifiés d'un dépôt public sans activité depuis 60 jours : un
clic sur *Enable workflow* les relance. Lancement à la main : *Actions* → *Meta update* →
*Run workflow*.

**Tests** : `cargo test --release --features "tui test-utils"` (moteur), `cargo clippy --features
"tui test-utils" -- -D warnings`, `python -m zero.parity` (jumeau PyTorch).

## 18. Dépannage

| Problème | Solution |
|---|---|
| `'o' n'est pas reconnu...` en lançant `run.bat` | fins de ligne cassées : `git pull` (le dépôt force le format Windows pour les `.bat`) |
| « Réseau introuvable / network not found » | `run.bat update` (réseau publié) ou `run.bat train`, ou `--weights fichier.bin` |
| « Branche windows-bin absente » | le workflow « Windows build » n'a pas encore tourné : attends-le (ou active les Actions sur ton fork) |
| `unexpected shape` au chargement d'un réseau | binaires plus anciens que le code : attends la fin du workflow « Windows build », puis `run.bat update` |
| La page ne s'ouvre pas depuis un autre PC | réseau Windows en « Public », ou accès refusé au premier lancement : passer le réseau en « Privé », ou dans un terminal administrateur `netsh advfirewall firewall add rule name="pocket-zero ui" dir=in action=allow protocol=TCP localport=8765` |
| Disque plein pendant l'entraînement | les anciennes parties sont supprimées automatiquement ; `--keep-shards` les garde |
| « Le format des entrées a changé » | un changement incompatible du réseau : nouveau `--run` |
| Une liste méta « ignorée : cartes inconnues du moteur » | nouvelle extension pas encore dans le moteur : la liste reviendra après la mise à jour du moteur |
| Avertissement NumPy « not writable » | sans effet |

## 19. Crédits et licence

- [deckgym-core](https://github.com/bcollazo/deckgym-core) de **bcollazo** et ses contributeurs :
  le simulateur sur lequel tout repose.
- Méta : [Limitless](https://play.limitlesstcg.com/), [Game8](https://game8.co/games/Pokemon-TCG-Pocket),
  [ptcgpocket.gg / DotGG](https://ptcgpocket.gg/) ; mode solo et règles du ranked :
  [Serebii](https://www.serebii.net/).
- Images : [Poképédia](https://www.pokepedia.fr/), [TCGdex](https://tcgdex.dev/),
  [Limitless](https://pocket.limitlesstcg.com/) (chargées à l'affichage, pas stockées).
- Licence [AGPL-3.0](../LICENSE.txt). Projet de fan, sans lien avec Nintendo, Creatures, GAME
  FREAK, The Pokémon Company ou DeNA.
