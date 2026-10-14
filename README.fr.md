# pocket-zero

**Un bot AlphaZero et des outils de construction de decks pour Pokémon TCG Pocket.**
*[English version: README.md](README.md)*

pocket-zero apprend à un réseau de neurones à jouer à Pokémon TCG Pocket en jouant des millions
de parties contre lui-même, puis se sert de ce réseau pour trouver, tester et améliorer des
decks : les listes méta, tes propres listes, ou des decks 100 % maison construits avec toutes les
cartes du jeu.

Il est construit sur **[deckgym-core](https://github.com/bcollazo/deckgym-core)** de
[bcollazo](https://github.com/bcollazo), un simulateur rapide de Pokémon TCG Pocket écrit en Rust
(AGPL-3.0). pocket-zero en est un fork : le moteur, la base de cartes et la plupart des
implémentations de cartes viennent de deckgym ; ce projet ajoute l'audit des règles et des
cartes, le bot AlphaZero, l'entraînement sur GPU, les outils de decks et l'interface web (voir
[Crédits](#crédits)).

## Ce qu'il fait

| | |
|---|---|
| **Joue comme un joueur** | Le bot ne voit que ce qu'un joueur voit à l'écran : sa main, le terrain, les outils, les statuts, les effets, les défausses, la *taille* de la main et du deck adverses, jamais leur contenu ni leur ordre. Il cherche avec l'information cachée (déterminisation) et le hasard (pile ou face). |
| **Apprend tout seul** | Entraînement façon AlphaZero : auto-jeu sur tous les GPU, un réseau entraîné sur les résultats, évalué contre des bots écrits à la main et contre les vrais matchups de tournoi de Limitless. |
| **Améliore les decks, preuve à l'appui** | La *forge* essaie un changement à la fois (un dresseur, une copie, une lignée, une énergie) contre la vraie méta, sur les *mêmes* parties que la liste actuelle, et ne le garde que s'il gagne nettement. Le mode maison construit de nouveaux archétypes avec tout le pool de cartes. |
| **Teste en conditions réelles** | Une saison de ranked simulée avec les règles officielles du classement (Débutant à Master Ball) et les vraies parts de méta. |
| **Reste à jour** | Les decklists méta (tournois Limitless, Game8, parts DotGG) se mettent à jour automatiquement deux fois par semaine. |
| **Montre tout** | Une interface web en français et en anglais : decklists avec les images officielles des cartes (clic pour zoomer), une page en direct qui suit l'entraînement et les decks qui gagnent le plus, et le suivi en direct de la forge, de la recherche de decks et du ranked. |

## Démarrage rapide (Windows)

Prérequis : Windows 10/11, [Git](https://git-scm.com/), [Python 3.10+](https://www.python.org/)
(coche « py launcher »), une carte NVIDIA avec un pilote récent pour l'entraînement (RTX 20xx à
50xx ; les outils de decks tournent aussi sur le processeur, plus lentement).

```
git clone https://github.com/Undi95/pocket-zero
cd pocket-zero
setup.bat
```

`setup.bat` installe tout **dans le dossier** (environnement Python, PyTorch CUDA, binaires du
moteur compilés par GitHub Actions, le réseau publié s'il y en a un). Rien n'est installé dans le
système, ni Rust ni Visual Studio ne sont nécessaires.

Ensuite :

```
run.bat ui                                  decklists + page de l'entraînement (http://localhost:8765)
run.bat forge --name test --deck mon.txt    améliore ton deck contre la méta
run.bat forge --name maison --new 12        12 archétypes maison tirés de tout le pool de cartes
run.bat ranked --deck mon.txt               saison de ranked simulée
run.bat train --run moi --net big           entraîne ton propre réseau
run.bat update                              dernière version du code, des binaires, du réseau et de la méta
run.bat help                                toutes les commandes et options
```

## Réseau pré-entraîné

`setup.bat` et `run.bat update` téléchargent le réseau publié dans `weights\pocket-zero.bin` : les
outils de decks marchent tout de suite, sans entraînement.

**Publié en ce moment : itération 160 du gros réseau (final)**, environ 72 % contre le bot `e3`
écrit à la main et 80 % contre `e2` ; il reproduit les vrais matchups à 7,3 points près (le
hasard seul : 4,7 ; `e3` : 13,9). `run.bat update` le récupère. Le fichier `VERSION` dans `weights\` dit lequel tu as.

## Documentation

- **[Guide complet (français)](docs/GUIDE.fr.md)** : chaque outil expliqué en détail :
  installation, entraînement et comment le lire, la forge, les decks maison, la recherche de
  decks, le ranked simulé, l'interface web, les mises à jour automatiques de la méta, dépannage,
  publier un réseau.
- **[Full guide (English)](docs/GUIDE.en.md)**
- [`lab/README.md`](lab/README.md) : notes d'audit du moteur (règles, cartes, fidélité).
- [`lab/decks/sources/README.md`](lab/decks/sources/README.md) : d'où viennent les listes méta.

## Comment ça marche, en bref

- `src/` : le moteur deckgym (règles, plus de 4 300 cartes, génération des coups), audité et
  corrigé ici.
- `src/zero/` : ce qu'un joueur voit, encodé en tenseurs (`encode.rs`), le réseau (`net.rs`), la
  recherche avec information cachée et comparaisons appariées (`mcts.rs`, `batch.rs`), l'auto-jeu
  et la ligne de commande `pocket-zero`.
- `zero/` : Python : le jumeau PyTorch du réseau, la boucle AlphaZero, l'auto-jeu et l'évaluation
  sur GPU, la forge, la recherche de decks, le ranked simulé et l'interface web.
- `lab/` : les decklists méta, les decks du mode solo, les outils d'audit et de méta.

Le [guide](docs/GUIDE.fr.md#comment-ça-marche) entre dans les détails.

## Crédits

- **[deckgym-core](https://github.com/bcollazo/deckgym-core)** de **bcollazo** et ses
  contributeurs : le simulateur Rust sur lequel ce projet est construit (moteur, base de cartes,
  implémentations des cartes, module Python). Licence AGPL-3.0.
- Données de méta : [Limitless](https://play.limitlesstcg.com/) (résultats et decklists de
  tournois), [Game8](https://game8.co/games/Pokemon-TCG-Pocket) (tier list),
  [ptcgpocket.gg / DotGG](https://ptcgpocket.gg/) (parts de méta),
  [Serebii](https://www.serebii.net/) (decks du mode solo et règles du ranked). Chaque liste garde
  l'URL de sa source et sa date en tête de fichier.
- Images des cartes, chargées par ton navigateur quand tu affiches un deck (aucune n'est stockée
  dans ce dépôt) : [Poképédia](https://www.pokepedia.fr/) (français), [TCGdex](https://tcgdex.dev/)
  et [Limitless](https://pocket.limitlesstcg.com/) (anglais).
- pocket-zero par [Undi95](https://github.com/Undi95).

## Licence

[AGPL-3.0](LICENSE.txt), comme deckgym-core. Tu peux utiliser, modifier et partager ce projet ;
si tu distribues une version modifiée ou la fais tourner comme service en ligne, tu dois publier
son code source sous la même licence.

*Pokémon, Pokémon TCG Pocket, les noms et images des cartes sont des marques de Nintendo,
Creatures, GAME FREAK et The Pokémon Company. Ceci est un projet de fan, sans lien avec eux ni
avec DeNA, et sans leur approbation.*
