# Guide d'utilisation de pocket-zero

Toutes les commandes se lancent depuis le dossier du projet, dans une invite de commandes
(`cmd`) ou PowerShell (`.\run.bat ...`). `run.bat help` liste les options de chaque outil.

## 1. Installer et mettre à jour

| Action | Commande |
|---|---|
| Première installation | double-clic sur `setup.bat` (Python `.venv`, PyTorch CUDA, binaires du moteur dans `bin\`) |
| Mise à jour (code + binaires) | `run.bat update` |
| Mise à jour quand `run.bat` lui-même ne marche plus | `git pull` puis `run.bat update` |

Les changements purement Python n'ont besoin que de `git pull`. `run.bat update` récupère aussi
les binaires compilés par GitHub Actions (workflow « Windows build », ~8 min après un changement
du moteur).

## 2. Entraîner le bot

```
run.bat train --iterations 140
```

- Reprend là où le run s'est arrêté (`runs\principal\`). `--iterations` est le total visé.
- Chaque itération : auto-jeu sur les deux GPU (10 000 parties), entraînement, évaluation contre
  le réseau précédent, `e2` et `e3`, résumé des moments forts. Toutes les 5 itérations,
  validation contre les vrais matchups Limitless (RMSE, `e3` fait 11-12 points).
- Fichiers : `log.txt`, `metrics.csv`, `weights\latest.bin` (le bot), `checkpoint.pt`.
- Lire la progression : `vs_e3` en moyenne sur 5 itérations (le log l'affiche). « contre le
  précédent » reste vers 50 %, c'est normal.
- Changement d'entrées du réseau (couche 2 : outils, cartes sous chaque Pokémon, effets,
  historique public) : un run existant continue, le réseau garde sa force et apprend à
  utiliser les nouvelles entrées. Rien à faire.
- Nouveau run depuis zéro : `--run autre_nom`. Gros réseau (~3,5x plus de paramètres, meilleur
  pilotage des decks techniques) : `--run grand --net big` (la taille est fixée au début du run).
- Entraînement ciblé : `--focus 0.5` (défaut) fait jouer la moitié des parties avec de vrais decks
  (méta, autres sources, decks forgés ajoutés par `--add-decks`) ; `--focus 0` = pool uniforme.

## 3. Trouver des decks (le but)

```
run.bat search --name recherche1
run.bat search --name autour_de_pikachu --lock "A1 96"
```

Criblage de toutes les lignées, paires, puis algorithme génétique sur tout le pool de cartes,
joués par le réseau (`z0` criblage, `z8` génétique, `z32` validation finale). Score : moitié
méta réelle, moitié terrain d'archétypes. Résultat : `results\<nom>\RESULTATS_D.md`.
Options : `--gens 40` (plus long, plus fouillé), `--search z32` (plus fort, plus lent),
`--lock` (cartes imposées, séparées par des virgules). Lance-la entraînement arrêté.

## 3 bis. La forge : améliorer des decks, preuve à l'appui (recommandé)

```
run.bat forge --name forge1                      # les 10 listes méta les plus jouées
run.bat forge --name forge1 --deck mon_deck.txt  # ton deck
run.bat forge --name maison1 --new 12            # 12 archétypes maison, tirés de tout le pool
```

Les parties se jouent sur les GPU (moteur par lots, `--engine cpu` pour forcer le processeur) :
amélioration en z32, mesure finale en z128. Chaque changement (dresseur, copie, lignée, énergie) est joué contre la méta réelle pondérée
(DotGG), sur les MÊMES parties que le deck actuel : il n'est gardé que s'il gagne nettement
(2 écarts-types), puis confirmé sur de nouvelles parties. Un gain mesuré entre deux versions du
même deck est quasi insensible aux biais du bot. Résultat : `results\forge\<nom>\FORGE.md`
(score, gain prouvé ✓ ou ≈, changements, matchups) et `decks\` (visibles dans `run.bat ui`).
Reprend où elle s'est arrêtée si on la relance avec le même nom.

Boucle complète pour les decks maison :
1. `run.bat forge --name maison1 --new 12`
2. `run.bat train --iterations 150 --add-decks results\forge\maison1\decks` (le bot apprend à les piloter)
3. `run.bat forge --name maison1 --reeval` (nouvelle mesure, moins biaisée)
4. `run.bat ranked --deck results\forge\maison1\decks\<deck>.txt`, puis en vrai dans le jeu

## 4. Tester en conditions réelles : le ranked simulé

```
run.bat ranked
run.bat ranked --deck results\recherche1\meilleur.txt
```

1. Population : les archétypes de la méta avec leur vraie part (DotGG), joués avec leurs listes
   connues (Limitless, Game8), plus tes decks `--deck`.
2. Matrice des matchups jouée par le réseau (20 parties par paire, mise en cache dans
   `results\ranked\matrix.json` : relancer est instantané).
3. Saison : 5 000 joueurs, 120 parties chacun, contre des adversaires de leur rang. Règles
   officielles : +10 par victoire, défaite 0 (Débutant) / -5 (Poké Ball, Super Ball) / -7
   (Hyper Ball) / -10 (Master Ball), bonus de série +3/+6/+9/+12 jusqu'à Super Ball, pas de
   descente sous le début du rang.

Résultat `results\ranked\CLASSEMENT.md` : victoires par deck et par rang, part des joueurs qui
atteignent Master Ball, parties nécessaires. Options : `--player z32`, `--games 40`,
`--players`, `--matches`.

## 5. L'interface : decklists et entraînement en direct

```
run.bat ui                  # decklists
run.bat ui --train          # directement la page de l'entraînement
run.bat ui --host 0.0.0.0   # accessible depuis le téléphone (même réseau ou VPN)
```

`http://localhost:8765` : tous les decks du projet (méta, autres sources, mode solo, résultats
de recherche) ou une liste collée, affichés en cartes (images en français depuis Poképédia, ou
en anglais), avec les vérifications du jeu (20 cartes, 2 exemplaires, un Pokémon de base).
« Copier le texte » pour le moteur, « Imprimer / PDF » pour la garder.

`http://localhost:8765/train` : le run en cours, rafraîchi toutes les 20 s (choix du run en
haut). Itération, étape, fin estimée, courbes (contre e3/e2, RMSE, pertes), dernière
validation, moments forts, journal, et **les decks qui gagnent le plus** dans l'auto-jeu (dernière
itération, 10 dernières ou tout le run ; tous, maison ou vrais decks). Un clic sur un deck
affiche ses cartes. C'est un repérage (contre tout le pool, decks faibles compris) : à confirmer
avec `run.bat forge --deck` ou le ranked. Laisse `run.bat ui` ouvert pendant l'entraînement :
les résultats de chaque itération sont enregistrés (`runs\<run>\victoires.json`).

## 6. Les decks des développeurs (mode solo)

`run.bat solo` : ligue par rang (Poké Ball à Master Ball) puis grand tournoi des meilleurs contre
la méta. Résultat `results\solo\CLASSEMENT.md`.

## 7. Mettre à jour la méta

```
run.bat meta
```

Re-télécharge les listes Game8 et les parts de méta DotGG dans `lab\decks\sources\` (source, URL
et dates en tête de chaque fichier) et recalcule `meta_share.json` (population du ranked).
Les listes Limitless sont dans `lab\decks\meta\`. Voir `lab\decks\sources\README.md`.

## 8. Format des decks

```
# source: https://...      (ligne de commentaire, ignorée)
Energy: Lightning
2 Oricorio A3 66
2 Dedenne ex B3b 24
```

Une carte par ligne : nombre, nom, extension, numéro. 20 cartes, 2 exemplaires max.

## 9. Bots disponibles (`--player`, `--a`, `--b`)

| Code | Bot |
|---|---|
| `z0` | réseau seul (politique), très rapide |
| `z8`, `z32`, `z128` | réseau + recherche Gumbel (8, 32, 128 simulations) |
| `e2`, `e3` | expectiminimax écrit à la main, profondeur 2 ou 3 |
| `r` | aléatoire |

Diagnostic d'un matchup : `bin\pocket-zero.exe match --deck-a A.txt --deck-b B.txt --player z32:runs\principal\weights\latest.bin --player-b e3 --games 200`
(qui pilote mal quel deck ?).

## 10. Dépannage

| Problème | Solution |
|---|---|
| `'o' n'est pas reconnu...` en lançant `run.bat` | fins de ligne cassées : `git pull` (le dépôt force désormais le format Windows) |
| Disque plein | les anciennes parties sont supprimées automatiquement ; `--keep-shards` les garde |
| Avertissement NumPy « not writable » | sans effet, déjà masqué dans les versions récentes |
| « Le format des entrées a changé » | un changement incompatible : nouveau `--run` |

## 11. Ce que le bot voit

Tout ce qu'un joueur voit à l'écran, rien de plus : les cartes en jeu et leurs PV, énergies,
outils, cartes sous chaque Pokémon, statuts, effets (avec leurs montants), coût de retraite réel,
zone d'énergie (actuelle et suivante), stade, ta main, les défausses, la composition de ton deck
restant (pas son ordre), la taille de la main et du deck adverses, les points, les KO et les
attaques des derniers tours. Jamais la main adverse ni l'ordre des decks.
