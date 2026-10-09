# Pocket Deck Lab

Fork de [deckgym-core](https://github.com/bcollazo/deckgym-core) (moteur Rust de Pokémon TCG Pocket, AGPL-3.0).
Objectif : un simulateur assez fidèle pour construire des decks automatiquement — decks 100 % IA contre le méta,
contre des decks aléatoires, ou construits autour d'une ou plusieurs cartes imposées.

## État (9 octobre 2026)

- **Cartes** : 4 317 / 4 317 implémentées. Les 46 dresseurs réimprimés du Deluxe Pack Mega (B4b) sont branchés
  sur la logique de leur impression d'origine (`lab/tools/wire_reprints.py`).
- **Tests** : 1 190 passent, 0 échec (`cargo test --release --features "tui test-utils"`).
- **Fidélité** : mesurée sur 10 matchups réels (Limitless, format B4a) entre 5 decks du méta.

| Bot | Écart moyen (RMSE) vs réel | Vitesse (2 cœurs) |
|---|---|---|
| Expectiminimax profondeur 2 (`e2`) | 19,7 points | ~12 parties/s |
| Expectiminimax profondeur 3 (`e3`) | 11,8 points | ~3,7 parties/s |
| MCTS (`m`) | non mesuré | ~0,08 partie/s |

Le goulot d'étranglement est la qualité du bot, pas les règles.

## Structure

```
lab/
  decks/meta/      listes réelles de tournoi (Limitless) : butterfree, lucario, altaria, dedenne, hoopa
  decks/           decks expérimentaux
  tools/validate.py       compare le simulateur aux vrais matchups : python3 lab/tools/validate.py e3 100
  tools/wire_reprints.py  rebranche les réimpressions de dresseurs non implémentées
  tools/audit_effects.py  audit statique texte des cartes <-> code (montants, qualificatifs) : à relancer à chaque extension
  tools/audit_reprints.py logique qui cite des cartes par numéro mais oublie des impressions de la même carte
  tools/fuzz.py           decks aléatoires de tout le pool les uns contre les autres, détecte les crashs
  search/pipeline.py      recherche de decks (voir l'en-tête du fichier), lancée par le workflow « Deck search »
```

## Audit du moteur (9 octobre 2026)

- Règles vérifiées dans le code : main de départ (5 cartes, 1 Pokémon de base garanti), le premier joueur pioche
  mais n'a pas d'énergie au tour 1, 1 énergie / 1 supporter / 1 retraite par tour, pas d'évolution au premier tour
  (sauf talent type Évoli), main max 10, pas de défaite deck vide, statuts (poison 10, brûlure 20 + pile,
  sommeil pile, paralysie jusqu'à la fin du prochain tour, confusion pile), faiblesse +20, points 1/2/3,
  fossile KO = 1 point, égalité à 30 tours au total, 1 stade par tour (il remplace celui en jeu), pas de carte
  jouable sans cible (Sabrina sans banc adverse, Brock sans Golem/Onix…).
- 1 198 tests, 0 échec ; fuzz 36 000 parties sur des decks aléatoires, 0 crash ; audit statique : 1 bug corrigé
  (Pichu Crackly Toss ciblait aussi les évolués).
- Dresseurs : les 163 cartes (82 supporters, 31 objets, 26 outils, 13 stades, 11 fossiles) relues une à une,
  texte officiel contre effet et condition de jeu. Corrigés :
  - Quick-Grow Extract et Wallace : le joueur choisit la cible (avant : cible tirée au hasard) ;
  - Mythical Slab : prend les Pokémon [P] (avant : n'importe quel Pokémon de base) ;
  - Koga : marche sur toutes les impressions de Muk / Weezing (avant : 3 numéros seulement) ;
  - Dragalge ex (Poison Point) et Iris (Haxorus) : réimpressions oubliées ;
  - « énergie au hasard » (Piers, Psychic, Crawdaunt) : vraiment au hasard (avant : toujours la dernière).
  Simplifications connues, sans effet sur la recherche : les cartes « regarde / remets dans l'ordre »
  (Pokédex, Hiker, Morty, Hand Scope, Looker) ne font rien.
- Le bot expectiminimax cherche désormais sur un état « déterminisé » (il ne voit plus le haut de son deck ni la
  main adverse). Fidélité inchangée (RMSE 11,2 → 12,0, dans le bruit).

## Feuille de route

1. **Bot** : descendre sous ~5 points d'écart avec les vrais matchups (fonction de valeur, profondeur, audit de
   l'information cachée). Point faible connu : le verrou de sommeil d'Altaria (Lucario vs Altaria : +28 points).
2. **Méta** : vraies listes des 16 archétypes principaux, pondérées par leur part de tournoi.
3. **Recherche** : algorithme génétique sur toutes les cartes légales, avec contraintes optionnelles
   (« construis autour de cette carte »), test de robustesse, top 3 expliqué.
4. **Boucle réelle** : les parties jouées en vrai servent à recalibrer.

## Notes

- Pièces : aucune preuve datamine ou officielle d'un biais ; le moteur suppose 50/50.
- Premier deck testé (Fort-Ivoire / Méga-Lockpin, issu d'un prototype Python) : ~50 % en moyenne avec les règles
  exactes et `e3`, loin des 77 % annoncés par le prototype. Les résultats viendront désormais de ce moteur.
