# Meta decklists / Listes méta

Updated automatically twice a week by the **Meta update** GitHub workflow
(`.github/workflows/meta-update.yml`), or right now with `run.bat meta`. Every file starts with
`#` lines (ignored by the engine): site, source URL, retrieval and publication dates, tier or
share.

*Mises à jour automatiquement deux fois par semaine par le workflow GitHub **Meta update**, ou
tout de suite avec `run.bat meta`. Chaque fichier commence par des lignes `#` (ignorées par le
moteur) : site, URL de la source, dates de récupération et de publication, tier ou part.*

| Source | What it gives / Ce qu'elle donne | Files / Fichiers |
|---|---|---|
| [Limitless](https://play.limitlesstcg.com/) public API | standard tournaments of the last 14 days (≥ 32 players): for each archetype ≥ 1 % of the players, the list of its best finisher, its share and win rate / tournois standard des 14 derniers jours : pour chaque archétype ≥ 1 %, la liste de son meilleur joueur, sa part et son taux de victoire | `limitless/*.txt`, `limitless/shares.json` |
| [Game8](https://game8.co/games/Pokemon-TCG-Pocket/archives/477754) "Best Decks Tier List" | complete lists of the S and A tiers / listes complètes des tiers S et A | `game8/*.txt` |
| [ptcgpocket.gg (DotGG)](https://ptcgpocket.gg/) public tier-list API | meta share of each archetype (no lists: premium) / part de méta de chaque archétype (pas de listes : réservées aux abonnés) | `dotgg/shares.json` |

`meta_share.json` spreads each DotGG archetype's share over the deck files (Limitless reference
lists in `lab/decks/meta/` and the lists here) that contain every Pokémon of its name: it is the
population of the simulated ranked and the meta of the forge. Archetypes with no known list are
in `unmatched`. / *répartit la part de chaque archétype DotGG sur les listes qui contiennent tous
les Pokémon de son nom : la population du ranked simulé et la méta de la forge.*

Rules of the update / Règles de la mise à jour:

- a file is rewritten only when its cards change / un fichier n'est réécrit que si ses cartes changent;
- a deck that leaves a source's list is removed / un deck qui sort de la liste d'une source est supprimé;
- a list with a card the engine does not know yet (new expansion) is skipped and reported /
  une liste avec une carte inconnue du moteur (nouvelle extension) est ignorée et signalée;
- a source whose page layout changed fails without touching its files / une source dont la page
  a changé échoue sans toucher à ses fichiers.

Adding a source / Ajouter une source: a function in `lab/tools/meta_sources.py` that writes its
decks with `write_deck(...)`, declared in `SOURCES`.
