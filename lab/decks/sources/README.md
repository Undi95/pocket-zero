# Decks méta hors Limitless

Plusieurs sources valent mieux qu'une : chaque source a son dossier, chaque fichier garde en
tête (`#`, ignoré par le moteur) le site, l'URL, la date de récupération et de publication, le
tier ou la part de méta. Les listes Limitless restent dans `lab/decks/meta/` pour comparer.

| Source | Ce qu'elle donne | Fichiers |
|---|---|---|
| Game8, « Best Decks Tier List » | 8 listes complètes (tiers S et A+), format Deluxe Pack Mega (B4b), mise à jour du 30/09/2026 | `game8/*.txt` |
| ptcgpocket.gg (DotGG), API publique de la tier list | part de méta de chaque archétype (decks enregistrés, période la plus récente, ex. `b4a-day10`) ; pas de listes (réservées aux abonnés) | `dotgg/shares.json` |

`meta_share.json` répartit la part de chaque archétype DotGG sur les fichiers de decks (Limitless
et autres sources) qui contiennent tous les Pokémon de son nom : c'est la population du ranked
simulé (`run.bat ranked`). Les archétypes sans liste connue sont dans `unmatched`.

## Mettre à jour (à chaque extension ou changement de méta)

```
.venv\Scripts\python lab\tools\meta_sources.py
```

Re-télécharge toutes les sources, réécrit les listes et `meta_share.json`. Une source dont la
page a changé de forme affiche « échec » et ses anciens fichiers restent en place. Pour ajouter
une source : une fonction dans `lab/tools/meta_sources.py`, déclarée dans `SOURCES`.
