# Élicitation incrémentale et recherche locale pour le problème du sac à dos multi-objectifs

## Depandances
- julia 1.10
Pour installer tous les packages nécessaires, lancer la commande suivante dans le terminal:
```
julia packages.jl
```
## Lancement

### 2 phases method
#### pharse 1:PLS
Pour lancer PLS:
le nobre d'objets et de critères sont modifiables dans le fichier PLS.jl, puis lancer le fichier `julia PLS.jl` Il y aura de la verbose dans la console et les logs seront dans `./logs_pls`.

Pour obtenir un ensemble de voisins à partir d'une solution actuelle $x $, deux listes sont créées, toutes deux de taille $L $ : une liste (L1) contenant les éléments candidats à être retirés (donc présents dans $x $) et une autre liste (L2) contenant les éléments candidats à être ajoutés (donc absents dans $x $).
Pour créer $\mathrm{L}1 $, les éléments, dans $x $, minimisant le ratio $R_{2} $, défini par
$$
R_{2}=\frac{\sum_{k=1}^{p} \lambda_{k} c_{k}^{s}}{ w^{s}}
$$
sont sélectionnés.

Pour créer L2, les éléments, non présents dans $x $, maximisant le ratio $R_{1} $ (défini par 
$$
R_{1}=\frac{\sum_{k=1}^{p} \lambda_{k} v_{k}^{s}}{\frac{w^{s}}{W-\sum_{i=1}^{n} w^{i} x_{i}+1}}
$$
) sont sélectionnés.

Nous avons utilisé un ND-Tree pour mettre à jour les solutions non dominées.

#### pharse 2: élicitation incrémentale

Pour lancer 3 types d'élicitation incrémentale:
lancer `julia elicitaion_weighted.jl` pour l'élicitation incrémentale somme pondérée, `julia elicitaion_OWA.jl` pour l'élicitation incrémentale OWA, `julia elicitaion_choquet.jl` pour l'élicitation incrémentale Choquet.

### 2em method: regret base recherche locale

lancer `julia rbls.jl` pour la méthode de regret base recherche locale.
