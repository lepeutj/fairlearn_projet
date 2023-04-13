# fairlearn_projet
Projet mené en groupe sur l'article.

# Résumé
Les algorithmes prennent énormément de décisions aujourd'hui. Il a été démontré à de nombreuses reprises que les algorithmes prennent des décisions biaisées. D'abord, puisqu'ils ne sont que synthèse des données réelles dont on dispose et nous n'échappons à nos propres biais, en partie ceux de selections. Puis, parce qu'en créant les algorithmes, on va chercher les variables qui discriminent pour notre problème de classification.

On passe alors dans un cadre mathématique pour définir une notion d'équité, que l'on est à même de mesurer, par rapport à une variable discriminante. 
L'objet de l'article, est de minimiser, voir d'annuler l'influence de la variable discriminante, que l'on appelle aussi variable sensible. 
Ici les modifications sont apportées après le traitement de données. On agit sur l'estimateur, qu'on sait biaisé.

Le modèle final perd en performance, sur la base de données observée, et un travail de réflexion doit se faire autour du compromis Equité/Précision (même problème en statistiques : biais/variance).

L'article propose des méthodes dans le cas où l'on est face à un problème de classification multiclasse, et où la variable sensible possède plusieurs modalités. Nous nous sommes placés dans un cadre plus simple pour tester. 2 classes à prédire, 2 classes dans la variable sensible.

 
