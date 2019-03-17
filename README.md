Plusieurs modèles ont été implémentés pour les prédictions : un gradient boosting et une régression logistique.

Les données étaient initialement nettoyées, je n'ai constaté aucune données manquantes, permettant ainsi une exploitation et une analyse plus simple de celles-ci. 

Dans le fichier "EDA.py" est implémentée la classe EDA qui permet d'analyser le dataset initial, de traiter et modifier les variables que ce soit en les binarisant, ou les 'dummyfiant' pour in fine arriver à un dataset prêt à l'emploi pour la partie de modélisation. Ce dataset est le fichier 'training_table.csv' qu'il est possible de directement importer pour la partie modélisation.

Le fichier "Modele.py" présente la classe "Model" permettant d'implémenter les différents dataset d'entraînement, les modèles, leur metrics d'évaluation, l'importance finale accordée aux features et leur prédictions finales.

Le rapport final se présente sous la forme d'un Jupyter Notebook avec une version de Python 3.7.2 (selon moi, jusqu'à 3.5 devrait fonctionner), toutes les cellules sont déjà run, vous pouvez tout relancer par vous-même, attention tout de même à l'entraînement des modèles avec recherche sur grille et cross-validation qui prennent environ 10minutes.

Il est possible, que ce soit dans le rapport ou dans le code, qu'il y ait un mix entre français et anglais, la plupart des termes techniques anglais étant utilisés ainsi en français.

Vous trouverez un fichier texte requirements.txt avec les versions des packages que j'ai utilisé.

# Setup Local
## create virtualenv
Link: <https://github.com/pyenv/pyenv>
    
	$ pyenv virtualenv 3.7.2 disease
	$ pyenv activate disease
## install libraries
    $ pip install --upgrade -r requirements.txt
## deactivate
    $ pyenv deactivate
