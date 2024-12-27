# ProjetM1

## Installation : 

Pour ajouter les librairies python : 
1. Creer un environnement virtuel
```{powershell}
python -m venv .env
```
2. Activer l'environnement virtuel
```{powershell}
.env/Scripts/activate
```
3. Importer les librairies
```{powershell}
pip install -r requirements.txt
```

## Autre
Si vous avez ajouté d'autre librairies il faut faire cette commande pour ajouter a requirements.txt
```{powershell}
pip freeze > requirements.txt
```
Si problème lors de l'installation des librairies a cause de variance dans les versions (numpy) a la place de la ligne 3
```
pip install --use-deprecated=legacy-resolver -r requirements.txt
```

## Lien utiles : 

Trello : https://trello.com/b/0uE73zkB/analyse-de-texte