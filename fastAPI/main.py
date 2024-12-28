from typing import List, Union, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import morpho

app = FastAPI()

class Text(BaseModel):
    texte: str = ""  # Initialiser avec une chaîne vide
    phrase: Optional[List[str]] = []
    indicateurs: Optional[List[str]] = []

# Chemin du fichier de résultats
resultat_path = Path(os.getcwd()).parent / "resultat" / "indicateurs.json"

# Créer le dossier 'resultat' s'il n'existe pas
resultat_path.parent.mkdir(parents=True, exist_ok=True)

# Charger ou créer un fichier JSON pour stocker les résultats
def load_indicateurs():
    if resultat_path.exists():
        with open(resultat_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"texte_complet": {"texte": "", "indicateurs": [], "phrases": []}}

def save_indicateurs(data):
    with open(resultat_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Initialisation de l'instance Text
txt = Text(texte="", phrase=[], indicateurs=[])

# Route de test pour vérifier si le serveur fonctionne
@app.get("/")
def read_root():
    return {"Hello": "World"}

# Endpoint pour envoyer une phrase et recevoir ses indicateurs
@app.post("/indic/")
def send_phrase(item: Text):  # Utilisation du modèle Text pour accepter le corps de la requête
    # Charger les indicateurs déjà enregistrés
    data = load_indicateurs()

    # Ajouter la nouvelle phrase au texte complet
    txt.texte += " " + item.texte  # Concaténation au texte complet
    txt.phrase.append(item.texte)  # Ajouter la phrase à la liste des phrases

    # Calculer les indicateurs pour la phrase
    indicateurs_phrase = morpho.stats_morpho_all(item.texte, "indicateurs")
    
    # Ajouter les indicateurs de la phrase au cumul des indicateurs du texte
    txt.indicateurs.extend(indicateurs_phrase)

    # Sauvegarder les données mises à jour
    save_indicateurs(data)

    # Mettre à jour les données avec le texte complet et les indicateurs
    data["texte_complet"] = {
        "texte": txt.texte,
        "indicateurs": txt.indicateurs,
        "phrases": txt.phrase
    }

    # Sauvegarder la mise à jour dans le fichier JSON
    save_indicateurs(data)

    return {"texte": txt.texte, "indicateurs": txt.indicateurs, "texte_complet": data["texte_complet"]}

# Endpoint pour récupérer les indicateurs du texte complet
@app.get("/indic/")
def get_indicateurs_complets():
    data = load_indicateurs()
    return data.get("texte_complet", {})

# Endpoint pour envoyer une phrase partielle et recevoir ses indicateurs
@app.post("/indic_part/")
def send_phrase_partiel(texte: str):  # Seul le texte (phrase) est envoyé
    # Calculer les indicateurs pour la phrase
    indicateurs_phrase = morpho.stats_morpho_all(texte, "indicateurs")
    
    # Retourner les indicateurs pour la phrase
    return {"phrase": texte, "indicateurs": indicateurs_phrase}

# Endpoint pour récupérer les indicateurs d'une phrase par indice
@app.get("/indic_phrase/{indice}")
def get_indicateurs_par_indice(indice: int):
    data = load_indicateurs()
    if "texte_complet" not in data or "phrases" not in data["texte_complet"]:
        raise HTTPException(status_code=404, detail="Aucune phrase enregistrée.")
    
    phrases = data["texte_complet"]["phrases"]
    indicateurs = data["texte_complet"]["indicateurs"]
    
    # Vérifier si l'indice est valide
    if indice < 0 or indice >= len(phrases):
        raise HTTPException(status_code=404, detail="Indice de phrase invalide.")
    
    # Récupérer la phrase et les indicateurs associés
    phrase = phrases[indice]
    phrase_indicateurs = indicateurs[indice]
    
    return {"phrase": phrase, "indicateurs": phrase_indicateurs}

# pour tester, excuter uvicorn main:app --reload dans l'environnement
# requete type : 
# type : POST
# URL : http://127.0.0.1:8000/indic/
# body (raw, JSON) : {  "texte": "Aujourd'hui il y'as beaucoup de nuages"}
# modification encore nécessaire pour le fichier indicateurs.json
# potentiellement retiré la gestion des indicateurs mis en place pour faire de la simple lecture dans le fichier result_indicateurs.json