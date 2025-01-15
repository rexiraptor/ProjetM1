import requests
import csv
import time
import sys
import os
import spacy
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import morpho

# URL du serveur FastAPI
url = "http://127.0.0.1:8000/indic/sep"

# Chemin vers le fichier CSV contenant les phrases
csv_file_path = os.path.join(os.path.dirname(__file__), '../datasets/data_fr/DAMT_FR/FR_D0420-S1-T05.csv')
csv_file_path = os.path.abspath(csv_file_path)

# Vérification que le fichier CSV existe
if not os.path.exists(csv_file_path):
    print(f"Fichier CSV introuvable : {csv_file_path}")
    sys.exit(1)

# Charger SpaCy pour la segmentation
nlp = spacy.load('fr_core_news_lg')

def segmenter_phrases(texte):
    doc = nlp(texte)
    return [phrase.text.strip() for phrase in doc.sents]

# Charger les phrases depuis le fichier CSV via morpho
dialogue_brut = morpho.export_patient_dialogue(csv_file_path)

if dialogue_brut is None:
    print("Erreur : Dialogue brut introuvable ou vide.")
    sys.exit(1)

# Segmenter le texte en phrases complètes
phrases = segmenter_phrases(dialogue_brut)

if not phrases:
    print("Erreur : Aucune phrase trouvée après segmentation.")
    sys.exit(1)

# Boucle pour envoyer chaque phrase complète au serveur
for phrase in phrases:
    data = {
        "texte": phrase,
        "phrase": [phrase],
        "indicateurs": []  # Optionnel
    }
    
    # Envoi de la requête POST
    response = requests.post(url, json=data)
    
    # Vérification de la réponse
    if response.status_code == 200:
        print(f"Réponse pour la phrase '{phrase}': {response.json()}")
    else:
        print(f"Erreur pour la phrase '{phrase}': {response.status_code}")
    
    # Pause entre les requêtes
    time.sleep(1)