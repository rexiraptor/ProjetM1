import requests
import time
import sys
import os
import spacy
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import morpho

# URL du serveur FastAPI
url = "http://127.0.0.1:8000/indic/sep"

# Chemin vers le fichier CSV contenant les phrases
csv_file_path = Path(__file__).resolve().parents[1] / "datasets/data_fr/DAMT_FR/FR_D0420-S1-T05.csv"

# Vérification que le fichier CSV existe
if not csv_file_path.exists():
    print(f"Fichier CSV introuvable : {csv_file_path}")
    sys.exit(1)

# Charger SpaCy pour la segmentation
nlp = spacy.load('fr_core_news_lg')

def segmenter_phrases(texte):
    doc = nlp(texte)
    return [phrase.text.strip() for phrase in doc.sents]

# Charger les phrases depuis le fichier CSV via morpho
dialogue_brut = morpho.export_patient_dialogue(str(csv_file_path))

if dialogue_brut is None:
    print("Erreur : Dialogue brut introuvable ou vide.")
    sys.exit(1)

# Segmenter le texte en phrases complètes
phrases = segmenter_phrases(dialogue_brut)

if not phrases:
    print("Erreur : Aucune phrase trouvée après segmentation.")
    sys.exit(1)

# Liste pour stocker les résultats
indicateurs = {
    "TTR": [],
    "taux_adverbes": [],
    "taux_verbes_conjugué": [],
    "densite_concept_total": [],
    "surprise_rate": [],
    "phrases": []
}

# Configurer le graphique
fig, ax = plt.subplots()
ax.set_xlim(0, len(phrases))
ax.set_ylim(0, 1)  # Plage normale des taux
ax.set_xlabel("Numéro de phrase")
ax.set_ylabel("Indicateurs")
line1, = ax.plot([], [], label="TTR", color="blue")
line2, = ax.plot([], [], label="Taux d'adverbes", color="green")
line3, = ax.plot([], [], label="Taux de verbes conjugué", color="red")
line4, = ax.plot([], [], label="densite total des concepts", color="purple")
line5, = ax.plot([], [], label="surprise", color="orange")
ax.legend()

# Fonction de mise à jour du graphique
def update(frame):
    if frame < len(phrases):
        phrase = phrases[frame]
        data = {
            "texte": phrase,
            "phrase": [phrase],
            "indicateurs": []
        }

        # Envoi de la requête POST
        response = requests.post(url, json=data)

        # Vérification de la réponse
        if response.status_code == 200:
            result = response.json()
            indicateurs_phrase = result["indicateurs"][0]
            indicateurs["TTR"].append(indicateurs_phrase["TTR"])
            indicateurs["taux_adverbes"].append(indicateurs_phrase["adv_rates"])
            indicateurs["taux_verbes_conjugué"].append(indicateurs_phrase["verb_conj_rate"])
            indicateurs["densite_concept_total"].append(indicateurs_phrase["total_concept_density"])
            indicateurs["surprise_rate"].append(indicateurs_phrase["surprise_rate"])
            indicateurs["phrases"].append(phrase)

            # Mise à jour des données du graphique
            x_data = list(range(len(indicateurs["phrases"])))
            line1.set_data(x_data, indicateurs["TTR"])
            line2.set_data(x_data, indicateurs["taux_adverbes"])
            line3.set_data(x_data, indicateurs["taux_verbes_conjugué"])
            line4.set_data(x_data, indicateurs["densite_concept_total"])
            line4.set_data(x_data, indicateurs["surprise_rate"])

    return line1, line2, line3

# Animation du graphique
ani = FuncAnimation(fig, update, frames=len(phrases), repeat=False, interval=1000)

# Afficher le graphique
plt.show()
