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
urlInit = "http://127.0.0.1:8000/init"
url2 = "http://127.0.0.1:8000/indic_part"

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
    "phrases": []
}

emotions = {
    "anger_rate": [],
    "disgust_rate": [],
    "fear_rate": [],
    "joy_rate": [],
    "sadness_rate": [],
    "surprise_rate": [],
    "phrases": []
}

indicateurs_T = {
    "TTR": [],
    "taux_adverbes": [],
    "taux_verbes_conjugué": [],
    "densite_concept_total": [],
    "phrases": []
}

emotions_T = {
    "anger_rate": [],
    "disgust_rate": [],
    "fear_rate": [],
    "joy_rate": [],
    "sadness_rate": [],
    "surprise_rate": [],
    "phrases": []
}


# Configurer le graphique
fig, ax = plt.subplots(2,2)
ax[0,0].set_xlim(0, 100)
ax[0,0].set_ylim(0, 1)
ax[0,0].set_xlabel("Numéro de phrase")
ax[0,0].set_ylabel("Indicateurs cumulé")
line1, = ax[0,0].plot([], [], label="TTR", color="blue")
line2, = ax[0,0].plot([], [], label="Taux d'adverbes", color="green")
line3, = ax[0,0].plot([], [], label="Taux de verbes conjugué", color="red")
line4, = ax[0,0].plot([], [], label="densite total des concepts", color="purple")
ax[0,0].legend()
ax[0,0].set_title("Graph indicateur cumulé")

ax[0,1].set_xlim(0, 100)
ax[0,1].set_ylim(0, 0.1)
ax[0,1].set_xlabel("Numéro de phrase")
ax[0,1].set_ylabel("Emotions_rate cumulé")
lineE1, = ax[0,1].plot([], [], label="sadness", color="blue")
lineE2, = ax[0,1].plot([], [], label="disgust", color="green")
lineE3, = ax[0,1].plot([], [], label="anger", color="red")
lineE4, = ax[0,1].plot([], [], label="fear", color="purple")
lineE5, = ax[0,1].plot([], [], label="joy", color="yellow")
lineE6, = ax[0,1].plot([], [], label="surprise", color="orange")
ax[0,1].legend()
ax[0,1].set_title("Graph emotions cumulé")

ax[1,0].set_xlim(0, 100)
ax[1,0].set_ylim(0, 1)
ax[1,0].set_xlabel("Numéro de phrase")
ax[1,0].set_ylabel("Indicateurs cumulé")
lineT1, = ax[1,0].plot([], [], label="TTR", color="blue")
lineT2, = ax[1,0].plot([], [], label="Taux d'adverbes", color="green")
lineT3, = ax[1,0].plot([], [], label="Taux de verbes conjugué", color="red")
lineT4, = ax[1,0].plot([], [], label="densite total des concepts", color="purple")
ax[1,0].legend()
ax[1,0].set_title("Graph indicateur phrase par phrase")

ax[1,1].set_xlim(0, 100)
ax[1,1].set_ylim(0, 0.1)
ax[1,1].set_xlabel("Numéro de phrase")
ax[1,1].set_ylabel("Emotions_rate cumulé")
lineTE1, = ax[1,1].plot([], [], label="sadness", color="blue")
lineTE2, = ax[1,1].plot([], [], label="disgust", color="green")
lineTE3, = ax[1,1].plot([], [], label="anger", color="red")
lineTE4, = ax[1,1].plot([], [], label="fear", color="purple")
lineTE5, = ax[1,1].plot([], [], label="joy", color="yellow")
lineTE6, = ax[1,1].plot([], [], label="surprise", color="orange")
ax[1,1].legend()
ax[1,1].set_title("Graph emotion phrase par phrase")

# Appel à l'endpoint pour initialiser le temps
response_init = requests.post(urlInit)
if response_init.status_code == 200:
    print("Temps initialisé avec succès :", response_init.json())
else:
    print("Erreur lors de l'initialisation du temps :", response_init.status_code, response_init.text)
    sys.exit(1)

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
            indicateurs["phrases"].append(phrase)
            
            emotions["anger_rate"].append(indicateurs_phrase["anger_rate"])
            emotions["disgust_rate"].append(indicateurs_phrase["disgust_rate"])
            emotions["fear_rate"].append(indicateurs_phrase["fear_rate"])
            emotions["joy_rate"].append(indicateurs_phrase["joy_rate"])
            emotions["sadness_rate"].append(indicateurs_phrase["sadness_rate"])
            emotions["surprise_rate"].append(indicateurs_phrase["surprise_rate"])
            emotions["phrases"].append(phrase)
                

            # Mise à jour des données du graphique
            x_data = list(range(len(indicateurs["phrases"])))
            line1.set_data(x_data, indicateurs["TTR"])
            line2.set_data(x_data, indicateurs["taux_adverbes"])
            line3.set_data(x_data, indicateurs["taux_verbes_conjugué"])
            line4.set_data(x_data, indicateurs["densite_concept_total"])
            
            lineE1.set_data(x_data, emotions["anger_rate"])
            lineE2.set_data(x_data, emotions["disgust_rate"])
            lineE3.set_data(x_data, emotions["fear_rate"])
            lineE4.set_data(x_data, emotions["joy_rate"])
            lineE5.set_data(x_data, emotions["sadness_rate"])
            lineE6.set_data(x_data, emotions["joy_rate"])
            
        response2 = requests.post(url2, json={"texte":phrase})

        # Vérification de la réponse
        if response2.status_code == 200: 
            result = response2.json()   
            for i in result:
                print(i)
                print(result[i])
            print("###########")
            indicateurs_phrase = result["indicateurs"]
            indicateurs_T["TTR"].append(indicateurs_phrase["TTR"])
            indicateurs_T["taux_adverbes"].append(indicateurs_phrase["adv_rates"])
            indicateurs_T["taux_verbes_conjugué"].append(indicateurs_phrase["verb_conj_rate"])
            indicateurs_T["densite_concept_total"].append(indicateurs_phrase["total_concept_density"])
            indicateurs_T["phrases"].append(phrase)
            
            emotions_T["anger_rate"].append(indicateurs_phrase["anger_rate"])
            emotions_T["disgust_rate"].append(indicateurs_phrase["disgust_rate"])
            emotions_T["fear_rate"].append(indicateurs_phrase["fear_rate"])
            emotions_T["joy_rate"].append(indicateurs_phrase["joy_rate"])
            emotions_T["sadness_rate"].append(indicateurs_phrase["sadness_rate"])
            emotions_T["surprise_rate"].append(indicateurs_phrase["surprise_rate"])
            emotions_T["phrases"].append(phrase)
            
            lineT1.set_data(x_data, indicateurs_T["TTR"])
            lineT2.set_data(x_data, indicateurs_T["taux_adverbes"])
            lineT3.set_data(x_data, indicateurs_T["taux_verbes_conjugué"])
            lineT4.set_data(x_data, indicateurs_T["densite_concept_total"])
            
            lineTE1.set_data(x_data, emotions_T["anger_rate"])
            lineTE2.set_data(x_data, emotions_T["disgust_rate"])
            lineTE3.set_data(x_data, emotions_T["fear_rate"])
            lineTE4.set_data(x_data, emotions_T["joy_rate"])
            lineTE5.set_data(x_data, emotions_T["sadness_rate"])
            lineTE6.set_data(x_data, emotions_T["joy_rate"])
            
            if(len(indicateurs['phrases'])>100):
            
                ax[0,0].set_xlim(0, len(indicateurs["phrases"]))
                ax[0,0].set_ylim(0, max(max(indicateurs["TTR"], default=1), 1))

                ax[0,1].set_xlim(0, len(emotions["phrases"]))
                ax[0,1].set_ylim(0, max(max(emotions["anger_rate"], default=0.1), 0.1))
                
                ax[1,0].set_xlim(0, len(indicateurs["phrases"]))
                ax[1,0].set_ylim(0, max(max(indicateurs["TTR"], default=1), 1))

                ax[1,1].set_xlim(0, len(emotions["phrases"]))
                ax[1,1].set_ylim(0, max(max(emotions["anger_rate"], default=0.1), 0.1))


    return line1, line2, line3, line4, lineE1, lineE2, lineE3, lineE4, lineE5, lineE6, lineT1, lineT2, lineT3, lineT4, lineTE1, lineTE2, lineTE3, lineTE4, lineTE5, lineTE6

# Animation du graphique
ani = FuncAnimation(fig, update, frames=len(phrases), repeat=False, interval=1000)

# Afficher le graphique
plt.show()
