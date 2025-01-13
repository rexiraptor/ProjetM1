import spacy
import os
import pandas as pd
from collections import Counter
import numpy as np
import json

nlp = spacy.load('fr_core_news_lg')
dataset_path = os.path.join(os.path.dirname(__file__), "datasets/data_fr")
result_path = os.path.join(os.path.dirname(__file__), "resultat")

def stats_words(texte):
    doc = nlp(texte)
    freq_list = Counter(token.text for token in doc if not token.is_punct and not token.pos_ == "DET" and not token.is_stop and not "\n"  in token.text)
    mean = np.mean(list(freq_list.values()))
    range = max(freq_list.values()) - min(freq_list.values())
    std = np.std(list(freq_list.values()))
    return mean, range, std

def lexical_diversity(texte, a):
    doc = nlp(texte)
    freq_list = Counter(token.text for token in doc if not token.is_punct and not token.is_stop and not "\n" in token.text)
    N = len(doc)
    V = len(freq_list)
    v1 = len([word for word in freq_list if freq_list[word] == 1])
    brunet_index = N ** (V ** -a)
    honore_stats = 100 * np.log(N) / (1 - v1 / N)
    ratio_unique = v1 / N
    return brunet_index, honore_stats, ratio_unique
def stats_morpho(texte):
    doc = nlp(texte)

    # POS tags
    pos_counts = {
        "ADJ": 0,  # Adjectifs
        "ADP": 0,  # Adpositions (prepositions/postpositions)
        "ADV": 0,  # Adverbes
        "CONJ": 0,  # Conjunctions
        "DET": 0,  # Determinants
        "NOUN": 0,  # Noms
        "PRON": 0,  # Pronoms
        "VERB": 0,  # Verbes
        "PROPN": 0,  # Noms propres
    }
    total_count = 0
    conjug_count = 0
    inf_count = 0
    for token in doc:
        if not token.is_punct and token.text != "\n":
            total_count += 1
            if token.pos_ in pos_counts:
                pos_counts[token.pos_] += 1
            if token.pos_ == "VERB":
                if token.morph.get("VerbForm")[0] == "Inf":
                    inf_count += 1
                else:
                    conjug_count += 1
    nb_ver = pos_counts["VERB"]
    rate_conjug = conjug_count/nb_ver
    rate_inf = inf_count/nb_ver
    pos_rates = {pos: count / total_count for pos, count in pos_counts.items()}
    return pos_rates, total_count, rate_conjug, rate_inf

def export_patient_dialogue(file_path):
    try :
        df = pd.read_csv(file_path, sep="\t")
    except :
        print("File not found")
        return None
    df = pd.read_csv(file_path, sep="\t")
    patient_df = df[df["speaker"] == 2].reset_index()
    patient_df = patient_df.drop(columns=["index", "original", "speaker"])
    patient_tab = patient_df.to_numpy()
    str_dialogue = " ".join(patient_tab[:, 0])
    return str_dialogue

def stats_morpho_all(file_path):
    nom_fichier = file_path.split("/")[-1]
    patient_dialogue = export_patient_dialogue(file_path)
    pos_rates, total_word, rate_conj, rate_inf = stats_morpho(patient_dialogue)
    mean, range, std = stats_words(patient_dialogue)
    brunet_index, honore_stats, ratio_unique = lexical_diversity(patient_dialogue, 0.165)
    json_file = {
        "adj_rate" : pos_rates["ADJ"],
        "adp_rate" : pos_rates["ADP"],
        "adv_rates": pos_rates["ADV"],  
        "conj_rate": pos_rates["CONJ"],  
        "det_rate": pos_rates["DET"],  
        "noun_rate": pos_rates["NOUN"],
        "pron_rate": pos_rates["PRON"],
        "verb_rate": pos_rates["VERB"],
        "propn_rate": pos_rates["PROPN"],
        "verb_conj_rate" : rate_conj,
        "verb_inf_rate" : rate_inf,
        "total_words" : total_word,
        "mean_freq_words" : float(mean),
        "range_freq_words" : range,
        "std_freq_words" : float(std),
        "Brunet_index" : brunet_index,
        "Honore_statistic" : float(honore_stats),
        "TTR" : ratio_unique
    }
    with open(os.path.join(result_path, "result_" + nom_fichier + ".json"), "w") as f:
        json.dump(json_file, f, indent=4)
    print("Fichier json généré")
    return 0

file = stats_morpho_all(os.path.join(dataset_path, "DAMT_FR/FR_D0420-S1-T05.csv"))

