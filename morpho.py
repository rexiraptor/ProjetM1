import spacy
import os
import pandas as pd
from collections import Counter
import numpy as np
nlp = spacy.load('fr_core_news_lg')
dataset_path = os.path.join(os.path.dirname(__file__), "datasets/data_fr")

def stats_words(texte):
    doc = nlp(texte)
    freq_list = Counter(token.text for token in doc if not token.is_punct and not token.pos_ == "DET" and not token.is_stop and not "\n"  in token.text)
    mean = np.mean(list(freq_list.values()))
    range = max(freq_list.values()) - min(freq_list.values())
    std = np.std(list(freq_list.values()))
    print("Moyenne :", mean)
    print("Range :", range)
    print("Ecart-type :", std)

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
    patient_dialogue = export_patient_dialogue(file_path)
    pos_rates, total_word, rate_conj, rate_inf = stats_morpho(patient_dialogue)
    print("--------------------------")
    print("Statistiques General")
    print("--------------------------")
    print("Nbre total de mot :", total_word)
    stats_words(patient_dialogue)
    print("--------------------------")
    print("Morphology Statistiques")
    print("--------------------------")
    print("POS Rates :")
    for pos, rate in pos_rates.items():
        print(f"{pos}: {rate:.3f}")
    print("Rate of conjugated verbs :", rate_conj)
    print("Rate of infinitive verbs :", rate_inf)
    brunet_index, honore_stats, ratio_unique = lexical_diversity(patient_dialogue, 0.165)
    print("--------------------------")
    print("Lexical Diversity")
    print("--------------------------")
    print("Brunet Index :", brunet_index)
    print("Honore Statistic :", honore_stats)
    print("Ratio Unique :", ratio_unique)

stats_morpho_all(os.path.join(dataset_path, "DAMT_FR/FR_D0420-S1-T05.csv"))