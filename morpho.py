import spacy
import os
import pandas as pd
nlp = spacy.load('fr_core_news_lg')
dataset_path = os.path.join(os.path.dirname(__file__), "datasets/data_fr")


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
    for token in doc:
        if not token.is_punct and token.text != "\n":
            total_count += 1
            if token.pos_ in pos_counts:
                pos_counts[token.pos_] += 1

    # pos_rates = {pos: count / total_count for pos, count in pos_counts.items()}
    return pos_counts, total_count
    # print(f"Le nombre total de tokens: {total_count}")
    # print("POS Rates:")
    # for pos, rate in pos_rates.items():
    #     print(f"{pos}: {rate:.3f}")

def export_patient_dialogue(file_path):
    df = pd.read_csv(file_path, sep="\t")
    patient_df = df[df["speaker"] == 2].reset_index()
    patient_df = patient_df.drop(columns=["index", "original", "speaker"])
    patient_tab = patient_df.to_numpy()
    return patient_tab

def stats_morpho_all(file_path):
    patient_dialogue = export_patient_dialogue(file_path)
    total_word = 0
    total_pos_counts = {
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
    for dialogue in patient_dialogue:
        # total_word = 0
        pos_count, dialogue_count = stats_morpho(dialogue[0])
        total_pos_counts = {pos: total_pos_counts[pos] + pos_count[pos] for pos in total_pos_counts}
        total_word = total_word + dialogue_count

    pos_rates = {pos: count / total_word for pos, count in total_pos_counts.items()}
    print(total_word)
    print("POS Rates:")
    for pos, rate in pos_rates.items():
        print(f"{pos}: {rate:.3f}")