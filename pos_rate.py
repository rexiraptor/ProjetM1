import spacy
from spacy.lang.fr.stop_words import STOP_WORDS

nlp = spacy.load('fr_core_news_lg')

text_path = "test.txt"
texte = ""
with open(text_path, encoding='utf-8') as f:
    texte = f.read()

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
    "AUX": 0,   # Auxiliaires
}

conjug_count = 0
inf_count = 0

total_count = 0

verb_obj = 0
verb_suj = 0
verb_aux = 0
total_verbs = 0

repetition_cons = 0
prop_sub = 0
total_phrase = 0

for phrase in doc.sents:
    total_phrase += 1
    contient_vo = False
    contient_vs = False
    contient_va = False
    token_list = set()

    for token in phrase:
        if not token.is_punct and token.text != "\n":
            total_count += 1

            if token.lemma_ not in STOP_WORDS:
                if token.lemma_ in token_list:
                    repetition_cons += 1
                else:
                    token_list.add(token.lemma_)

            if token.pos_ in pos_counts:
                pos_counts[token.pos_] += 1
            if token.pos_ == "VERB":
                total_verbs += 1
                
                if token.morph.get("VerbForm"):
                    if token.morph.get("VerbForm")[0] == "Inf":
                        inf_count += 1
                    else:
                        conjug_count += 1

                if any(child.dep_ in {"obj", "iobj"} for child in token.children):
                    contient_vo = True

                if any(child.dep_ in {"nsubj", "csubj"} for child in token.children):
                    contient_vs = True

                if any(child.pos_ == "AUX" for child in token.children):
                    contient_va = True

        if token.dep_ in {"advcl", "csubj", "ccomp", "relcl"}:
            prop_sub += 1

    if contient_vo:
        verb_obj += 1
    if contient_vs:
        verb_suj += 1
    if contient_va:
        verb_aux += 1

pos_rates = {pos: count / total_count for pos, count in pos_counts.items()}

mean_prop_sub = prop_sub / total_phrase

nb_ver = pos_counts["VERB"]
print(f"taux de verbes conjugues: {conjug_count/nb_ver}")
print(f"taux de verbes a l'infinitif: {inf_count/nb_ver}")
print(f"taux de phrases verbales contenant des objets: {verb_obj / total_phrase}")
print(f"taux de phrases verbales contenant des sujets: {verb_suj / total_phrase}")
print(f"taux de phrases verbales contenant des auxiliaires: {verb_aux / total_phrase}")
print(f"Moyenne des propositions subordonnées par phrase: {mean_prop_sub}")
print(f"Le nombre total de repetitions consecutives: {repetition_cons}")

print(f"Le nombre total de tokens: {total_count}")
print(f"Le nombre total de phrases: {total_phrase}")
print("POS Rates:")
for pos, rate in pos_rates.items():
    print(f"{pos}: {rate:.3f}")
