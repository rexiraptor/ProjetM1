import spacy

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
}

conjug_count = 0
inf_count = 0

total_count = 0

verb_obj = 0
verb_suj = 0
verb_aux = 0
total_verbs = 0

prop_sub = 0
total_sentences = len(list(doc.sents))

for token in doc:
    if not token.is_punct and token.text != "\n":
        total_count += 1
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
        if token.pos_ == "VERB":
            total_verbs += 1
            
            if token.morph.get("VerbForm")[0] == "Inf":
                inf_count += 1
            else:
                conjug_count += 1

            if any(child.dep_ in {"obj", "iobj"} for child in token.children):
                verb_obj += 1

            if any(child.dep_ in {"nsubj", "csubj"} for child in token.children):
                verb_suj += 1

            if any(child.pos_ == "AUX" for child in token.children):
                verb_aux += 1

    if token.dep_ in {"advcl", "csubj", "ccomp", "relcl"}:
        prop_sub += 1

pos_rates = {pos: count / total_count for pos, count in pos_counts.items()}

nb_ver = pos_counts["VERB"]
print(f"taux de verbes: {nb_ver/total_count}")
print(f"taux de verbes conjugues: {conjug_count/nb_ver}")
print(f"taux de verbes a l'infinitif: {inf_count/nb_ver}")
print(f"taux de phrases verbales contenant des objets: {verb_obj / total_verbs}")
print(f"taux de phrases verbales contenant des sujets: {verb_suj / total_verbs}")
print(f"Ratio de phrases verbales contenant des auxiliaires: {verb_aux / total_verbs}")
print(f"Moyenne des propositions subordonnées par phrase: {prop_sub / total_sentences}")

print(f"Le nombre total de tokens: {total_count}")
print(f"Le nombre total de phrases: {total_sentences}")
print("POS Rates:")
for pos, rate in pos_rates.items():
    print(f"{pos}: {rate:.3f}")
