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

pos_rates = {pos: count / total_count for pos, count in pos_counts.items()}

nb_ver = pos_counts["VERB"]
print(f"taux de verbes conjugues: {conjug_count/nb_ver}")
print(f"taux de verbes a l'infinitife: {inf_count/nb_ver}")

print(f"Le nombre total de tokens: {total_count}")
print("POS Rates:")
for pos, rate in pos_rates.items():
    print(f"{pos}: {rate:.3f}")
