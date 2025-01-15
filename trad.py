# Modele utilisé :  https://huggingface.co/Helsinki-NLP/opus-mt-en-fr
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import os

# 1 : Psychiatrist, 2 : Patient

# Nom du modèle pour la traduction EN->FR
model_name = "Helsinki-NLP/opus-mt-en-fr"

# Charger le tokenizer et le modèle
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_DAIC():
    data_path = os.path.join(os.path.dirname(__file__), "datasets/data/DAIC-WOZ")
    data_fr_path = os.path.join(os.path.dirname(__file__), "datasets/data_fr/DAIC-WOZ_FR")
    files = os.listdir(data_path)
    # files_fr = os.listdir(data_fr_path)
    # DAIC-WOZ
    for file in files:
        data = pd.read_csv(os.path.join(data_path, file), sep="\t")
        df_fr = pd.DataFrame(columns=["speaker", "original", "traduit"])
        for line in data.iterrows():
            speaker = line[1]["speaker"]
            if speaker == "Ellie":
                speaker = 1
            else:
                speaker = 2
            dialog = line[1]["value"]
            if type(dialog)  != str or dialog == "[]" or dialog == None:
                continue
            tokenized_text = tokenizer(dialog, return_tensors="pt", padding=True)
            translated = model.generate(**tokenized_text)
            tempo = pd.DataFrame(columns=["speaker", "original", "traduit"], data=[[speaker, dialog, tokenizer.decode(translated[0], skip_special_tokens=True)]])
            df_fr = pd.concat([df_fr, tempo], ignore_index=True)
        df_fr.to_csv(os.path.join(data_fr_path, "FR_" + file), sep="\t", index=False)
    print("Dossier DAIC-WOZ traduit avec succès !")

# DAMT
def translate_DAMT():
    data_path = os.path.join(os.path.dirname(__file__), "datasets/data/DAMT/transcripts/transcribed")
    files = os.listdir(data_path)
    data_fr_path = os.path.join(os.path.dirname(__file__), "datasets/data_fr/DAMT_FR")
    for file in files:
        if len(files) % 10 == 0:
            print(f"Traduction de {file} en cours... (DAMT)")
        data = pd.read_json(os.path.join(data_path, file))
        df_fr = pd.DataFrame(columns=["speaker", "original", "traduit"])
        for line in data.iterrows():
            speaker = line[1]["speaker"]
            dialog = line[1]["dialogue"]
            for sent in dialog:
                tokenized_text = tokenizer(sent, return_tensors="pt", padding=True)
                translated = model.generate(**tokenized_text)     
                tempo = pd.DataFrame(columns=["speaker", "original", "traduit"], data=[[speaker, sent, tokenizer.decode(translated[0], skip_special_tokens=True)]])
                df_fr = pd.concat([df_fr, tempo], ignore_index=True)
        df_fr.to_csv(os.path.join(data_fr_path, "FR_" + file.split(".")[0] + ".csv"), sep="\t", index=False)
    print("Le dossier DAMT traduit avec succès !")

def translate_AFINN ():
    path = os.path.join(os.path.dirname(__file__), "dictionnaire/AFINN")
    df = pd.read_csv(os.path.join(path, 'AFINN-111.txt'), sep= "\t", header=None)
    df_fr = pd.DataFrame(columns=["mot", "valeur"])
    for line in df.iterrows():
        word = line[1][0]
        tokenized_text = tokenizer(word, return_tensors="pt", padding=True)
        translated = model.generate(**tokenized_text)
        tempo = pd.DataFrame(columns=["mot", "valeur"], data=[[tokenizer.decode(translated[0], skip_special_tokens=True), line[1][1]]])
        df_fr = pd.concat([df_fr, tempo], ignore_index=True)
    df_fr.to_csv(os.path.join(path, "AFINN-111_FR.txt"), sep="\t", index=False)
    print("Dictionnaire AFINN traduit avec succès !")

def translate_DAIC_2():
    data_path = os.path.join(os.path.dirname(__file__), "datasets/data_fr/DAIC-WOZ_FR")
    files = os.listdir(data_path)
    data_fr_path = os.path.join(os.path.dirname(__file__), "datasets/data_fr/DAIC-WOZ_FR2")
    for file in files: 
        df_fr = pd.DataFrame(columns=["speaker", "original", "traduit"])
        df = pd.read_csv(os.path.join(data_path, file), sep="\t")
        df['group'] = (df['speaker'] != df['speaker'].shift()).cumsum()
        # Regrouper par ces nouveaux groupes
        grouped_df = df.groupby(['group', 'speaker']).agg({
            "original": " ".join,
            "traduit": " ".join
        }).reset_index()
        # Supprimer la colonne "group" si inutile
        grouped_df = grouped_df.drop(columns=["group"])
        # print(grouped_df.head())
        for line in grouped_df.iterrows():
            dialog = line[1]["original"]
            speaker = line[1]["speaker"]
            tokenized_text = tokenizer(dialog, return_tensors="pt", padding=True)
            translated = model.generate(**tokenized_text)
            tempo = pd.DataFrame(columns=["speaker", "original", "traduit"], data=[[speaker, dialog, tokenizer.decode(translated[0], skip_special_tokens=True)]])
            df_fr = pd.concat([df_fr, tempo], ignore_index=True)
        df_fr.to_csv(os.path.join(data_fr_path, file), sep="\t", index=False)
        print("Fichier traduit")

# translate_DAIC()
# translate_DAMT()
translate_AFINN()
print("Traduction terminée !")