import spacy
import os
import pandas as pd
nlp = spacy.load('fr_core_news_lg')
dataset_path = os.path.join(os.path.dirname(__file__), "datasets/data_fr")