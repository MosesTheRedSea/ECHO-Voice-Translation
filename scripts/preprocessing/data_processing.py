import os
import re

# Data Manipulation 
import pandas as pd
import numpy as np

# Machine Learning
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Utilities - Like Our Constants
import utils as utils
from utils import DATATOGRAB
from utils import DATATOSTORE
from utils import LANGUAGES
from utils import CONTRACTIONS

import string
# import transformers
# from transformers import AutoModel, BertTokenizerFast

# Path To DataSet For Specifed Translation
dataInputPath = "../../data/raw/"
dataOutputPath = "../../data/processed/"

# Method For Data Cleaning
def dataSetCleaning(df):
    # Removing Punctuation From Both Data set's so that translation will be easier
    df['English'] = df['English'].str.translate(str.maketrans('', '', string.punctuation))
    df['Spanish'] = df['Spanish'].str.translate(str.maketrans('', '', string.punctuation))

    # Eliminating duplicate sentence pairs
    df.drop_duplicates(subset=['English', 'Spanish'])

    #  Remove rows with missing translations.
    df[df['Spanish'] != '']
    return df

# Method For Contraction Integration
def dataSetContractionIntegration(df):
    # Lowercasing all sentences
    df['English'] = df['English'].str.lower()
    df['Spanish'] = df['Spanish'].str.lower().fillna('')

    new_data = []
    for index, row in df.iterrows():
        # Get The Original Sentence in English
        english = row['English']

         # CONTRACTIONS is a dictionary in the utils.py python file only a few
        for contraction, expansion in CONTRACTIONS.items():
            if contraction.lower() in english.lower():
                english_sentence = {
                    'English': english.replace(contraction, expansion),
                    'Code': row['Code'],
                    # Keep The Original Spanish Translation
                    'Spanish': row['Spanish']
                }
                new_data.append(english_sentence)

        # Now let's do every contraction in the sentence
        full_sentence_expansion = english
        for contraction, expansion in CONTRACTIONS.items():
             if contraction.lower() in full_sentence_expansion.lower():
                full_sentence_expansion.replace(contraction, expansion)
        english_sentence = {
                    'English': full_sentence_expansion,
                    'Code': row['Code'],
                    # Keep The Original Spanish Translation
                    'Spanish': row['Spanish']
                }
        new_data.append(english_sentence)

    dataFrame = pd.DataFrame(new_data)   
    return pd.concat([df, dataFrame], ignore_index=True)

# BERT - Hugging Face Transformers For Preprocessing Tokenization Converstion
def dataSetBERToken(df):
    
    pass


# LOAD MODEL DATA - Convert To CSV & Clean
# This is our dataset - Finally Got the relative path so it can work on call devices
df = pd.read_csv(dataInputPath + DATATOGRAB, sep='\t', header=None) 
# Incase we want to implement this with other languages AVOID HARDCODING - Moses Adewolu
df.columns = [LANGUAGES[0], "Code", LANGUAGES[1]]

# Data Processing Sequence
print(f"\n Data Set \n{df}")

# Perform Contraction Integration
df = dataSetContractionIntegration(df)
print(f"\n Contraction Integration \n{df}")

# Clean The DataSet
df = dataSetCleaning(df)
print(f"\n Dataset Cleaning \n{df}")

# BERT Tokenization


# # English/Spanish/English_Spanish_translation.csv
# outputAddOns = f"{LANGUAGES[0]}/{LANGUAGES[1]}/{LANGUAGES[0]}_{LANGUAGES[1]}_translation.csv"
# df.to_csv(dataOutputPath + outputAddOns, index=False, encoding="utf-8")
# print(f"DataFrame -> CSV -> {dataOutputPath + outputAddOns}")