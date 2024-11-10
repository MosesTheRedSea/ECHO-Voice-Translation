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

# Utilities - Like Our Constants and Other Methods
import utils as utils
from utils import DATATOGRAB
from utils import DATATOSTORE
from utils import LANGUAGES
from utils import CONTRACTIONS
from utils import INPUTPATH
from utils import TRAINPATH
from utils import TESTPATH

import string
import transformers
from transformers import BertTokenizer, BertModel

# Method For Data Cleaning
def dataSetCleaning(df):
    # Lowercasing all sentences
    df['English'] = df['English'].str.lower()
    df['Spanish'] = df['Spanish'].str.lower().fillna('')

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
    new_data = []
    for _, row in df.iterrows():
        words = row["English"].split()
        expanded_words = [CONTRACTIONS[word.lower()] if word.lower() in CONTRACTIONS else word for word in words]
        expanded_sentence = ' '.join(expanded_words)
        english_sentence = {
            "English": expanded_sentence,
            "Spanish": row["Spanish"]
        }
        new_data.append(english_sentence)
    dataFrame = pd.DataFrame(new_data)
    return pd.concat([df, dataFrame]).drop_duplicates().reset_index(drop=True)

# BERT - Hugging Face Transformers For Preprocessing Tokenization Converstion
def dataSetBertEmbeddings(text, model, tokenizer):
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = model(**encoded_input)
    word_embeddings = output.last_hidden_state[:, 0, :]
    return word_embeddings.squeeze().numpy()  

# Testing Code 
# Method For Contraction Integration
# def dataSetContractionIntegration(df):
#     new_data = []
#     for index, row in df.iterrows():
#         # Get The Original Sentence in English
#         full_sentence_expansion = row['English']
#         for contraction, expansion in CONTRACTIONS.items():
#              if contraction.lower() in full_sentence_expansion.lower():
#                 full_sentence_expansion = full_sentence_expansion.replace(contraction, expansion)
#         english_sentence = {
#                     'English': full_sentence_expansion,
#                     'Spanish': row['Spanish']
#         }
#         new_data.append(english_sentence)
#     dataFrame = pd.DataFrame(new_data)  
#     #return pd.concat([df, dataFrame])
#     return pd.concat(([df, dataFrame])).drop_duplicates().reset_index(drop=True)

# def dataSetContractionIntegration(df):
#     new_data = []
#     for index, row in df.iterrows():
#         original_english = row['English']
#         variations = [original_english]
        
#         for contraction, expansion in CONTRACTIONS.items():
#             temp_variations = []
#             for sentence in variations:
#                 if contraction.lower() in sentence.lower():
#                     temp_variations.append(sentence)
#                     expanded_sentence = sentence.replace(contraction, expansion)
#                     temp_variations.append(expanded_sentence)
#                 else:
#                     temp_variations.append(sentence)
#             variations = temp_variations 
#         variations = list(set(variations))
#         for variation in variations:
#             new_data.append({
#                 'English': variation,
#                 'Spanish': row['Spanish'] 
#             })
#     new_data_df = pd.DataFrame(new_data)
#     return pd.concat([df, new_data_df], ignore_index=True)
#     # return pd.concat(([df, dataFrame])).drop_duplicates().reset_index(drop=True)

# LOAD MODEL DATA - Convert To CSV & Clean
# This is our dataset - Finally Got the relative path so it can work on call devices
df = pd.read_csv(INPUTPATH + DATATOGRAB, sep='\t', header=None) 

# Incase we want to implement this with other languages AVOID HARDCODING - Moses Adewolu
df.columns = ["number", LANGUAGES[0], "Code", LANGUAGES[1]]

df = df[[LANGUAGES[0], LANGUAGES[1]]]

# Data Processing Sequence
print(f"\n Data Set \n{df}")

# Perform Contraction Integration
df = dataSetContractionIntegration(df)
print(f"\n Contraction Integration \n{df}")

# Clean The DataSet
df = dataSetCleaning(df)
print(f"\n Dataset Cleaning \n{df}")

# BERT Tokenization
# Initialize the BERT tokenizer and model
english_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
english_model = BertModel.from_pretrained("bert-base-uncased")

# We Found BETO : A Spanish BERT (Tokenization)
# https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased
spanish_tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
spanish_model = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

# Get BERT embeddings for English
df['English BERT'] = df['English'].apply(lambda x: dataSetBertEmbeddings(x, english_model, english_tokenizer))

# Get BERT embeddings for Spanish
df['Spanish BERT'] = df['Spanish'].apply(lambda x: dataSetBertEmbeddings(x, spanish_model, spanish_tokenizer))

df = df[[LANGUAGES[0], 'English BERT', LANGUAGES[1], 'Spanish BERT']]

# Clean The DataSet - 
print(f"\n BERT Embedding \n{df}")

# Splitting The Dataset 
# 80% For Training Our Model 
# 20% Testing our GRU 

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print(f"Training set shape: {train_df.shape}")

print(f"Testing set shape: {test_df.shape}")

# I need to save the Training and Testing Data to the Correct Path

# Correct Path For Train Data Location Data/Processed/English/Spanish/Train
train_output_path = TRAINPATH + f"{LANGUAGES[0]}_{LANGUAGES[1]}_train.csv"

# Correct Path For Test Data Location Data/Processed/English/Spanish/Test
test_output_path = TESTPATH + f"{LANGUAGES[0]}_{LANGUAGES[1]}_test.csv"

# Creating The Final CSV We Use to Train our Model 
train_df.to_csv(train_output_path, index=False, encoding="utf-8")

# Creating The Final CSV We Use to Test our Model 
test_df.to_csv(test_output_path, index=False, encoding="utf-8")

print(f"Training DataFrame saved to: {train_output_path}")

print(f"Testing DataFrame saved to: {test_output_path}")

# import os
# import re

# # Data Manipulation 
# import pandas as pd
# import numpy as np

# # Machine Learning
# import torch
# import torch.nn as nn
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# # Utilities - Like Our Constants and Other Methods
# import utils as utils
# from utils import DATATOGRAB
# from utils import DATATOSTORE
# from utils import LANGUAGES
# from utils import CONTRACTIONS
# from utils import INPUTPATH
# from utils import TRAINPATH
# from utils import TESTPATH

# import string
# import transformers
# from transformers import BertTokenizer, BertModel

# # Method For Data Cleaning
# def dataSetCleaning(df):
#     # Lowercasing all sentences
#     df['English'] = df['English'].str.lower()
#     df['Spanish'] = df['Spanish'].str.lower().fillna('')

#     # Removing Punctuation From Both Data set's so that translation will be easier
#     df['English'] = df['English'].str.translate(str.maketrans('', '', string.punctuation))
#     df['Spanish'] = df['Spanish'].str.translate(str.maketrans('', '', string.punctuation))

#     # Eliminating duplicate sentence pairs
#     df.drop_duplicates(subset=['English', 'Spanish'])

#     #  Remove rows with missing translations.
#     df[df['Spanish'] != '']
#     return df

# # Method For Contraction Integration
# def dataSetContractionIntegration(df):
#     new_data = []
#     for _, row in df.iterrows():
#         words = row["English"].split()
#         expanded_words = [CONTRACTIONS[word.lower()] if word.lower() in CONTRACTIONS else word for word in words]
#         expanded_sentence = ' '.join(expanded_words)
#         english_sentence = {
#             "English": expanded_sentence,
#             "Spanish": row["Spanish"]
#         }
#         new_data.append(english_sentence)
#     dataFrame = pd.DataFrame(new_data)
#     return pd.concat([df, dataFrame]).drop_duplicates().reset_index(drop=True)

# # BERT - Hugging Face Transformers For Preprocessing Tokenization Converstion
# def dataSetBertEmbeddings(text, model, tokenizer):
#     print("Starting embedding for:", text)
#     encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
#     print("Tokenization complete.")
#     with torch.no_grad():
#         output = model(**encoded_input)
#     print("Model inference complete.")
#     word_embeddings = output.last_hidden_state[:, 0, :]
#     print("Embedding extraction complete.")
#     return word_embeddings.squeeze().numpy()  

# Testing Code 
# Method For Contraction Integration
# def dataSetContractionIntegration(df):
#     new_data = []
#     for index, row in df.iterrows():
#         # Get The Original Sentence in English
#         full_sentence_expansion = row['English']
#         for contraction, expansion in CONTRACTIONS.items():
#              if contraction.lower() in full_sentence_expansion.lower():
#                 full_sentence_expansion = full_sentence_expansion.replace(contraction, expansion)
#         english_sentence = {
#                     'English': full_sentence_expansion,
#                     'Spanish': row['Spanish']
#         }
#         new_data.append(english_sentence)
#     dataFrame = pd.DataFrame(new_data)  
#     #return pd.concat([df, dataFrame])
#     return pd.concat(([df, dataFrame])).drop_duplicates().reset_index(drop=True)

# def dataSetContractionIntegration(df):
#     new_data = []
#     for index, row in df.iterrows():
#         original_english = row['English']
#         variations = [original_english]
        
#         for contraction, expansion in CONTRACTIONS.items():
#             temp_variations = []
#             for sentence in variations:
#                 if contraction.lower() in sentence.lower():
#                     temp_variations.append(sentence)
#                     expanded_sentence = sentence.replace(contraction, expansion)
#                     temp_variations.append(expanded_sentence)
#                 else:
#                     temp_variations.append(sentence)
#             variations = temp_variations 
#         variations = list(set(variations))
#         for variation in variations:
#             new_data.append({
#                 'English': variation,
#                 'Spanish': row['Spanish'] 
#             })
#     new_data_df = pd.DataFrame(new_data)
#     return pd.concat([df, new_data_df], ignore_index=True)
#     # return pd.concat(([df, dataFrame])).drop_duplicates().reset_index(drop=True)

# LOAD MODEL DATA - Convert To CSV & Clean
# This is our dataset - Finally Got the relative path so it can work on call devices
# df = pd.read_csv(INPUTPATH + DATATOGRAB, sep='\t', header=None) 

# # Incase we want to implement this with other languages AVOID HARDCODING - Moses Adewolu
# df.columns = ["number", LANGUAGES[0], "Code", LANGUAGES[1]]

# df = df[[LANGUAGES[0], LANGUAGES[1]]]

# # Data Processing Sequence
# print(f"\n Data Set \n{df}")

# # Perform Contraction Integration
# df = dataSetContractionIntegration(df)
# print(f"\n Contraction Integration \n{df}")

# # Clean The DataSet
# df = dataSetCleaning(df)
# print(f"\n Dataset Cleaning \n{df}")

# # BERT Tokenization
# # Initialize the BERT tokenizer and model
# english_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# english_model = BertModel.from_pretrained("bert-base-uncased")

# # We Found BETO : A Spanish BERT (Tokenization)
# # https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased
# spanish_tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
# spanish_model = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

# # Get BERT embeddings for English
# df['English BERT'] = df['English'].apply(lambda x: dataSetBertEmbeddings(x, english_model, english_tokenizer))

# # Get BERT embeddings for Spanish
# df['Spanish BERT'] = df['Spanish'].apply(lambda x: dataSetBertEmbeddings(x, spanish_model, spanish_tokenizer))

# df = df[[LANGUAGES[0], 'English BERT', LANGUAGES[1], 'Spanish BERT']]

# # Clean The DataSet - 
# print(f"\n BERT Embedding \n{df}")

# # Splitting The Dataset 
# # 80% For Training Our Model 
# # 20% Testing our GRU 

# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# # Print the shapes of the training and testing sets
# print(f"Training set shape: {train_df.shape}")

# print(f"Testing set shape: {test_df.shape}")

# # I need to save the Training and Testing Data to the Correct Path

# # Correct Path For Train Data Location Data/Processed/English/Spanish/Train
# train_output_path = TRAINPATH + f"{LANGUAGES[0]}_{LANGUAGES[1]}_train.csv"

# # Correct Path For Test Data Location Data/Processed/English/Spanish/Test
# test_output_path = TESTPATH + f"{LANGUAGES[0]}_{LANGUAGES[1]}_test.csv"

# # Creating The Final CSV We Use to Train our Model 
# train_df.to_csv(train_output_path, index=False, encoding="utf-8")

# # Creating The Final CSV We Use to Test our Model 
# test_df.to_csv(test_output_path, index=False, encoding="utf-8")

# print(f"Training DataFrame saved to: {train_output_path}")

# print(f"Testing DataFrame saved to: {test_output_path}")