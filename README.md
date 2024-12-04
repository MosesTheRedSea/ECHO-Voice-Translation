# ECHO

![Logo](https://github.com/MosesTheRedSea/Voice-Based-Text-Language-Translation-System/blob/main/echo.png)
[mattcolewilson](https://dribbble.com/mattcolewilson)

## Project Overview
The **Voice Translation System** is designed to provide accurate and efficient real-time translations of spoken language. By leveraging machine learning, this system can process audio inputs and return translations in multiple languages, catering to a diverse global audience.

The project includes:
- **Preprocessing of audio data**.
- **Training machine learning models** for language translation.
- **Evaluation of model performance** and real-time implementation.

[Voice Translation System](https://mosestheredsea.github.io/ECHO-Voice-Translation/)

# Project Name

This repository contains the code and resources for the English-to-Spanish translation project. The repository includes directories for data, models, scripts for preprocessing and training, and necessary requirements to run the project.

## Directory Structure

### /data/raw/English
This directory contains the English-to-Spanish dataset in `.txt` format.

- `/data/raw/English`: Holds the raw English-to-Spanish text dataset.

### /models/Tet_Text
This directory includes the models for translation, including different architectures like GRU, LSTM, and Transformer.

- `/models/Tet_Text/gru_model.py`: Contains the code for the GRU-based translation model.
- `/models/Tet_Text/lst_model.py`: Contains the code for the LSTM-based translation model.
- `/models/Tet_Text/transformer_model.py`: Contains the code for the Transformer-based translation model.

### /scripts/preprocessing
This directory includes the necessary preprocessing code for preparing the dataset before training.

- `/scripts/preprocessing/data_processing.py`: Contains the functions for processing the raw data, such as tokenization, padding, etc.
- `/scripts/preprocessing/utils.py`: Contains utility functions used for preprocessing, like vocabulary creation or text cleaning.

### /scripts/training
This directory holds the training scripts for training the model.

- `/scripts/training/process.py`: Contains the functions for loading the data, preparing the dataset, and handling the training loop.
- `/scripts/training/train.py`: Main script for training the translation model.

### /requirements.txt
This file contains the list of required Python packages needed to run the project.

- `/requirements.txt`: A text file listing the Python dependencies required for the project.
