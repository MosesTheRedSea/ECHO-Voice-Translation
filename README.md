# ECHO
- Voice Translation System

![Logo](https://github.com/MosesTheRedSea/Voice-Based-Text-Language-Translation-System/blob/main/echo.png)
[mattcolewilson](https://dribbble.com/mattcolewilson)

## Project Overview
The **Voice Translation System** is designed to provide accurate and efficient real-time translations of spoken language. By leveraging machine learning, this system can process audio inputs and return translations in multiple languages, catering to a diverse global audience.

The project includes:
- **Preprocessing of audio data**.
- **Training machine learning models** for language translation.
- **Evaluation of model performance** and real-time implementation.

[Voice Translation System](https://mosestheredsea.github.io/Voice-Based-Text-Language-Translation-System/)

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with the project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/Voice-Based-Text-Language-Translation-System
cd Voice-Based-Text-Language-Translation-System
```

## Install the dependencies:
```bash
pip install -r requirements.txt
```

##  Alternatively, Create Enviroment Using Conda 
```bash
conda env create -f environment.yml
conda activate voice-translation
```

## Data

The dataset used for this project includes diverse speech datasets and language corpora. Data is located in the `data/` folder.

- **Raw Data**: Unprocessed speech files.
- **Processed Data**: Preprocessed files used for model training.

You can find information about preprocessing in `src/data_preprocessing.py`.

## Model Training

The ML models used for voice translation are implemented in the `src/models.py`. We have experimented with various machine learning techniques, including:

- **Gated Recurrent Unit - (GRU)**
-  **Long Short Term Memory  - (LSTM)**
- **Transformer Models**
