import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.././scripts/preprocessing/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../scripts/models/Test_Text/'))
import torch
import utils
import gru_model
import transformers
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu
from transformers import BertTokenizer
import matplotlib as plt
from gru_model import EncoderGRU, DecoderGRU, Seq2Seq
from utils import TRAINPATH, LANGUAGES 

# Define the dataset class
class TranslationDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        english_sentence = self.data.iloc[idx]['English'].values 
        spanish_sentence = self.data.iloc[idx]['Spanish'].values
        bert_embedding_english = torch.tensor(eval(self.data.iloc[idx]['English BERT']), dtype=torch.float32)
        bert_embedding_spanish = torch.tensor(eval(self.data.iloc[idx]['Spanish BERT']), dtype=torch.float32)
        return english_sentence, spanish_sentence, bert_embedding_english, bert_embedding_spanish

# Instantiate dataset and dataloaders
train_dataset = TranslationDataset(TRAINPATH + f"{LANGUAGES[0]}_{LANGUAGES[1]}_train.csv")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Hyperparameters
hidden_size = 768  # Typically, this should match the BERT output size
num_layers = 3
output_size = len(BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased').get_vocab())
learning_rate = 0.001
num_epochs = 10

# Initialize models, loss function, and optimizer
encoder = EncoderGRU(hidden_size, num_layers)
decoder = DecoderGRU(hidden_size, output_size, num_layers)
model = Seq2Seq(encoder, decoder)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize lists to store metrics
losses, bleu_scores, f1_scores = [], [], []

# Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    all_predictions = []
    all_references = []

    for english_sentences, spanish_sentences, bert_embeddings_english, bert_embeddings_spanish in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(bert_embeddings_english, bert_embeddings_spanish)

        # Calculate loss (considering only the actual target sentences)
        outputs = outputs.view(-1, outputs.size(-1))  # Reshape for loss calculation
        target = spanish_sentences[:, 1:].contiguous().view(-1)  # Skip <start> token
        loss = criterion(outputs, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # Generate predictions
        _, predicted_indices = torch.max(outputs, dim=1)
        predicted_sentences = [BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased').convert_ids_to_tokens(ids.tolist()) for ids in predicted_indices]

        all_predictions.extend(predicted_sentences)
        all_references.extend(spanish_sentences[:, 1:].tolist())  # Store the reference sentences

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)  # Append average loss to the list

    # Calculate BLEU and F1 scores for the epoch
    bleu_score = sum(sentence_bleu([ref], pred) for ref, pred in zip(all_references, all_predictions)) / len(all_predictions)
    bleu_scores.append(bleu_score)  # Append BLEU score to the list

    f1 = f1_score([item for sublist in all_references for item in sublist], 
                   [item for sublist in all_predictions for item in sublist], 
                   average='weighted')
    f1_scores.append(f1)  # Append F1 score to the list

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, BLEU: {bleu_score:.4f}, F1 Score: {f1:.4f}")

# Save the trained model
model_save_path = "model.pt"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(losses, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(bleu_scores, label='BLEU Score', color='orange')
plt.title('BLEU Score over Epochs')
plt.xlabel('Epochs')
plt.ylabel('BLEU Score')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(f1_scores, label='F1 Score', color='green')
plt.title('F1 Score over Epochs')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()