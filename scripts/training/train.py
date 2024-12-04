import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '/storage/ice1/2/6/madewolu9/Voice-Based-Language-Translation-System/scripts/preprocessing/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '/storage/ice1/2/6/madewolu9/Voice-Based-Language-Translation-System/models/Text_Text/'))
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from gru_model import Encoder, Decoder, Seq2Seq
#from lstm_model import Encoder, Decoder, Seq2Seq uncomment to use lstm model, comment other models
#from transformer_model import Encoder, Decoder, Seq2Seq

from utils import TRAINPATH, LANGUAGES
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu

# Set device for training (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset preparation (assuming you have a DataFrame with the required data)
class TranslationDataset(Dataset):
    def __init__(self, data_frame, tokenizer, max_len=512):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        english_text = str(self.data_frame.iloc[idx, 0])
        spanish_text = str(self.data_frame.iloc[idx, 2])

        # Tokenize English and Spanish sentences
        eng_input = self.tokenizer.encode(english_text, max_length=self.max_len, truncation=True, padding='max_length')
        spa_target = self.tokenizer.encode(spanish_text, max_length=self.max_len, truncation=True, padding='max_length')

        # Convert to tensor
        eng_tensor = torch.tensor(eng_input, dtype=torch.long)
        spa_tensor = torch.tensor(spa_target, dtype=torch.long)

        return eng_tensor, spa_tensor

# Load tokenizer and data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Use a BERT tokenizer

# Load your training data from the specified path
train_data_path = "../../data/processed/English/Spanish/Train/English_Spanish_train.csv"
train_data = pd.read_csv(train_data_path)  # Load the dataset

#train_data_sampled = train_data.sample(frac=0.10, random_state=200) 
#train_dataset = TranslationDataset(train_data_sampled, tokenizer, max_len=256) //if you want to sample a 10 percent of data

# Prepare Dataset and DataLoader
train_dataset = TranslationDataset(train_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)

# Model setup
embedding_dim = 512
hidden_dim = 512
output_dim = len(tokenizer)
input_dim = len(tokenizer.vocab)
n_layers = 2
dropout = 0.5

encoder = Encoder(input_dim=input_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
decoder = Decoder(output_dim=output_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)

model = Seq2Seq(encoder, decoder, device).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Evaluation metrics setup
def calculate_bleu(reference, hypothesis):
    return sentence_bleu([reference], hypothesis)

def calculate_f1(reference, hypothesis):
    return f1_score(reference, hypothesis, average='macro')

# Checkpoint save and load functions
checkpoint_path = '/storage/ice1/2/6/madewolu9/Voice-Based-Language-Translation-System/models/Text_Text/GRU/Checkpoints/'

def save_checkpoint(epoch, model, optimizer, train_losses, bleu_scores, f1_scores):
    checkpoint_filename = f'checkpoint_epoch_{epoch}.pt'  # You can customize the filename as needed
    checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_filename)  # Combine directory and filename
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'bleu_scores': bleu_scores,
        'f1_scores': f1_scores
    }, checkpoint_filepath)
    print(f"Checkpoint saved at epoch {epoch}.")

def load_checkpoint():
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        bleu_scores = checkpoint['bleu_scores']
        f1_scores = checkpoint['f1_scores']
        print(f"Checkpoint loaded from epoch {start_epoch}.")
        return start_epoch, train_losses, bleu_scores, f1_scores
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0, [], [], []

# Load checkpoint if available
start_epoch, train_losses, bleu_scores, f1_scores = load_checkpoint()

# Training loop
epochs = 10
for epoch in range(start_epoch, epochs):

    '''
    If the sample is changed every 3 epochs
    if epoch % 3 == 0: 
        train_data_sampled = train_data.sample(frac=0.10, random_state=epoch) 
        train_dataset = TranslationDataset(train_data_sampled, tokenizer, max_len=256)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    '''

    print("Start EPOCH")
    model.train()
    total_loss = 0
    epoch_bleu = 0
    epoch_f1 = 0

    #num_batches = len(train_loader) saves the number of batches
    for batch_idx, (eng_input, spa_target) in enumerate(train_loader):
        eng_input = eng_input.to(device)
        spa_target = spa_target.to(device)

        optimizer.zero_grad()
        output = model(eng_input, spa_target)
        output = output.view(-1, output_dim)
        spa_target = spa_target.view(-1)

        loss = criterion(output, spa_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            hypothesis = output.argmax(dim=1).cpu().numpy()
            reference = spa_target.cpu().numpy()
            bleu = calculate_bleu(reference, hypothesis)
            f1 = calculate_f1(reference, hypothesis)
            epoch_bleu += bleu
            epoch_f1 += f1
    '''
    averages results + saves them
    avg_loss = total_loss / num_batches
    avg_bleu = epoch_bleu / num_batches
    avg_f1 = epoch_f1 / num_batches

    train_losses.append(avg_loss)
    bleu_scores.append(avg_bleu)
    f1_scores.append(avg_f1)
    '''
    
    # Save checkpoint at the end of each epoch
    print("EPOC FINISHED")
    save_checkpoint(epoch, model, optimizer, train_losses, bleu_scores, f1_scores)

# Plot metrics
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(bleu_scores)), bleu_scores, label='BLEU Score', color='blue')
plt.plot(range(len(f1_scores)), f1_scores, label='F1 Score', color='red')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.title('BLEU & F1 Scores over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# OLD CODE THAT WORKS

# import torch
# import pandas as pd
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from transformers import BertTokenizer
# import matplotlib.pyplot as plt
# from gru_model import Encoder, Decoder, Seq2Seq
# from utils import TRAINPATH, LANGUAGES
# from sklearn.metrics import f1_score
# from nltk.translate.bleu_score import sentence_bleu

# # Set device for training (GPU if available, else CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Dataset preparation (assuming you have a DataFrame with the required data)
# class TranslationDataset(Dataset):
#     def __init__(self, data_frame, tokenizer, max_len=512):
#         self.data_frame = data_frame
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.data_frame)

#     def __getitem__(self, idx):
#         english_text = str(self.data_frame.iloc[idx, 0])
#         spanish_text = str(self.data_frame.iloc[idx, 2])

#         # Tokenize English and Spanish sentences
#         eng_input = self.tokenizer.encode(english_text, max_length=self.max_len, truncation=True, padding='max_length')
#         spa_target = self.tokenizer.encode(spanish_text, max_length=self.max_len, truncation=True, padding='max_length')

#         # Convert to tensor
#         eng_tensor = torch.tensor(eng_input, dtype=torch.long)
#         spa_tensor = torch.tensor(spa_target, dtype=torch.long)

#         return eng_tensor, spa_tensor

# # Load tokenizer and data
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Use a BERT tokenizer

# # Load your training data from the specified path
# train_data_path = "../../data/processed/English/Spanish/Train/English_Spanish_train.csv"
# train_data = pd.read_csv(train_data_path)  # Load the dataset

# # Prepare Dataset and DataLoader
# train_dataset = TranslationDataset(train_data, tokenizer)
# train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

# # Model setup
# embedding_dim = 512  # Embedding dimension
# hidden_dim = 512     # Hidden layer dimension (same as embedding size)
# output_dim = len(tokenizer)  # Output size should be vocab size
# input_dim = len(tokenizer.vocab)
# n_layers = 2         # Number of layers in GRU
# dropout = 0.5        # Dropout to avoid overfitting

# encoder = Encoder(input_dim=input_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
# decoder = Decoder(output_dim=output_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)

# model = Seq2Seq(encoder, decoder, device).to(device)

# # Loss function and optimizer
# criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # Ignore padding tokens
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Evaluation metrics setup
# def calculate_bleu(reference, hypothesis):
#     return sentence_bleu([reference], hypothesis)

# def calculate_f1(reference, hypothesis):
#     return f1_score(reference, hypothesis, average='macro')

# # Training loop
# epochs = 10
# train_losses = []
# bleu_scores = []
# f1_scores = []

# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     epoch_bleu = 0
#     epoch_f1 = 0

#     for batch_idx, (eng_input, spa_target) in enumerate(train_loader):
#         eng_input = eng_input.to(device)
#         spa_target = spa_target.to(device)

#         optimizer.zero_grad()

#         # Forward pass
#         output = model(eng_input, spa_target) # This is the issue ERROR LINE

#         # Reshape output and target for loss calculation
#         output = output.view(-1, output_dim)
#         spa_target = spa_target.view(-1)

#         # Calculate loss
#         loss = criterion(output, spa_target)
#         loss.backward()

#         # Gradient descent step
#         optimizer.step()

#         total_loss += loss.item()

#         # Calculate BLEU score and F1 score for the batch
#         with torch.no_grad():
#             hypothesis = output.argmax(dim=1).cpu().numpy()  # Predicted words
#             reference = spa_target.cpu().numpy()  # True target words

#             # Calculate BLEU and F1 score for each sentence in the batch
#             bleu = calculate_bleu(reference, hypothesis)
#             f1 = calculate_f1(reference, hypothesis)

#             epoch_bleu += bleu
#             epoch_f1 += f1

#     avg_loss = total_loss / len(train_loader)
#     avg_bleu = epoch_bleu / len(train_loader)
#     avg_f1 = epoch_f1 / len(train_loader)

#     print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, BLEU: {avg_bleu:.4f}, F1 Score: {avg_f1:.4f}')
    
#     train_losses.append(avg_loss)
#     bleu_scores.append(avg_bleu)
#     f1_scores.append(avg_f1)

# # Save the trained model
# torch.save(model.state_dict(), 'gru_translation_model.pth')

# # Plotting the metrics
# plt.figure(figsize=(10,5))

# # Loss plot
# plt.subplot(1, 2, 1)
# plt.plot(range(epochs), train_losses, label='Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss over Epochs')
# plt.legend()

# # BLEU & F1 Score plot
# plt.subplot(1, 2, 2)
# plt.plot(range(epochs), bleu_scores, label='BLEU Score', color='blue')
# plt.plot(range(epochs), f1_scores, label='F1 Score', color='red')
# plt.xlabel('Epochs')
# plt.ylabel('Score')
# plt.title('BLEU & F1 Scores over Epochs')
# plt.legend()

# plt.tight_layout()
# plt.show()




# DO NOT USE THSI CODE BELOW

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Define the dataset class
# class TranslationDataset(Dataset):
#     def __init__(self, csv_file, tokenizer, max_length=512):
#         self.data = pd.read_csv(csv_file)
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         english_bert_str = self.data.iloc[idx]['English BERT'].strip('[]')
#         spanish_bert_str = self.data.iloc[idx]['Spanish BERT'].strip('[]')
        
#         # Convert space-separated BERT embeddings from string to list of floats
#         bert_embedding_english = [float(x) for x in english_bert_str.split()]
#         bert_embedding_spanish = [float(x) for x in spanish_bert_str.split()]

#         # Tokenize the English and Spanish sentences with padding and truncation
#         english_tokens = self.tokenizer.encode(english_bert_str, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)
#         spanish_tokens = self.tokenizer.encode(spanish_bert_str, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)
        
#         # Convert tokenized sentences into tensors
#         english_tensor = torch.tensor(english_tokens, dtype=torch.long)
#         spanish_tensor = torch.tensor(spanish_tokens, dtype=torch.long)

#         return english_tensor, spanish_tensor


# # Tokenizer for Spanish BERT
# tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')

# # Hyperparameters
# embedding_dim = 768  # BERT output size
# hidden_dim = 512  # Size of hidden state in GRU
# num_layers = 3
# output_size = len(tokenizer.get_vocab())  # size of the Spanish vocabulary
# learning_rate = 0.001
# num_epochs = 10
# teacher_forcing_ratio = 0.5

# # Initialize the model
# input_dim = len(tokenizer.get_vocab())  # Vocabulary size of the input language (English)
# output_size = len(tokenizer.get_vocab())  # Vocabulary size of the target language (Spanish)
# dropout = 0.1  # Add dropout as it is part of the model
# model = GRUTranslationModel(hidden_dim, num_layers, output_size, dropout).to(device)

# criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Load data
# train_dataset = TranslationDataset("../../data/processed/English/Spanish/Train/English_Spanish_train.csv", tokenizer)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# # Metrics tracking
# losses, bleu_scores, f1_scores = [], [], []

# # Training loop
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     all_predictions = []
#     all_references = []
    
#     for bert_embeddings_english, bert_embeddings_spanish in train_loader:
#         bert_embeddings_english = bert_embeddings_english.to(device)  # Tensor of token indices
#         bert_embeddings_spanish = bert_embeddings_spanish.to(device)

#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(bert_embeddings_english, bert_embeddings_spanish, teacher_forcing_ratio)

#         # Reshape outputs and target for loss calculation
#         outputs = outputs.view(-1, outputs.size(-1))  # Flatten the output
#         target = bert_embeddings_spanish[:, 1:].contiguous().view(-1)  # Ignore <start> token

#         # Calculate the loss
#         loss = criterion(outputs, target)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#         # Generate predictions
#         _, predicted_indices = torch.max(outputs, dim=1)
#         predicted_sentences = [tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in predicted_indices]

#         all_predictions.extend(predicted_sentences)
#         all_references.extend(bert_embeddings_spanish[:, 1:].tolist())  # Store reference sentences

#     avg_loss = total_loss / len(train_loader)
#     losses.append(avg_loss)

#     # Calculate BLEU and F1 scores for the epoch
#     bleu_score = sum(sentence_bleu([ref], pred) for ref, pred in zip(all_references, all_predictions)) / len(all_predictions)
#     bleu_scores.append(bleu_score)

#     f1 = f1_score([item for sublist in all_references for item in sublist], 
#                    [item for sublist in all_predictions for item in sublist], 
#                    average='weighted')
#     f1_scores.append(f1)

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, BLEU: {bleu_score:.4f}, F1 Score: {f1:.4f}")

# # Save the trained model
# model_save_path = '/storage/ice1/2/6/madewolu9/Voice-Based-Language-Translation-System/models/Text_Text/GRU/model.pth'
# torch.save(model.state_dict(), model_save_path)
# print(f"Model saved to {model_save_path}")

# # Plot metrics
# plt.figure(figsize=(12, 5))

# # Plot loss
# plt.subplot(1, 3, 1)
# plt.plot(losses, label='Loss')
# plt.title('Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# # Plot BLEU Score
# plt.subplot(1, 3, 2)
# plt.plot(bleu_scores, label='BLEU Score', color='orange')
# plt.title('BLEU Score over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('BLEU Score')
# plt.legend()

# # Plot F1 Score
# plt.subplot(1, 3, 3)
# plt.plot(f1_scores, label='F1 Score', color='green')
# plt.title('F1 Score over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('F1 Score')
# plt.legend()

# plt.tight_layout()
# plt.show()