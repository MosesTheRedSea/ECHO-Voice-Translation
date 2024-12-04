import torch
import torch.nn as nn
import random  # Import random module for teacher forcing

import torch
import torch.nn as nn
import torch.optim as optim

# Encoder (GRU-based)
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)  # Using embedding_dim
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

# Decoder (GRU-based)
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # Convert input token indices to embeddings
        embedded = self.embedding(input).unsqueeze(0)  # (1, batch_size, embedding_dim)
        embedded = self.dropout(embedded)

        # Adjust hidden dimensions for single-batch case
        if hidden.dim() == 3 and hidden.size(1) == 1:
            hidden = hidden.squeeze(1)  # Squeeze batch dimension for unbatched case

        # Pass through GRU
        output, hidden = self.gru(embedded, hidden.unsqueeze(1) if hidden.dim() == 2 else hidden)

        # Generate predictions
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden.squeeze(1) if hidden.dim() == 3 else hidden



# Seq2Seq Model (Encoder + Decoder)
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        _, hidden = self.encoder(src)  # Ensure only hidden is passed

        input = trg[0, :]  # First target token
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output

            top1 = output.argmax(1)
            input = trg[t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs

# class GRUTranslationModel(nn.Module):
#     def __init__(self, hidden_dim, num_layers, output_size, dropout=0.1):
#         super(GRUTranslationModel, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.output_size = output_size
        
#         # Projection layer to match the GRU input size
#         self.embedding_projection = nn.Linear(768, 512)  # Assuming BERT embeddings of size 768
        
#         # Encoder GRU
#         self.encoder_gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
#         # Decoder GRU
#         self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
#         # Output linear layer
#         self.fc_out = nn.Linear(hidden_dim, output_size)

#         # Dropout layer
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, src, trg, teacher_forcing_ratio=0.5):
#         # Ensure src is float before passing it to the projection layer
#         src = src.float()  # Convert to float32
        
#         # Project the BERT embeddings to the correct size for the GRU
#         src = self.embedding_projection(src)  # Projecting to 512 dimensions
        
#         # Encoder
#         _, hidden = self.encoder_gru(src)  # We discard the outputs of the GRU

#         # Prepare the initial decoder input (usually <SOS> token, now an embedding)
#         input = trg[:, 0].unsqueeze(1)  # First token in target sequence
#         input = input.float()  # Convert to float32 if needed
        
#         # Decoder
#         outputs = []
#         for t in range(1, trg.size(1)):  # Start from the second token
#             output, hidden = self.decoder_gru(input, hidden)  # GRU forward pass

#             output = self.fc_out(output)  # Linear layer to get prediction for the token
#             outputs.append(output)
            
#             # Teacher forcing: use the true token as the next input (or predicted token)
#             teacher_force = random.random() < teacher_forcing_ratio  # Use random to decide if we use teacher forcing
#             top1 = output.argmax(2)  # Get the index of the highest probability token

#             input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)  # Teacher forcing or prediction

#         # Concatenate all the output predictions along the sequence dimension
#         outputs = torch.cat(outputs, dim=1)
#         return outputs
