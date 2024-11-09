import torch
import torch.nn as nn

# GRU (Gated Recurrent Unit) Model

# A GRU, or Gated Recurrent Unit, is a type of RNN that uses gates to control the flow of information through the network. 
# This allows it to capture long-term dependencies in sequential data more effectively than traditional RNNs.

"""
Update Gate: Determines how much of the previous hidden state to keep.
Reset Gate: Determines how much of the previous hidden state to forget.
New Hidden State: Calculated based on the input, the previous hidden state, and the gates.
"""

# Steps For Implementing the GRU

# Tokenization: Break down sentences into words or subwords.
# Numericalization: Convert tokens into numerical representations.
# Padding: Make sequences of equal length for efficient batch processing.

# I used the BERT Transformer 
# (Bidirectional Encoder Representations from Transformers (BERT) was developed by Google as a way 
# to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.)

"""
Model Architecture
Encoder-Decoder Architecture:

Encoder:

Input: English sentence (numericalized and padded).
Process: GRU layers process the input sequence, capturing contextual information.
Output: Context vector representing the entire input sequence.
Decoder:

Input: Start token and previous output token (initially a start token).
Process: GRU layers process the input, conditioned on the context vector from the encoder.
Output: Probability distribution over the vocabulary, predicting the next token in the Spanish translation.
Decoding: Use techniques like beam search or greedy search to generate the complete translation.
"""

class EncoderGRU(nn.Module):
    def __init__(self, hidden_size, num_layers=3):
        super(EncoderGRU, self).__init__()
        self.gru = nn.GRU(768, hidden_size, num_layers, batch_first=True)  

    def forward(self, bert_embeddings):
        outputs, hidden = self.gru(bert_embeddings)
        return outputs, hidden

class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=3): 
        super(DecoderGRU, self).__init__()
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        output, hidden = self.gru(input, hidden)
        output = self.fc(output)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        encoder_outputs, hidden = self.encoder(src)
        decoder_input = trg[:, 0].unsqueeze(1) 
        outputs = []

        for t in range(1, trg.size(1)):
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs.append(output)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else output.argmax(1).unsqueeze(1)

        return torch.stack(outputs, dim=1)

# class TranslationGRU(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=1):
#         super(TranslationGRU, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         # Encoder
#         self.encoder_gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         # Decoder
#         self.decoder_gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
        
#     def forward(self, english_embedding, spanish_embedding=None, training=True):
#         # Encoding the English embedding
#         encoder_output, hidden = self.encoder_gru(english_embedding)
#         # Decoding to Spanish
#         decoder_output, _ = self.decoder_gru(hidden)
#         # Output layer to predict each token
#         output = self.fc(decoder_output)
#         return output