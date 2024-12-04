import torch
import torch.nn as nn

# Encoder (Transformer-based)
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout, max_len=512):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, embedding_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_layers, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.embedding_dim = embedding_dim

    def forward(self, src):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.embedding_dim, dtype=torch.float32, device=src.device))
        src = src + self.pos_encoder[:, :src.size(1)]
        return self.transformer_encoder(src)

# Decoder (Transformer-based)
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout, max_len=512):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, embedding_dim))
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=n_layers, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(embedding_dim, output_dim)
        self.embedding_dim = embedding_dim

    def forward(self, tgt, memory):
        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(self.embedding_dim, dtype=torch.float32, device=tgt.device))
        tgt = tgt + self.pos_encoder[:, :tgt.size(1)]
        tgt_mask = torch.triu(torch.ones(tgt.size(0), tgt.size(0), device=tgt.device), diagonal=1).bool()
        output = self.transformer_decoder(tgt, memory, tgt_mask)
        return self.fc_out(output)

# Seq2Seq Model (Encoder + Decoder)
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        src = src.to(self.device)
        trg = trg.to(self.device)
        memory = self.encoder(src)
        trg_len = trg.size(0)
        batch_size = trg.size(1)
        trg_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size, device=self.device)

        input = trg[0, :] # First target token
        for t in range(1, trg_len):
            output = self.decoder(input.unsqueeze(0), memory)
            outputs[t] = output.squeeze(0)

            top1 = output.argmax(2).squeeze(0)
            input = trg[t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs