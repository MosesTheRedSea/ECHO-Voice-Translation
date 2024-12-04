import torch
import torch.nn as nn

# Encoder (Transformer-based)
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout, max_len=512):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim) # Using embedding_dim
        self.pos = nn.Parameter(torch.zeros(1, max_len, embedding_dim))
        self.layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_layers, dim_feedforward=hidden_dim, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=n_layers)
        self.d = embedding_dim

    def forward(self, src):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d, dtype=torch.float32, device=src.device))
        src = src + self.pos[:, :src.size(1)]
        return self.encoder(src)

# Decoder (Transformer-based)
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout, max_len=512):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, embedding_dim))
        self.layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=n_layers, dim_feedforward=hidden_dim, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.layer, num_layers=n_layers)
        self.fc_out = nn.Linear(embedding_dim, output_dim)
        self.d = embedding_dim

    def forward(self, tgt, m):
        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d, dtype=torch.float32, device=tgt.device))
        tgt = tgt + self.pos[:, :tgt.size(1)]
        mask = torch.triu(torch.ones(tgt.size(0), tgt.size(0), device=tgt.device), diagonal=1).bool()
        decoder = self.decoder(tgt, m, mask)
        return self.fc_out(decoder)

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
        m = self.encoder(src)
        length = trg.size(0)
        batch_size = trg.size(1)
        trg_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(length, batch_size, trg_vocab_size, device=self.device)

        input = trg[0, :] # First target token
        for t in range(1, length):
            output = self.decoder(input.unsqueeze(0), m)
            outputs[t] = output.squeeze(0)

            top1 = output.argmax(2)
            top1 = top1.squeeze(0)
            input = trg[t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs