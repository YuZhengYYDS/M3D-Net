"""Clinical encoder preserved from the experimental fusion model.

The historical iTransformer name denotes a sequence-length-one Transformer
encoder in this project; it is not a port of the time-series iTransformer.
"""
import numpy as np
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class iTransformer(nn.Module):
    def __init__(self, input_features=22, seq_length=1, hidden_size=128, num_layers=4, dropout=0.3):
        super().__init__()
        self.input_features = input_features
        self.seq_length = seq_length
        self.hidden_size = hidden_size

        self.input_embedding = nn.Linear(input_features, hidden_size)

        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * seq_length, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, 1000)  
        )

    def forward(self, x):

        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_features]

        embedded = self.input_embedding(x)  # [batch, seq_length, hidden_size]

        embedded = self.pos_encoder(embedded)

        encoded = self.transformer_encoder(embedded)  # [batch, seq_length, hidden_size]

        flattened = encoded.reshape(encoded.size(0), -1)  # [batch, seq_length * hidden_size]

        output = self.output_layer(flattened)  # [batch, 1000]
        return output
