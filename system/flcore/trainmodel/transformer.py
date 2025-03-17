import math
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, nlayers: int, num_classes: int, 
                 dropout: float = 0.1, max_len: int = 200, d_hid: int = 2048):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.hidden_dim = d_model
        self.fc = nn.Linear(d_model, num_classes)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, attn_mask: Tensor = None) -> Tensor:
        if type(src) == type([]):
            src, _ = src
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            attn_mask: Tensor, shape [batch_size, seq_len]

        Returns:
            output Tensor of shape [batch_size, num_classes]
        """
        x = self.embedding(src) * math.sqrt(self.hidden_dim)
        x = self.pos_encoder(x)

        x = self.encoder(x, attn_mask)
        x = x[:, 0] # [CLS] token
        output = self.fc(x)

        return output
