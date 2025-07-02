import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 6):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        b_z, ts, dim = x.shape
        x = x + self.pe[:, :x.size(1), :]
        return x
    
# =========================================================================== #
class VideoModel(nn.Module):
    def __init__(self, dim, hidden_dim=64, dr_rate=0.6, gamma=0.5, video_length=12):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        
        self.dropout= nn.Dropout(dr_rate)
        self.pos_encoder = PositionalEncoding(dim, max_len=video_length)
        self.rnn = nn.TransformerEncoderLayer(d_model=dim, dim_feedforward=dim, nhead=8, batch_first=True)
        self.fc1 = nn.Linear(video_length * dim, self.dim)
        
    def embed_video(self, x):
        x = self.pos_encoder(x)
        out = self.rnn(x)
        out = self.dropout(out)
        ft = torch.flatten(out, start_dim = 1)
        out = self.fc1(ft)
            
        return out
    
    def forward(self, anchor, pos, neg):
        anchor_emb = self.embed_video(anchor)
        pos_emb = self.embed_video(pos)
        neg_emb = self.embed_video(neg)
        
        pos_dist = torch.sum((anchor_emb - pos_emb) ** 2, 1)
        neg_dist = torch.sum((anchor_emb - neg_emb) ** 2, 1)
        
        loss = pos_dist - neg_dist + self.gamma
        loss[loss < 0] = 0
        loss = torch.sum(loss)

        return loss
    