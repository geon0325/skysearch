import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    def __init__(self, output_dim, num_channel=1, drop_prob=0.1):
        super().__init__()
        fc_input = 139500
            
        self.conv2d = nn.Sequential(
            nn.Conv2d(num_channel, 10, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Dropout2d(drop_prob),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(fc_input, output_dim)
        )
        
    def forward(self, x):
        x = self.conv2d(x)
        return x

class projection_head(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc_layer1 = nn.Linear(input_dim, output_dim, bias=True)
        self.fc_layer2 = nn.Linear(output_dim, output_dim, bias=True)
        
    def forward(self, x):
        x = self.fc_layer1(x)
        x = F.relu(x)
        x = self.fc_layer2(x)
        return x
    
# =========================================================================== #
class model(nn.Module):
    def __init__(self, dim, num_channel=3, gamma=0.5, requires_grad=True):
        super().__init__()
        self.dim = dim
        self.gamma = gamma
        self.proj_dim = dim
        
        self.encoder = CNN(self.dim, num_channel=num_channel)
        self.proj = projection_head(self.proj_dim, self.dim)
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        
    def forward(self, anchor, pos, neg):
        anchor_emb = self.embed_image(anchor)
        pos_emb = self.embed_image(pos)
        neg_emb = self.embed_image(neg)
        
        pos_dist = torch.sum((anchor_emb - pos_emb) ** 2, 1)
        neg_dist = torch.sum((anchor_emb - neg_emb) ** 2, 1)
        
        loss = pos_dist - neg_dist + self.gamma
        loss[loss < 0] = 0
        loss = torch.sum(loss)

        return loss
    
    def embed_image(self, img):
        img = self.encoder(img) # batchsize, self.dim
        emb = self.proj(img.view(-1, self.proj_dim))
        return emb