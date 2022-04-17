import torch
import pytorch_lightning as pl
from torchrecsys.datasets import InteractionsDataset
from torchrecsys.models import BaseModel
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn

class NeuralCF(BaseModel):
    def __init__(self, data_schema):
        super().__init__()
        interactions_schema = data_schema["interactions"]
        
        n_users = interactions_schema[0][1]
        n_items = interactions_schema[1][1]
        self.user_embedding = nn.Embedding(n_users, 64)
        self.item_embedding = nn.Embedding(n_users, 64)
        
        self.linear = nn.Linear(64, 128)
        self.final_linear = nn.Linear(128, 1)

    def forward(self, x):
        interactions, context, items, users = x
        
        user = self.user_embedding(interactions[:, 0])
        item = self.user_embedding(interactions[:, 1])
        x = user * item
        x = self.linear(x)
        x = self.final_linear(x)
        x = torch.sigmoid(x)
        
        return x
    
 