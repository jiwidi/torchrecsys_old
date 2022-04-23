import torch
import pytorch_lightning as pl
from torchrecsys.datasets import InteractionsDataset
from torchrecsys.models import BaseModel
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn

class NeuralCF(BaseModel):
    def __init__(self, data_schema, lr_rate=0.01, embedding_size=64, layers=[512], ):
        super().__init__()
        interactions_schema = data_schema["interactions"]
        
        self.n_users = interactions_schema[0]
        self.n_items = interactions_schema[1]
        
        self.user_embedding = nn.Embedding(self.n_users+1, embedding_size)
        self.item_embedding = nn.Embedding(self.n_items+1, embedding_size)
        
        
        
        self.linear = nn.Linear(embedding_size, layers[0])
        
        self.final_linear = nn.Linear(layers[0], 1)
        self.final_activation = torch.sigmoid if data_schema["objetive"]=="binary" else lambda x: x
        
        
        
        self.criterion = torch.nn.BCELoss() if data_schema["objetive"]=="binary" else torch.nn.MSELoss()
        
        self.lr_rate = lr_rate

    def forward(self, interactions):
        
        user = self.user_embedding(interactions[:, 0].long())
        item = self.item_embedding(interactions[:, 1].long())
        
        x = (user * item)
        x = self.linear(x)
        x = self.final_linear(x)
        x = self.final_activation(x)
        

        return x
    
    def training_step(self, batch):
        interactions, context, items, users = batch
        
        yhat = self(interactions).float()
        yhat = torch.squeeze(yhat)
        
        ytrue = interactions[:, 2].float()
        
        loss = self.criterion(yhat, ytrue)
        
        self.log("train/step_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
    
        return loss
        
        
    def validation_step(self, batch):
        interactions, context, items, users = batch
        
        yhat = self(interactions)
        ytrue = torch.squeeze(interactions[:, 2])
        loss = self.criterion(yhat, ytrue)
        
        self.log("validation/step_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.lr_rate)
        
        return optimizer
    
 