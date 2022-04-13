import torch
import pytorch_lightning as pl
from torchrecsys.datasets import InteractionsDataset
from abc import ABC, abstractmethod
import numpy as np
import torch

class BaseModel(pl.LightningModule, ABC):
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("`forward` method must be implemented by the user")

    def training_step(self, batch, batch_idx):
        x = self.forward(batch)
        loss = self.task(*x)
        
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = self.forward(batch)
        loss = self.task(*x)
        
        self.log("val_loss", loss)
        return loss
    
        

    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer

    def compile(self,):
        pass
    
    @abstractmethod
    def get_n_recommendation_batch(self, query_vectors, n, params):
        """
        Recommendation for batch users in list.
        """
        pass
    
    
    
    
class popularityModel(BaseModel):
    """
    Model recommend top-n items by global popularity.
    """
    def __init__(self, data: InteractionsDataset, set_popularity=False):
        super().__init__()
        data = data[:,data.interaction_index]
        u, count = np.unique(data, return_counts=True)
        count_sort_ind = np.argsort(-count)
        u[count_sort_ind]
    
    def get_n_recommendation_batch(self, query_vectors, n, params):
        """
        Recommendation for batch users in list.
        """
        pass
    
    def forward(self):
        pass
    
    def get_n_recommendation_batch(self):
        pass
    
    
class matrixFactorizationModel(BaseModel):
    def __init__(self, data: InteractionsDataset, n_factors=128):
        super().__init__()
        n_users = data.n_items
        n_items = data.n_users
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.user_biases = torch.nn.Embedding(n_users, 1)
        self.item_biases = torch.nn.Embedding(n_items,1)
        torch.nn.init.xavier_uniform_(self.user_factors.weight)
        torch.nn.init.xavier_uniform_(self.item_factors.weight)
        self.user_biases.weight.data.fill_(0.)
        self.item_biases.weight.data.fill_(0.)
        
    def forward(self, user, item):
        pred = self.user_biases(user) + self.item_biases(item)
        pred += (self.user_factors(user) * self.item_factors(item)).sum(1, keepdim=True)
        return pred.squeeze()

    def get_n_recommendation_batch(self):
        pass