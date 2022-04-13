import torch
import torch.nn.functional as F
import numpy as np

class InteractionsDataset(torch.utils.data.Dataset):
    def __init__(self, interactions, item_features, user_features, sample_negatives=0, target_column=None):
        self.interactions = interactions.values
        self.item_features = dict(zip(item_features.item, item_features.values))
        self.user_features = dict(zip(user_features.user, user_features.values))
        
        if target_column and sample_negatives:
            assert 1==0 #Error because logic wont work
            
        #Create a nice way of loading context + item features into a single dataset. Generate schema that models read from and are able to create 
        self.n_users = len(user_features)
        self.n_items = len(item_features)
        
        #Get ids 
    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        interaction = self.interactions[idx]
        user_features = self.item_features[interaction[0]]
        item_features = self.item_features[interaction[1]]
        
        return interaction
    
    def data_schema(self):
        return {}

    
    
    
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, identifier_column=0, interaction_column=1, sample_negatives = False, negatives_per_positive=0, max_len=10, training=True):
        self.data = dataframe
        self.identifier_column = identifier_column
        self.interaction_column = interaction_column
        self.max_len = max_len
        self.training = training
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx][self.identifier_column]
        
        sequence = torch.LongTensor(sequence[-self.max_len:])
        sequence = F.pad(sequence, (self.max_len - sequence.shape[0],0))
        
        if self.training:
            interaction = self.data[idx][self.interaction_column]
            return torch.LongTensor(sequence), interaction
        else:
            return torch.LongTensor(sequence)
