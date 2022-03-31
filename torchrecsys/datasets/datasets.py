import torch
import torch.nn.functional as F

class InteractionsDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, identifier_index=0, interaction_index=1, sample_negatives = False, negatives_per_positive=0):
        self.data = dataframe
        self.identifier_index = identifier_index
        self.interaction_index = interaction_index
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        identifier = self.data[idx][self.identifier_index]
        interaction = self.data[idx][self.interaction_index]
        
        return identifier, interaction
    
    
    
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, identifier_index=0, interaction_index=1, sample_negatives = False, negatives_per_positive=0, max_len=10, training=True):
        self.data = dataframe
        self.identifier_index = identifier_index
        self.interaction_index = interaction_index
        self.max_len = max_len
        self.training = training
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx][self.identifier_index]
        
        sequence = torch.LongTensor(sequence[-self.max_len:])
        sequence = F.pad(sequence, (self.max_len - sequence.shape[0],0))
        
        if self.training:
            interaction = self.data[idx][self.interaction_index]
            return torch.LongTensor(sequence), interaction
        else:
            return torch.LongTensor(sequence)
