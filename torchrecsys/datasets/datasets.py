import torch

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
