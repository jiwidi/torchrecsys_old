import torch
import torch.nn.functional as F

from torchrecsys.datasets.utils import dataframe_schema


class InteractionsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        interactions,
        user_features,
        item_features,
        user_id="user_id",
        item_id="item_id",
        interaction_id="interaction",
        sample_negatives=0,
        target_column=None,
    ):

        self.user_id = user_id
        self.item_id = item_id
        self.interaction_id = interaction_id

        ##Check proper dataframe columns order
        ## Call assert subfunction to chekc user is first, item second and interaction third
        ## Assert in both user and item dfs too

        self.interactions = interactions[interactions.columns[:3]]
        self.context_features = interactions[interactions.columns[3:]]

        self.user_features = dict(
            # Eliminate user_id feature that is first on the matrix
            zip(user_features[self.user_id], user_features.values[:, 1:])
        )
        self.item_features = dict(
            zip(item_features[self.item_id], item_features.values[:, 1:])
        )

        if target_column and sample_negatives:
            assert 1 == 0  # Error because logic wont work

        # Create a nice way of loading context + item features into a single dataset. Generate schema that models read from and are able to create
        self.n_users = user_features[self.user_id].max()
        self.n_items = item_features[self.item_id].max()

        self.interactions_pd_schema = dataframe_schema(self.interactions)
        self.context_pd_schema = dataframe_schema(self.context_features)
        self.user_pd_schema = dataframe_schema(user_features.drop(self.user_id, axis=1))
        self.item_pd_schema = dataframe_schema(item_features.drop(self.item_id, axis=1))

        if self.interactions[self.interaction_id].isin([0, 1]).all():
            self.target_type = "binary"
        else:
            self.target_type = "continuous"

        self.interactions = interactions.values
        self.context_features = self.context_features.values
        # To do add custom target column

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        interaction = self.interactions[idx]
        context_features = self.context_features[idx]

        user_features = self.user_features[interaction[0]]
        item_features = self.item_features[interaction[1]]

        return interaction, context_features, user_features, item_features

    def __castdtypes(self, data):
        """Ensure needed dtypes for the data_Schema"""

    @property
    def data_schema(self):
        return {
            "interactions": [self.n_users, self.n_items],
            "context": self.context_pd_schema,
            "user_features": self.user_pd_schema,
            "item_features": self.item_pd_schema,
            "objetive": self.target_type,
        }

    @staticmethod
    def collate_fn(batch_output):
        return batch_output


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe,
        identifier_column=0,
        interaction_column=1,
        sample_negatives=False,
        negatives_per_positive=0,
        max_len=10,
        training=True,
    ):
        self.data = dataframe
        self.identifier_column = identifier_column
        self.interaction_column = interaction_column
        self.max_len = max_len
        self.training = training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx][self.identifier_column]

        sequence = torch.LongTensor(sequence[-self.max_len :])
        sequence = F.pad(sequence, (self.max_len - sequence.shape[0], 0))

        if self.training:
            interaction = self.data[idx][self.interaction_column]
            return torch.LongTensor(sequence), interaction
        else:
            return torch.LongTensor(sequence)
