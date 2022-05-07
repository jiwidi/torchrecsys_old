from typing import List

import torch
import torch.nn.functional as F


##TODO
class feature:
    def __init__(self, name: str, dtype: str, unique_value_count: int) -> None:
        self.name = name
        self.dtype = dtype
        self.unique_value_count = unique_value_count
        self.layer_name = None

    def __str__(self):
        return f"Featue [Name: {self.name}, dtype: {self.dtype}, unique_value_count: {self.unique_value_count}]"


def dataframe_schema(df) -> List[feature]:
    r = []
    for col in df.columns.values:
        col_feature = feature(
            name=col, dtype=df[col].dtype.name, unique_value_count=len(df[col].unique())
        )
        r.append(col_feature)

    return r


class InteractionsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        interactions,
        user_features,
        item_features,
        user_id="user_id",
        item_id="item_id",
        interaction="interaction",
        sample_negatives=0,
        target_column=None,
    ):

        self.user_id = user_id
        self.item_id = item_id
        self.interaction = interaction

        ##Check proper dataframe columns order
        ## Call assert subfunction to chekc user is first, item second and interaction third
        ## Assert in both user and item dfs too

        self.interactions = interactions[interactions.columns[:3]]
        self.context_features = interactions[interactions.columns[3:]]

        self.user_features = dict(
            zip(user_features[self.user_id], user_features.values)
        )
        self.item_features = dict(
            zip(item_features[self.item_id], item_features.values)
        )

        if target_column and sample_negatives:
            assert 1 == 0  # Error because logic wont work

        # Create a nice way of loading context + item features into a single dataset. Generate schema that models read from and are able to create
        self.n_users = user_features[self.user_id].max()
        self.n_items = item_features[self.item_id].max()

        self.interactions_pd_schema = dataframe_schema(self.interactions)
        self.context_pd_schema = dataframe_schema(self.context_features)
        self.item_pd_schema = dataframe_schema(item_features)
        self.user_pd_schema = dataframe_schema(user_features)

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
            "objetive": "notbinary?",
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
