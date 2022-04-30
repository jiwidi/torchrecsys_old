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


def dataframe_schema(df) -> List[feature]:
    r = []
    for col in df.columns.values:
        col_feature = feature(
            name=col, dtype=df[col].dtype, unique_value_count=len(df[col].unique())
        )
        r.append(col_feature)

    return r


class InteractionsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        interactions,
        user_features,
        item_features,
        sample_negatives=0,
        target_column=None,
    ):

        if not target_column:
            interactions = interactions[interactions.columns[:2]]
            context_features = interactions[interactions.columns[2:]]
        else:
            interactions = interactions[interactions.columns[:3]]
            context_features = interactions[interactions.columns[3:]]

        self.interactions = interactions.values  # First two columns: userid, itemid
        self.context_features = context_features.values

        self.user_features = dict(zip(user_features.user, user_features.values))
        self.item_features = dict(zip(item_features.item, item_features.values))

        if target_column and sample_negatives:
            assert 1 == 0  # Error because logic wont work

        # Create a nice way of loading context + item features into a single dataset. Generate schema that models read from and are able to create
        self.n_users = user_features.user.max()
        self.n_items = item_features.item.max()

        self.interactions_pd_schema = dataframe_schema(interactions)
        self.context_pd_schema = dataframe_schema(context_features)
        self.item_pd_schema = dataframe_schema(item_features)
        self.user_pd_schema = dataframe_schema(user_features)

        self.target_column = target_column

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
