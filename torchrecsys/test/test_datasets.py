from torchrecsys.datasets import InteractionsDataset
from torchrecsys.models import NCF
from torchrecsys.test.fixtures import (
    dummy_interaction_dataset,
    dummy_interactions,
    dummy_item_features,
    dummy_user_features,
)  # NOQA
from torch.utils.data import DataLoader


def test_interactions_dataset(dummy_interaction_dataset):
    item = dummy_interaction_dataset[0]
    dataloader = DataLoader(dummy_interaction_dataset)
    item = next(iter(dataloader))


def test_sequence_dataset(dummy_interaction_dataset):
    pass

