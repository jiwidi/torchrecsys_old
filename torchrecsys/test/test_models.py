import torch
from pytorch_lightning import Trainer

from torchrecsys.models import NCF
from torchrecsys.test.fixtures import (  # NOQA
    dummy_interaction_dataset,
    dummy_interactions,
    dummy_item_features,
    dummy_user_features,
)


def test_ncf(dummy_interaction_dataset):
    print(dummy_interaction_dataset.item_features)
    dataloader = torch.utils.data.DataLoader(dummy_interaction_dataset, batch_size=2)
    model = NCF(dummy_interaction_dataset.data_schema)
    trainer = Trainer(max_epochs=1)
    trainer.fit(model, dataloader)

    pair = torch.tensor([[1, 2]])
    context = torch.tensor([])
    user = torch.tensor([[0, 1, 0, 1]])
    item = torch.tensor([[0, 0]])

    model(pair, context, user, item)
