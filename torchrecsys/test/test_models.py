import numpy as np
import pytorch_lightning as pl
import torch

from torchrecsys.datasets import InteractionsDataset
from torchrecsys.models import matrixFactorizationModel, popularityModel


def test_popularityModel():
    itd = np.random.rand(64, 2)
    itd = InteractionsDataset(itd)
    train_dataloader = torch.utils.data.DataLoader(itd, batch_size=32)
    model = popularityModel(itd)
    # training
    trainer = pl.Trainer()
    trainer.fit(model, train_dataloader)


def test_matrixFactorizationModel():
    itd = np.random.rand(3, 2)
    itd = InteractionsDataset(itd)

    matrixFactorizationModel(itd)
