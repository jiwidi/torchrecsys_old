import torch
import pytorch_lightning as pl
from torchrecsys.datasets import InteractionsDataset
from torchrecsys.models import BaseModel
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn


class NeuralCF(BaseModel):
    def __init__(
        self,
        data_schema,
        lr_rate=0.01,
        embedding_size=64,
        feature_embedding_size=8,
        layers=[512],
    ):
        super().__init__()
        interactions_schema = data_schema["interactions"]

        self.n_users = interactions_schema[0]
        self.n_items = interactions_schema[1]

        self.user_embedding = nn.Embedding(self.n_users + 1, embedding_size)
        self.item_embedding = nn.Embedding(self.n_items + 1, embedding_size)

        self.linear = nn.Linear(embedding_size, layers[0])

        self.final_linear = nn.Linear(layers[0], 1)

        # User features encoding

        # Item features encoding
        self.item_feature_types = []
        for feature_idx, feature in enumerate(data_schema["item_features"]):
            feature_type, value_count = feature
            if feature_type == "category":
                layer_name = f"feature{feature_idx}_embedding"
                setattr(
                    self,
                    layer_name,
                    nn.Embedding(value_count + 1, feature_embedding_size),
                )
                self.item_feature_types.append([feature[0], layer_name, feature_idx])

        self.criterion = (
            torch.nn.BCEWithLogitsLoss()
            if data_schema["objetive"] == "binary"
            else torch.nn.MSELoss()
        )

        self.lr_rate = lr_rate

    def forward(self, interactions, context, users, items):

        user = self.user_embedding(interactions[:, 0].long())
        item = self.item_embedding(interactions[:, 1].long())

        aux = self.encode_item(items)
        x = user * item
        x = self.linear(x)
        x = self.final_linear(x)

        return x

    def encode_user(self, user):
        return user

    def encode_item(self, items):
        r = []
        for idx, feature in enumerate(self.item_feature_types):
            feature_type, layer_name, feature_idx = feature
            if feature_type == "category":
                feature_embedding = getattr(self, layer_name)(items[:, feature_idx])
                r.append(feature_embedding)
        r = torch.cat(r)
        return r

    def training_step(self, batch):
        yhat = self(*batch).float()
        yhat = torch.squeeze(yhat)

        ytrue = batch[0][:, 2].float()

        loss = self.criterion(yhat, ytrue)

        self.log("train/step_loss", loss, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def validation_step(self, batch):
        yhat = self(*batch).float()
        ytrue = batch[0][:, 2].float()
        loss = self.criterion(yhat, ytrue)

        self.log(
            "validation/step_loss", loss, on_step=True, on_epoch=False, prog_bar=False
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.lr_rate)

        return optimizer

