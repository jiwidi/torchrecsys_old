from typing import List

import torch
from torch import nn

from torchrecsys.layers import CategoricalLayer, NumericalLayer
from torchrecsys.models.base import BaseModel


class NCF(BaseModel):
    def __init__(
        self,
        data_schema,
        lr_rate: float = 0.01,
        embedding_size: int = 64,
        feature_embedding_size: int = 8,
        mlp_layers: List[int] = [512, 256],
    ):
        super().__init__()
        interactions_schema = data_schema["interactions"]

        # Ad 1 to the ids and use the latest id as default id for unseen, #TODO
        self.n_users = interactions_schema[0]
        self.n_items = interactions_schema[1]

        # User features encoding
        self.user_features = nn.ModuleList()
        self.user_feature_dimension = 0

        ##Make feature encoding a function and move it to base
        for feature_idx, feature in enumerate(data_schema["user_features"]):
            if feature.dtype == "category":
                layer_name = f"user_{feature.name}_embedding"
                f_layer = CategoricalLayer(
                    name=layer_name,
                    n_unique_values=feature.unique_value_count,
                    dimensions=feature_embedding_size,
                    idx=feature_idx,
                )
                self.user_features.append(f_layer)
                self.user_feature_dimension += feature_embedding_size
            elif feature.dtype == "int64":
                layer_name = f"user_{feature.name}_numerical"
                f_layer = NumericalLayer(
                    name=layer_name,
                    idx=feature_idx,
                )
                self.user_features.append(f_layer)
                self.user_feature_dimension += 1

        # Item features encoding
        self.item_features = nn.ModuleList()
        self.item_feature_dimension = 0
        for feature_idx, feature in enumerate(data_schema["item_features"]):
            if feature.dtype == "category":
                layer_name = f"item_{feature.name}_embedding"
                f_layer = CategoricalLayer(
                    name=layer_name,
                    n_unique_values=feature.unique_value_count,
                    dimensions=feature_embedding_size,
                    idx=feature_idx,
                )
                self.item_features.append(f_layer)
                self.item_feature_dimension += feature_embedding_size
            elif feature.dtype == "int64":
                layer_name = f"item_{feature.name}_numerical"
                f_layer = NumericalLayer(
                    name=layer_name,
                    idx=feature_idx,
                )
                self.item_features.append(f_layer)
                self.item_feature_dimension += 1

        aux_user_id_dimensions = self.user_feature_dimension + embedding_size
        aux_item_id_dimensions = self.item_feature_dimension + embedding_size

        max_dimension = max(aux_user_id_dimensions, aux_user_id_dimensions)
        user_id_dimensions = embedding_size + (max_dimension - aux_user_id_dimensions)
        item_id_dimensions = embedding_size + (max_dimension - aux_item_id_dimensions)

        self.user_embedding = nn.Embedding(self.n_users, user_id_dimensions)
        self.item_embedding = nn.Embedding(self.n_items, item_id_dimensions)

        mlp_layers = [max_dimension * 2] + mlp_layers
        # Remember activation functions
        self.mlp = torch.nn.Sequential(
            *[
                nn.Linear(mlp_layers[i], mlp_layers[i + 1])
                for i in range(0, len(mlp_layers) - 1)
            ]
        )

        self.final_linear = nn.Linear(mlp_layers[-1] + max_dimension, 1)
        self.criterion = (
            torch.nn.BCEWithLogitsLoss()
            if data_schema["objetive"] == "binary"
            else torch.nn.MSELoss()
        )

        self.lr_rate = lr_rate

    def forward(self, interactions, context, users, items):

        user = self.user_embedding(interactions[:, 0])
        item = self.item_embedding(interactions[:, 1])

        user_features = self.encode_user(users)
        item_features = self.encode_item(items)

        user = torch.cat([user, user_features], dim=1)
        item = torch.cat([item, item_features], dim=1)

        mlp_output = self.mlp(torch.cat([user, item], dim=1))
        gmf_output = user * item

        x = self.final_linear(torch.cat([gmf_output, mlp_output], dim=1))

        return x

    def predict(self, pair, context_features, user_features, item_features):
        pair = torch.tensor([pair])

        user_features = torch.tensor([user_features])
        item_features = torch.tensor([item_features])

        return self(pair, context_features, user_features, item_features)

    def encode_user(self, user):
        r = []
        for idx, feature in enumerate(self.user_features):
            feature_representation = feature(user[:, feature.idx])
            r.append(feature_representation)
        r = torch.cat(r, dim=1)  # Concatenate all features
        return r

    def encode_item(self, item):
        r = []
        for idx, feature in enumerate(self.item_features):
            feature_representation = feature(item[:, feature.idx])
            r.append(feature_representation)
        r = torch.cat(r, dim=1)  # Concatenate all features
        return r

    def training_step(self, batch):
        interactions, context, users, items = batch
        yhat = self(interactions.long(), context, users, items).float()
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