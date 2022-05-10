import torch
from torch import nn

from torchrecsys.layers.features import FeatureLayer
from torchrecsys.models.base import BaseModel


class ALS(BaseModel):
    def __init__(
        self,
        data_schema,
        lr_rate: float = 0.01,
        embedding_size: int = 64,
        feature_embedding_size: int = 8,
    ):
        super().__init__()
        interactions_schema = data_schema["interactions"]

        self.n_users = interactions_schema[0]
        self.n_items = interactions_schema[1]

        # User features encoding
        self.user_features = nn.ModuleList()
        self.user_feature_dimension = 0

        ##Make feature encoding a function and move it to base
        for feature_idx, feature in enumerate(data_schema["user_features"]):
            if feature.dtype == "category":
                layer_name = f"user_{feature.name}_embedding"
                layer = nn.Embedding(
                    feature.unique_value_count + 1, feature_embedding_size
                )

                f_layer = FeatureLayer(name=layer_name, layer=layer, idx=feature_idx)
                self.user_features.append(f_layer)
                self.user_feature_dimension += feature_embedding_size

        # Item features encoding
        self.item_features = nn.ModuleList()
        self.item_feature_dimension = 0
        for feature_idx, feature in enumerate(data_schema["item_features"]):
            if feature.dtype == "category":
                layer_name = f"item_{feature.name}_embedding"
                layer = nn.Embedding(
                    feature.unique_value_count + 1, feature_embedding_size
                )

                f_layer = FeatureLayer(name=layer_name, layer=layer, idx=feature_idx)
                self.item_features.append(f_layer)
                self.item_feature_dimension += feature_embedding_size

        aux_user_id_dimensions = self.user_feature_dimension + embedding_size
        aux_item_id_dimensions = self.item_feature_dimension + embedding_size

        max_dimension = max(aux_user_id_dimensions, aux_user_id_dimensions)
        user_id_dimensions = embedding_size + (max_dimension - aux_user_id_dimensions)
        item_id_dimensions = embedding_size + (max_dimension - aux_item_id_dimensions)

        self.user_embedding = nn.Embedding(self.n_users + 1, user_id_dimensions)
        self.item_embedding = nn.Embedding(self.n_items + 1, item_id_dimensions)

        self.user_biases = nn.Embedding(self.n_users, 1)
        self.item_biases = nn.Embedding(self.n_items, 1)

        self.criterion = (
            torch.nn.BCEWithLogitsLoss()
            if data_schema["objetive"] == "binary"
            else torch.nn.MSELoss()
        )

        self.lr_rate = lr_rate

    def forward(self, interactions, context, users, items):

        user = self.user_embedding(interactions[:, 0].long())
        item = self.item_embedding(interactions[:, 1].long())

        user_features = self.encode_user(users)
        item_features = self.encode_item(items)

        user_factor = torch.cat([user, user_features], dim=1)
        item_factor = torch.cat([item, item_features], dim=1)

        pred = self.user_biases(user) + self.item_biases(item)
        pred += (user_factor * item_factor).sum(1, keepdim=True)
        return pred.squeeze()

    def encode_user(self, user):
        r = []
        for idx, feature in enumerate(self.user_features):
            feature_embedding = feature(user[:, feature.idx])
            r.append(feature_embedding)
        r = torch.cat(r, dim=1)
        return r

    def encode_item(self, item):
        r = []
        for idx, feature in enumerate(self.item_features):
            feature_embedding = feature(item[:, feature.idx])
            r.append(feature_embedding)
        r = torch.cat(r, dim=1)
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
        user_weights = [
            self.user_embedding.weight,
            self.user_features.weight,
            self.user_biases.weight,
        ]
        item_weights = [
            self.item_embedding.weight,
            self.item_features.weight,
            self.item_biases.weight,
        ]

        ##Need to figure out how to do the custom loss steps, when i get internet
        item_optimizer = torch.optim.SGD(user_weights, self.lr_rate)
        user_optimizer = torch.optim.SGD(item_weights, self.lr_rate)
        return [item_optimizer, user_optimizer]
