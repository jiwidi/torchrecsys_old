import pytorch_lightning as pl
import torch
from torch import nn

from torchrecsys.models.bert_utils import BERTEmbedding, TransformerBlock


class Bert4Rec(pl.LightningModule):
    """
    Bert4Rec implementation wrapping pytorch lightning module.
    """

    def __init__(
        self,
        data_schema,
        n_layers=2,
        heads=4,
        hidden_units=512,
        dropout=0.1,
        n_attention_layers=3,
        training_metrics=False,
        learning_rate=3e-5,
        num_rec=14,
        index_pad_token=0,
    ):
        super().__init__()

        n_items = data_schema["n_items"]
        max_length = data_schema["max_length"]

        self.index_pad_token = index_pad_token
        vocab_size = n_items + 2
        self.learning_rate = learning_rate
        self.num_rec = num_rec

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(
            vocab_size=vocab_size,
            embed_size=hidden_units,
            max_length=max_length,
            dropout=dropout,
        )

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_units,
                    heads,
                    hidden_units * 4,
                    dropout,
                    n_attention_layers=n_attention_layers,
                )
                for _ in range(n_layers)
            ]
        )

        # Final layer
        self.out = nn.Linear(hidden_units, n_items + 1)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.index_pad_token
        )  # ignore index from mask token

        self.training_metrics = training_metrics
        # Weights init
        # self.weight_init() # To implement

        # Save hyperparameters for logging
        self.save_hyperparameters()

    def weight_init(self):
        # To implement
        pass

    def forward(self, x):
        mask = (
            (x != self.index_pad_token)
            .unsqueeze(1)
            .repeat(1, x.size(1), 1)
            .unsqueeze(1)
        )

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        x = self.out(x)

        return x

    def training_step(self, batch, batch_idx):
        seqs, labels = batch

        logits = self(seqs)  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T

        loss = self.criterion(logits, labels)

        if self.training_metrics:
            # training metrics slow down training! Use only for debugging, no
            # production.
            logits = logits.softmax(1)

            acc = self.acc(logits, labels)
            recall = self.recall(logits, labels)

            self.log(
                f"train/step_recall_top_{self.num_rec}",
                recall,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
            self.log(
                f"train/step_acc_top_{self.num_rec}",
                acc,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

        self.log(
            "train/step_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        seqs, labels = batch
        logits = self(seqs)  # B x T x V

        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Keep last item logits and label on all batch
        logits = logits[:, -1, :]
        labels = labels[:, -1]

        logits = logits.softmax(1)

        return {
            "val_loss": loss,
            "preds": logits,
            "target": labels,
        }

    def validation_step_end(self, outputs):
        out = outputs["preds"]
        target = outputs["target"]

        acc = self.acc(out, target)
        recall = self.recall(out, target)

        # No logging here
        return {
            "val_loss": outputs["val_loss"],
            "val_acc": acc,
            "val_recall": recall,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_ac = torch.stack([x["val_acc"] for x in outputs]).mean()
        avg_recall = torch.stack([x["val_recall"] for x in outputs]).mean()

        self.log("val/loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", avg_ac, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "val/recall",
            avg_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def predict_step(self, batch, batch_idx):
        seqs, users = batch
        logits = self(seqs)  # B x T x V
        logits = logits[:, -1, :]
        logits = logits.softmax(1)

        topk = torch.topk(logits, self.num_rec).indices  # Top K products for the user

        return users, topk

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer]
