import math
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchmetrics


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    """
    BERT4REC Layernorm.
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GELU(nn.Module):
    """
    Gelu implementation. BERT4REC mentions its usage
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )


class PositionwiseFeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Layer is a type of feedforward layer consisting of two dense layers that applies to
    the last dimension, which means the same dense layers are used for each position item in the sequence, so called position-wise.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


# -----------------Attention modules
class Attention(nn.Module):
    """
    Compute Scaled Dot Product Attention
    """

    def forward(
        self,
        query,
        key,
        value,
        mask: Optional[torch.Tensor] = None,
        dropout: Optional[float] = None,
    ):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            _MASKING_VALUE = (
                -1e9 if query.dtype == torch.float32 else -1e4
            )  # 32 & 16bit support
            scores = scores.masked_fill(mask == 0, _MASKING_VALUE)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = torch.nn.functional.dropout(
                p_attn,
                p=dropout,
                training=self.training,
            )  # Change dropout to functional for torchscript compatibility
            # p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Module for attention mechanisms which runs through an attention mechanism several times in parallel.
    """

    def __init__(self, h, d_model, dropout=0.1, n_layers=3):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(n_layers)]
        )
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = dropout  # nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


# ----------------Embedding


class PositionalEmbedding(nn.Module):
    """
    Computes positional embedding following "Attention is all you need"
    """

    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos

        sum of all these features are output of BERTEmbedding
    """

    def __init__(
        self,
        vocab_size,
        embed_size,
        max_len,
        token_embedding,
        dropout=0.1,
    ):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        #         self.token = nn.Embedding(
        #             vocab_size, embed_size, padding_idx=0
        #         )  # TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.token = token_embedding
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence)
        x += self.position(sequence)  # + self.segment(segment_label)
        return self.dropout(x)


# -------------Transformer block


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(
        self,
        hidden,
        attn_heads,
        feed_forward_hidden,
        dropout,
        n_attention_layers=3,
    ):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(
            h=attn_heads,
            d_model=hidden,
            dropout=dropout,
            n_layers=n_attention_layers,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
        )
        # self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.input_norm = LayerNorm(hidden)
        self.input_dropout = nn.Dropout(p=dropout)

        # self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_norm = LayerNorm(hidden)
        self.ooutput_dropout = nn.Dropout(p=dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # Input sublayer
        x_b = self.input_norm(x)
        x_b = self.attention(x_b, x_b, x_b, mask=mask)
        x_b = self.input_dropout(x_b)
        x = x + x_b

        # Output sublayer
        x_b = self.output_norm(x)
        x_b = self.feed_forward(x_b)
        x_b = self.ooutput_dropout(x_b)
        x = x + x_b

        # x = x + self.input_dropout(sublayer(self.norm(x)))

        # With lambdas - Is incompatible with torchscript so i moved it to this forward fn
        # x = self.input_sublayer(
        #     x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask)
        # )
        # x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


# COMMAND ----------


class BertDataset(data.Dataset):
    """
    Dataset that applys BERT masking logic to sequences.

    If mode==train the dataset will randomly mask sequences, if mode==validate it will only mask the last interaction following "leave one out" policy.
    """

    def __init__(
        self,
        data,
        max_len,
        num_items,
        mask_prob=0.15,
        mask_token=1,
        mode="train",
        random_seq_start=False,
        session_ids=None,
        item_catalog=None,
    ):
        self.u2seq = data
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = np.random
        self.random_seq_start = random_seq_start
        self.mode = mode
        self.session_ids = session_ids
        self.item_catalog = item_catalog

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, index):
        seq = self.u2seq[index]
        tokens = []
        labels = []
        if self.mode == "train":
            seq = seq[:-1]  # Remove the latest one we will use for validation

            if self.rng.random() < 0.25:
                np.random.shuffle(seq)

            for s in seq:
                prob = self.rng.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:
                        tokens.append(self.mask_token)
                    elif prob < 0.9:
                        tokens.append(self.rng.randint(1, self.num_items))
                    else:
                        tokens.append(s)

                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)
            if not self.random_seq_start or len(tokens) < self.max_len + 1:
                tokens = tokens[-self.max_len :]
                labels = labels[-self.max_len :]
            # randomize the sequene within the full sequence, only if
            # max_len<len(tokens)
            else:
                idx = np.random.randint(self.max_len, len(tokens))
                tokens = tokens[:idx][-self.max_len :]
                labels = labels[:idx][-self.max_len :]

            mask_len = self.max_len - len(tokens)

            tokens = [0] * mask_len + tokens
            labels = [0] * mask_len + labels

            tokens = np.array([self.item_catalog[x] for x in tokens])

            return torch.LongTensor(tokens), torch.LongTensor(labels)
        elif self.mode == "validate":
            mask_len = self.max_len - len(seq)

            tokens = np.copy(seq).tolist()[-self.max_len :]
            labels = np.copy(seq).tolist()[-self.max_len :]

            tokens = [0] * mask_len + tokens[:-1] + [self.mask_token]
            labels = [0] * mask_len + labels

            tokens = np.array([self.item_catalog[x] for x in tokens])

            return torch.LongTensor(tokens), torch.LongTensor(labels)
        elif self.mode == "inference":
            mask_len = self.max_len - len(seq)

            tokens = np.copy(seq).tolist()[-(self.max_len) :]

            tokens = [0] * mask_len + tokens + [self.mask_token]
            tokens = tokens[-self.max_len :]

            tokens = np.array([self.item_catalog[x] for x in tokens])

            session_id = self.session_ids[index]

            return torch.LongTensor(tokens), torch.LongTensor(tokens), session_id


# COMMAND ----------
class featureEmbedder(torch.nn.Module):
    def __init__(self, n_features, embedding_size, embedding_dimensions):
        super().__init__()
        self.n_features = n_features
        for feature in range(self.n_features):
            setattr(
                self,
                f"embedding_feature_{feature}",
                torch.nn.Embedding(
                    embedding_size[feature], embedding_dimensions[feature]
                ),
            )

    def forward(self, x):
        r = []
        for feature in range(self.n_features):
            aux = getattr(self, f"embedding_feature_{feature}", 0)(x[:, :, feature])
            r.append(aux)

        return torch.cat(r, dim=2)


class RECBERTO(pl.LightningModule):
    """
    Bert4Rec implementation wrapping pytorch lightning module.
    """

    def __init__(
        self,
        max_len,
        num_items,
        n_layers=4,
        heads=4,
        hidden_units=512,
        dropout=0.1,
        training_metrics=False,
        learning_rate=3e-5,
    ):
        super().__init__()
        vocab_size = num_items + 2
        self.learning_rate = learning_rate
        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(
            vocab_size=vocab_size,
            embed_size=hidden_units,
            max_len=max_len,
            token_embedding=featureEmbedder(
                17, [30000] + [1028 for u in range(17)], [256] + [16 for u in range(17)]
            ),
            dropout=dropout,
        )

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_units, heads, hidden_units * 4, dropout)
                for _ in range(n_layers)
            ]
        )

        # Final layer
        self.out = nn.Linear(hidden_units, num_items + 1)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Metrics
        self.acc = torchmetrics.Accuracy(top_k=100)
        self.recall = torchmetrics.Recall(top_k=100)

        self.training_metrics = training_metrics
        # Weights init
        # self.weight_init() # To implement

        # Save hyperparameters for later loading
        self.save_hyperparameters()

    def weight_init(self):
        # To implement
        pass

    def forward(self, x):
        aux = x[:, :, 0]
        mask = (aux > 0).unsqueeze(1).repeat(1, aux.size(1), 1).unsqueeze(1)

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

            self.log(
                "train/step_acc_top_14",
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

    def test_step(self, batch, batch_idx):
        seqs, labels, sessionid = batch
        logits = self(seqs)  # B x T x V
        top_indices = torch.topk(
            logits, 100
        ).indices  # Top n_recos products for the user
        return {"sessionid": sessionid, "top_indices": top_indices}

    def test_epoch_end(self, outputs):
        sessions = torch.cat([x["sessionid"] for x in outputs])
        y_hat = torch.cat([x["top_indices"] for x in outputs])

        sessions = sessions.tolist()
        y_hat = y_hat.tolist()

        data = {"session_id": sessions, "item_id": y_hat}
        df = pd.DataFrame.from_dict(data)
        df["item_id"] = df.item_id.apply(lambda x: x[0])
        df = df.explode("item_id")
        df["rank"] = df.groupby(["session_id"]).cumcount() + 1
        df.to_csv("predict.csv", index=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer]
