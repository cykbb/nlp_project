from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from d2l.model import ClassifierModel

def _make_rnn_kwargs(
    embedding_dim: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool,
) -> dict[str, object]:
    return {
        "input_size": embedding_dim,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout if num_layers > 1 else 0.0,
        "batch_first": True,
        "bidirectional": bidirectional,
    }

class TextClassificationModel(ClassifierModel, ABC):
    """Shared recurrent text classification backbone used by RNN/LSTM/GRU variants."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_classes: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        pad_token_id: int = 0,
        embedding: Optional[nn.Embedding] = None,
        encoder: nn.Module,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_prob = dropout
        self.bidirectional = bidirectional
        self.pad_token_id = pad_token_id

        self.embedding = embedding or nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )

        self.encoder = encoder
        self.classifier = nn.Linear(
            hidden_size * (2 if bidirectional else 1), num_classes
        )
        self.dropout = nn.Dropout(dropout)

    def _compute_lengths(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1)
        else:
            lengths = (input_ids != self.pad_token_id).sum(dim=1)
        return torch.clamp(lengths, min=1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        lengths = self._compute_lengths(input_ids, attention_mask)
        embedded = self.dropout(self.embedding(input_ids))

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        hidden_state = self._encode_sequences(packed)
        features = self._get_feature(hidden_state)
        return self.classifier(self.dropout(features))

    @abstractmethod
    def _encode_sequences(
        self, packed_embeddings: nn.utils.rnn.PackedSequence
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_feature(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.bidirectional:
            forward_last = hidden_state[-2]
            backward_last = hidden_state[-1]
            return torch.cat((forward_last, backward_last), dim=1)
        return hidden_state[-1]

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(y_hat, y)

class TextClassificationLSTM(TextClassificationModel):
    """BiLSTM based classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_classes: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        pad_token_id: int = 0,
        embedding: Optional[nn.Embedding] = None,
    ) -> None:
        encoder = nn.LSTM(
            **_make_rnn_kwargs(
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        )
        super().__init__(
            vocab_size,
            embedding_dim,
            hidden_size,
            num_classes,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            pad_token_id=pad_token_id,
            embedding=embedding,
            encoder=encoder,
        )

    def _encode_sequences(
        self, packed_embeddings: nn.utils.rnn.PackedSequence
    ) -> torch.Tensor:
        _, (hidden_state, _) = self.encoder(packed_embeddings)
        return hidden_state
class TextClassificationGRU(TextClassificationModel):
    """BiGRU based classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_classes: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        pad_token_id: int = 0,
        embedding: Optional[nn.Embedding] = None,
    ) -> None:
        encoder = nn.GRU(
            **_make_rnn_kwargs(
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        )
        super().__init__(
            vocab_size,
            embedding_dim,
            hidden_size,
            num_classes,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            pad_token_id=pad_token_id,
            embedding=embedding,
            encoder=encoder,
        )

    def _encode_sequences(
        self, packed_embeddings: nn.utils.rnn.PackedSequence
    ) -> torch.Tensor:
        _, hidden_state = self.encoder(packed_embeddings)  # type: ignore[misc]
        return hidden_state
class TextClassificationRNN(TextClassificationModel):
    """Vanilla RNN based classifier (supports tanh/relu)."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_classes: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        pad_token_id: int = 0,
        embedding: Optional[nn.Embedding] = None,
        nonlinearity: str = "tanh",
    ) -> None:
        encoder = nn.RNN(
            nonlinearity=nonlinearity,
            **_make_rnn_kwargs(
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            ),
        )
        super().__init__(
            vocab_size,
            embedding_dim,
            hidden_size,
            num_classes,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            pad_token_id=pad_token_id,
            embedding=embedding,
            encoder=encoder,
        )

    def _encode_sequences(
        self, packed_embeddings: nn.utils.rnn.PackedSequence
    ) -> torch.Tensor:
        _, hidden_state = self.encoder(packed_embeddings)  # type: ignore[misc]
        return hidden_state
