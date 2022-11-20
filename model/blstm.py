"""Model definition for BLSTM token classification model using task specific word embeddings trained from scratch"""

import torch.nn as nn
import torch.nn.functional as F


class PhraseBreakPredictor(nn.Module):
    """Phrase break prediction model"""

    def __init__(self, num_words, word_embedding_dim, num_blstm_layers, blstm_layer_size, num_puncs, padding_idx):
        """Instantiate the model"""
        super().__init__()

        self.num_words = num_words
        self.word_embedding_dim = word_embedding_dim
        self.num_blstm_layers = num_blstm_layers
        self.blstm_layer_size = blstm_layer_size
        self.num_puncs = num_puncs

        # Word embedding
        self.word_embedding = nn.Embedding(
            num_embeddings=num_words, embedding_dim=word_embedding_dim, padding_idx=padding_idx
        )

        # BLSTM
        self.blstm = nn.LSTM(
            input_size=word_embedding_dim,
            hidden_size=blstm_layer_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Output (fully connected layer)
        self.output = nn.Linear(in_features=blstm_layer_size, out_features=num_puncs)

    def forward(self, x):
        """Forward pass"""
        # [B, T_max] -> [B, T_max, word_embedding_dim]
        x = self.word_embedding(x)

        # [B, T_max, word_embedding_dim] -> [B, T_max, blstm_size]
        x, _ = self.blstm(x)

        # [B, T_max, blstm_size] -> [B * T_max, blstm_size]
        x = x.reshape(-1, x.shape[2]).contiguous()

        # [B * T_max, blstm_size] -> [B * T_max, num_puncs]
        x = self.output(x)

        return F.log_softmax(x, dim=1)
