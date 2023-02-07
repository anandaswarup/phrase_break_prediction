"""Model definition for BLSTM token classification model using task specific word embeddings trained from scratch"""

import torch.nn as nn


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

        # Dropout layer
        self.dropout_layer = nn.Dropout(p=0.5)

        # Output (fully connected layer)
        self.output = nn.Linear(in_features=blstm_layer_size, out_features=num_puncs)

    def forward(self, texts):
        """Forward pass"""
        # [B, T_max] -> [B, T_max, word_embedding_dim]
        embeddings = self.word_embedding(texts)

        # [B, T_max, word_embedding_dim] -> [B, T_max, blstm_size]
        blstm_outputs, _ = self.blstm(embeddings)
        blstm_outputs = self.dropout_layer(blstm_outputs)

        # [B, T_max, blstm_size] -> [B, T_max, num_puncs]
        logits = self.output(blstm_outputs)

        return logits
