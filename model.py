"""Model definition"""

import torch.nn as nn
import torch.nn.functional as F


class PhraseBreakPredictor(nn.Module):
    """Phrase break prediction model
    """

    def __init__(self, cfg, vocab_size, num_tags):
        """Instantiate the model
        """
        super().__init__()

        # Embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, cfg.embedding_dim)

        # BLSTM layer
        self.blstm = nn.LSTM(cfg.embedding_dim, cfg.blstm_size // 2, num_layers=2, batch_first=True, bidirectional=True)

        # Output fully connected layer
        self.fc = nn.Linear(cfg.blstm_size, num_tags)

    def forward(self, x):
        """Forward pass
        """
        # [B, T_max] -> [B, T_max, embedding_dim]
        x = self.embedding_layer(x)

        # [B, T_max, embedding_dim] -> [B, T_max, blstm_size]
        x, _ = self.blstm(x)

        # [B, T_max, blstm_size] -> [B * T_max, blstm_size]
        x = x.reshape(-1, x.shape[2]).contiguous()

        # [B * T_max, blstm_size] -> [B * T_max, num_tags]
        x = self.fc(x)

        return F.log_softmax(x, dim=1)
