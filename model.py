"""Model definition, loss function and evaluation metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score


class PhraseBreakPredictor(nn.Module):
    """Phrase break prediction model
    """

    def __init__(self, cfg):
        """Instantiate the model
        """
        super().__init__()

        # Embedding layer
        self.embedding_layer = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)

        # BLSTM layer
        self.blstm = nn.LSTM(cfg.embedding_dim, cfg.blstm_size // 2, num_layers=2, batch_first=True, bidirectional=True)

        # Output fully connected layer
        self.fc = nn.Linear(cfg.blstm_size, cfg.num_tags)

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


def loss_fn(outputs, labels):
    """Compute the cross-entropy loss; while excluding the loss terms for padding tokens

        Args:
            outputs (tensor): log softmax output of the model [B * T_max, num_tags]
            labels (tensor): Ground truth labels where each element is a label index in [0, 1, ....num_tags - 1] or
                             -1 in the case of padding tokens [B, T_max]
        
        Returns:
            loss (tensor): cross-entropy loss for all tokens in the batch
    """
    # [B, T_max] -> [B * T_max]
    labels = labels.view(-1).contiguous()

    # Generate mask for padding terms
    mask = (labels >= 0).float()

    # Since padding tokens have index -1, we need to convert them to a positive number
    labels = labels % outputs.shape[1]

    # Total number of tokens
    num_tokens = int(torch.sum(mask))

    # Compute cross-entropy loss for all tokens (except padding tokens)
    return -torch.sum(outputs[range(outputs.shape[0]), labels] * mask) / num_tokens

def f1_measure(outputs, labels):
    """Compute the F1 score between the predicted outputs and ground truth labels

        Args:
            outputs (np.ndarray): log softmax output of the model [B * T_max, num_tags]
            labels (np.ndarray): Ground truth labels where each element is a label index in [0, 1, ....num_tags - 1] or
                             -1 in the case of padding tokens [B, T_max]
        
        Returns:
           f1_score (float): The F1 score for all tokens in the batch 
    """
    # [B, T_max] -> [B * T_max]
    labels = labels.ravel()

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)

    # Compute F1 score on batch
    return f1_score(labels, outputs, average="micro")
