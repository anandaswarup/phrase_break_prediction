"""Model definition for fine tuned BERT model with a token classification head"""

import torch.nn as nn


class BERTPhraseBreakPredictor(nn.Module):
    """Model definition for fine tuned BERT model with a token classification head"""

    def __init__(self, bert_model, num_puncs):
        """Instantiate the model"""
        super().__init__()

        self.bert_model = bert_model
        self.bert_hidden_size = self.bert_model.config.hidden_size
        self.num_puncs = num_puncs

        # Dropout layer
        self.dropout_layer = nn.Dropout(0.5)

        # Output (fully connected layer)
        self.output_layer = nn.Linear(in_features=self.bert_hidden_size, out_features=self.num_puncs)

    def forward(self, texts, attention_masks):
        """Forward pass"""
        # [[B, T_max], [B, T_max]] -> [B, T_max, bert_hidden_size]
        bert_outputs = self.bert_model(texts, attention_masks)
        bert_outputs = bert_outputs.last_hidden_state
        bert_outputs = self.dropout_layer(bert_outputs)

        # [B, T_max, bert_hidden_size] -> [B, T_max, num_puncs]
        logits = self.output_layer(bert_outputs)

        return logits
