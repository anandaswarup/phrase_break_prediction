"""Model definition for BERT model with a token classification head"""

import torch.nn as nn
from transformers import BertModel


class BERTPhraseBreakPredictor(nn.Module):
    """BERT model with a token classification head"""

    def __init__(self, model_name, num_puncs):
        """Instantiate the model"""
        super().__init__()

        # BERT Encoder
        self.bert_layer = BertModel.from_pretrained(model_name)

        # Dropout
        self.dropout_layer = nn.Dropout(0.1)

        # Output layer
        bert_hidden_size = self.bert_layer.config.hidden_size
        self.output_layer = nn.Linear(in_features=bert_hidden_size, out_features=num_puncs)

    def forward(self, text_ids, attention_masks):
        """Forward pass"""
        # BERT encoder
        bert_outputs = self.bert_layer(input_ids=text_ids, attention_mask=attention_masks)

        # Last hidden state output
        last_hidden_state = bert_outputs[0]
        last_hidden_state = self.dropout_layer(last_hidden_state)

        # Output logits
        logits = self.output_layer(last_hidden_state)

        return logits
