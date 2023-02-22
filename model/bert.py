"""Model definition for fine tuned BERT model with a token classification head"""

import torch.nn as nn
from transformers import BertForTokenClassification


class BERTPhraseBreakPredictor(nn.Module):
    """Model definition for fine tuned BERT model with token classification head"""

    def __init__(self, cfg, num_puncs):
        """Instantiate the model"""
        super().__init__()

        self.num_puncs = num_puncs
        self.bert_model = BertForTokenClassification.from_pretrained(cfg["bert_model_name"], num_labels=num_puncs)

    def forward(self, texts, attention_mask, labels):
        """Forward pass"""
        outputs = self.bert_model(texts, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs[0], outputs[1]

        return loss, logits
