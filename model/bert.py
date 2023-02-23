"""Model definition for BERT model with a token classification head"""

import torch.nn as nn
from transformers import AutoModelForTokenClassification


class BERTPhraseBreakPredictor(nn.Module):
    """BERT model with a token classification head"""

    def __init__(self, model_name, num_puncs):
        """Instantiate the model"""
        super().__init__()

        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_puncs)

    def forward(self, texts, attention_masks, punctuations):
        """Forward pass"""
        outputs = self.model(input_ids=texts, attention_mask=attention_masks, labels=punctuations)

        loss, logits = outputs[0], outputs[1]

        return loss, logits
