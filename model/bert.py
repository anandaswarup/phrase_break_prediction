"""Model definition for fine tuned BERT model with a token classification head"""

import torch.nn as nn
from transformers import BertModel


class BERTPhraseBreakPredictor(nn.Module):
    """Model definition for fine tuned BERT model with a token classification head"""

    def __init__(self, cfg, num_puncs):
        """Instantiate the model"""
        super().__init__()

        # BERT model
        self.bert_model = BertModel.from_pretrained(cfg["bert_model_name"], add_pooling_layer=False)

        # Dropout layer
        self.dropout_layer = nn.Dropout(0.5)

        # Output (fully connected layer)
        self.output_layer = nn.Linear(in_features=self.bert_model.config.hidden_size, out_features=num_puncs)

    def forward(self, texts, attention_masks):
        """Forward pass"""
        outputs = self.bert_model(
            input_ids=texts,
            attention_mask=attention_masks,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout_layer(sequence_output)

        logits = self.output_layer(sequence_output)

        return logits
