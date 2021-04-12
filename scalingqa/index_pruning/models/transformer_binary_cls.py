import collections

from torch import nn
from transformers import AutoModel, PreTrainedModel, ElectraModel
from transformers.activations import get_activation
from transformers.models.electra.modeling_electra import ElectraClassificationHead


class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.out_proj = nn.Linear(config["hidden_size"], config["num_labels"])

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class TransformerBinaryClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = AutoModel.from_pretrained(config["model_type"], cache_dir=config["cache_dir"])
        self.dropout = nn.Dropout(config["cls_dropout"])
        if type(self.transformer) == ElectraModel:
            self.classifier = ElectraClassificationHead({
                "hidden_size": self.transformer.config.hidden_size,
                "hidden_dropout_prob": self.transformer.config.hidden_dropout_prob,
                "num_labels": 1,
            })
        else:
            self.classifier = nn.Linear(self.transformer.config.hidden_size, 1)

        self.init_weights(type(self.transformer))

    def init_weights(self, clz):
        """ Applies model's weight initialization to all non-pretrained parameters of this model"""
        for ch in self.children():
            if issubclass(ch.__class__, nn.Module) and not issubclass(ch.__class__, PreTrainedModel):
                ch.apply(lambda module: clz._init_weights(self.transformer, module))

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.transformer(input_ids, token_type_ids=token_type_ids,
                                   attention_mask=attention_mask)

        if type(self.transformer) == ElectraModel:
            pooled_output = outputs[0]
        else:
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits.squeeze(-1)
