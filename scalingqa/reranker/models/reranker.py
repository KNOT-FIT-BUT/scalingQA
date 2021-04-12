import torch
from transformers import PreTrainedModel


class BaselineReranker(torch.nn.Module):
    """ Baseline passage reranker used in the paper. """

    def __init__(self, config, encoder):
        super().__init__()

        self.config = config
        self.encoder = encoder
        self.vt = torch.nn.Linear(config.hidden_size, 1, bias=False)

        self.init_weights(type(self.encoder))

    def init_weights(self, clz):
        """ Applies model's weight initialization to all non-pretrained parameters of this model"""
        for ch in self.children():
            if issubclass(ch.__class__, torch.nn.Module) and not issubclass(ch.__class__, PreTrainedModel):
                ch.apply(lambda module: clz._init_weights(self.encoder, module))

    def forward(self, batch):
        """
        The input looks like:
        [CLS] Q [SEP] <t> title <c> context [EOS]
        """

        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"]
        }

        outputs = self.encoder(**inputs)[1]

        scores = self.vt(outputs)
        scores = scores.view(1,-1)

        return scores