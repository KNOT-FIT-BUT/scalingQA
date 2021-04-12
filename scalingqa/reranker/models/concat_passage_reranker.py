import torch
from transformers import PreTrainedModel


class ConcatPassageReranker(torch.nn.Module):
    """ 
    The passage reranker for early passage concatenation. The model is not
    used in the paper.
    """

    def __init__(self, config, encoder, query_builder, with_cls=True):
        super().__init__()

        self.config = config
        self.encoder = encoder
        self.query_builder = query_builder
        self.with_cls = with_cls

        self.vt = torch.nn.Linear((2+with_cls)*config.hidden_size, 1, bias=False)

        self.init_weights(type(self.encoder))

    def init_weights(self, clz):
        """ Applies model's weight initialization to all non-pretrained parameters of this model"""
        for ch in self.children():
            if issubclass(ch.__class__, torch.nn.Module) and not issubclass(ch.__class__, PreTrainedModel):
                ch.apply(lambda module: clz._init_weights(self.encoder, module))

    def forward(self, batch):
        """
        The input looks like:
        [CLS] question [SEP] <t> title#1 <c> context#1 <t> title#2 <c> context#2 <t> ... <t> title#N <c> context#N [EOS]
        """
        
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"]
        }

        outputs = self.encoder(**inputs)[0]
        batch_size, _, hidden_size = outputs.shape

        ret = []

        for input_, output in zip(batch["input_ids"], outputs):
            non_zero = torch.nonzero((input_ == self.query_builder.start_context_token_id), as_tuple=True)
            context = output[non_zero[0]]
            non_zero = torch.nonzero((input_ == self.query_builder.start_title_token_id), as_tuple=True)
            title = output[non_zero[0]]

            cls = output[0].repeat(context.shape[0], 1)

            representation = [cls, title, context] if self.with_cls else [title, context]

            ret.append(torch.cat(representation, -1))

        max_ = max(item.shape[0] for item in ret)

        device = ret[0].get_device()

        scores = torch.zeros(batch_size, max_, (2+self.with_cls)*hidden_size).to(device)
        scores_mask = torch.empty(batch_size, max_).fill_(float("-Inf")).to(device)

        for i, item in enumerate(ret):
            scores[i,:item.shape[0]] = item
            scores_mask[i,:item.shape[0]] = torch.zeros(item.shape[0]) 

        scores = self.vt(scores)
        scores = scores.squeeze(-1)

        scores = scores + scores_mask

        return scores