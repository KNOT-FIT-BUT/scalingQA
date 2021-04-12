import torch.nn.functional as F

from torch import Tensor as T
from torch import nn
from transformers import AutoModel, PreTrainedModel, AutoConfig
from typing import Union


class LRM_encoder(nn.Module):
    def __init__(self, config: dict, do_not_download_weights=False):
        super().__init__()
        if do_not_download_weights:
            self.lrm = AutoModel.from_config(
                AutoConfig.from_pretrained(config["model_type"], cache_dir=config["model_cache_dir"]))
        else:
            self.lrm = AutoModel.from_pretrained(config["model_type"], cache_dir=config["model_cache_dir"])
        output_shape = self.lrm.config.hidden_size
        projection_bias = config.get("use_projection_bias", True)
        self.linear = nn.Linear(output_shape, config["emb_dim"], bias=projection_bias) if config[
            "use_projection"] else nn.Identity()
        self.dropout = nn.Dropout(config["dropout_rate"]) if config.get("dropout_rate", 0.) > 1e-6 else nn.Identity()
        self.config = config

        self.init_weights(type(self.lrm))

    def init_weights(self, clz):
        """
        Initialize all children of this model apart from subclass of PreTrainedModel
        with the same initialization function as transformer model uses (e.g., usually truncated normal).
        """
        for ch in self.children():
            if issubclass(ch.__class__, nn.Module) and not issubclass(ch.__class__, PreTrainedModel):
                ch.apply(lambda module: clz._init_weights(self.lrm, module))

    def forward(self, q_ids: T, q_segment_ids: T, q_mask: T, embeddings: T, targets: Union[T, None]):
        q = self.encode(q_ids, q_segment_ids, q_mask)
        return self.get_scores(q, embeddings, targets)

    def encode(self, input: T, q_segment_ids: T, q_mask: T):
        outputs = self.lrm(input_ids=input, token_type_ids=q_segment_ids, attention_mask=q_mask)
        d = outputs[0][:, 0, :]
        encoded_q = self.dropout(self.linear(d))
        return encoded_q

    def get_scores(self, q: T, embeddings: T,
                   targets: Union[T, None],
                   return_scores: bool = False, return_only_scores: bool = False):
        scores = q @ embeddings.T

        if return_only_scores:
            return scores

        xe = F.cross_entropy(scores, targets, reduction='none')
        if return_scores:
            return xe, scores
        return xe
