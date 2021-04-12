import torchtext


class RerankerDataset(torchtext.data.Dataset):

    def __init__(self, data, query_builder, passages_per_query=1, numerized=False, **kwargs):
        self.query_builder = query_builder
        self.passages_per_query = passages_per_query
        self.numerized = numerized

        fields = self.prepare_fields(1.)
        examples = self.get_example_list(data, fields)

        super().__init__(examples, fields, **kwargs)

    def get_example_list(self, data, fields):
        question = data["question"]
        passages = data["passages"]

        if not self.numerized:
            question = self.query_builder.tokenize_and_convert_to_ids(question)
            passages = [(self.query_builder.tokenize_and_convert_to_ids(item[0]), self.query_builder.tokenize_and_convert_to_ids(item[1])) for item in passages]

        max_length = self.query_builder.max_seq_length if self.query_builder.max_seq_length else self.query_builder.tokenizer.model_max_length
        max_length-= self.query_builder.num_special_tokens_to_add
        max_length-= len(question)

        query_length = 0
        query_passages = []
        examples = []
        for (t, p) in passages:

            if query_length + len(t) + len(p) + 2 > max_length or len(query_passages) >= self.passages_per_query:
                features = self.query_builder(question, query_passages, self.numerized)
                examples.append(
                    torchtext.data.Example.fromlist(
                        [
                            features["input_ids"],
                            features["attention_mask"],
                        ], 
                        fields
                    )
                )
                query_passages = []
                query_length = 0

            query_passages.append((t, p))
            query_length += len(t) + len(p) + 2

        features = self.query_builder(question, query_passages, self.numerized)
        examples.append(
            torchtext.data.Example.fromlist(
                [
                    features["input_ids"],
                    features["attention_mask"],
                ], 
                fields
            )
        )

        return examples

    @staticmethod
    def prepare_fields(pad_t):
        return [
            ("input_ids", torchtext.data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=pad_t)),
            ("attention_mask", torchtext.data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0.)),
            #("score_mask", torchtext.data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=float("-Inf")))
        ]

    @classmethod
    def download(cls, root, check=None):
        raise NotImplementedError


    def filter_examples(self, field_names):
        raise NotImplementedError

