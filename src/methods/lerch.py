from typing import List, Tuple, Iterable

from methods.tfidf import TfIdfBaseModel
from preprocess.seq_coder import SeqCoder


class LerchModel(TfIdfBaseModel):
    def __init__(self, coder: SeqCoder):
        super().__init__(coder)

    def fit(self, sim_train_data: List[Tuple[int, int, int]], unsup_data: Iterable[int] = None) -> 'LerchModel':
        super().fit(sim_train_data, unsup_data)
        return self

    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        anchor_tfidf = self.tfidf_module.tfidf_vectorizer.transform(anchor_id)
        stacks = [set(self.tfidf_module.coder.to_seq(stack_id)) for stack_id in stack_ids]

        scores = []
        for stack in stacks:
            score = 0
            for word in stack:
                if word not in anchor_tfidf:
                    continue
                tf, idf = anchor_tfidf[word]
                tf_idf_pow2 = tf * idf ** 2
                score += tf_idf_pow2

            scores.append(score)

        return scores

    def name(self) -> str:
        return self.tfidf_module.name() + "_lerch"
