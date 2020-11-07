from collections import Counter
from typing import List, Tuple, Iterable

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from methods.tfidf import TfIdfBaseModel
from preprocess.seq_coder import SeqCoder


class CosineModel(TfIdfBaseModel):
    def __init__(self, coder: SeqCoder):
        super().__init__(coder)
        self.coder = coder

    def fit(self, sim_train_data: List[Tuple[int, int, int]], unsup_data: Iterable[int] = None) -> 'CosineModel':
        super().fit(sim_train_data, unsup_data)
        return self

    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        scores = []

        def to_dict(stack_id):
            stack_freqs = Counter(self.coder(stack_id))
            # return stack_freqs
            # return {frame: 1 for frame, cnt in stack_freqs.items()}  # codine(1)
            # return {word: v[0] * v[1] ** 2 for word, v in self.tfidf_model.tfidf_vectorizer.transform(stack_id).items()}
            return {word: v[1] for word, v in self.tfidf_module.tfidf_vectorizer.transform(stack_id).items()}  # codine(idf)

        anchor = to_dict(anchor_id)
        for stack_id in stack_ids:
            array = DictVectorizer().fit_transform([anchor, to_dict(stack_id)])
            scores.append(cosine_similarity(array[0:], array[1:])[0, 0])

        return scores

    def name(self) -> str:
        return self.coder.name() + "_cosine"
