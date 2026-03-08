import logging
from functools import lru_cache
from typing import List, Tuple

import numpy as np

from .embeddings import get_embedder

logger = logging.getLogger(__name__)


class SentimentClassifier:
    def __init__(self) -> None:
        self.embedder = get_embedder()
        # Два якоря для бинарной классификации отзывов по тональности (косинусная близость).
        label_texts = {
            "negative": "Негативный отзыв клиента о банке: жалоба, недовольство, критика, ошибка, задержка.",
            "positive": "Позитивный отзыв клиента о банке: благодарность, одобрение, рекомендация, удовлетворённость.",
        }
        self.labels = list(label_texts.keys())
        vectors = self.embedder.embed(list(label_texts.values()))
        self.label_vectors = np.array(vectors, dtype=float)

    def classify_vector(self, vec: List[float]) -> Tuple[str, float]:
        """
        Classify sentiment from an already computed embedding vector.
        Only positive and negative.

        Returns:
            (label, score) where label in {"negative", "positive"},
            score — косинусная близость к выбранному классу.
        """

        v = np.array(vec, dtype=float)
        sims = self.label_vectors @ v
        idx = int(np.argmax(sims))
        return self.labels[idx], float(sims[idx])


@lru_cache
def get_sentiment_classifier() -> SentimentClassifier:
    return SentimentClassifier()

