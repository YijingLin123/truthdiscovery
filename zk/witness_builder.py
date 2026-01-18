"""Witness builder for a zk-friendly TruthFinder pipeline."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, getcontext
from typing import List, Sequence

import numpy as np

from field import FIELD_MODULUS

getcontext().prec = 50

SCALE = Decimal("1e8")


def _sanitize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def extract_vocabulary(texts: Sequence[str], target_size: int | None = None) -> List[str]:
    vocab = []
    seen = set()
    for text in texts:
        for token in text.split():
            cleaned = _sanitize_token(token)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                vocab.append(cleaned)
    if target_size is not None:
        if len(vocab) > target_size:
            raise ValueError(f"Vocabulary size {len(vocab)} exceeds target {target_size}")
        while len(vocab) < target_size:
            filler = f"pad{len(vocab)}"
            vocab.append(filler)
            seen.add(filler)
    return vocab


def _scale_decimal(value: Decimal) -> int:
    scaled = (value * SCALE).to_integral_value(rounding=ROUND_HALF_EVEN)
    return int(scaled) % FIELD_MODULUS


@dataclass
class Witness:
    tfidf_vectors: List[List[int]]
    similarities: List[List[int]]
    fact_confidence: List[int]
    trustworthiness: List[int]
    idf: List[int]
    gamma: int
    rho: int
    trust_init: List[int]


class TruthFinderWitnessBuilder:
    """Builds fixed-precision witnesses ready for zk circuits."""

    def __init__(self, vocabulary: Sequence[str]):
        self.vocab = [v.lower() for v in vocabulary]
        self.index = {token: idx for idx, token in enumerate(self.vocab)}

    def tokenize(self, text: str) -> List[int]:
        counts = [0] * len(self.vocab)
        tokens = [_sanitize_token(tok) for tok in text.split()]
        for tok in tokens:
            if not tok:
                continue
            idx = self.index.get(tok)
            if idx is not None:
                counts[idx] += 1
        return counts

    def _compute_idf(self, documents: List[List[int]]) -> List[Decimal]:
        n_docs = Decimal(len(documents))
        df = [0] * len(self.vocab)
        for doc in documents:
            for idx, cnt in enumerate(doc):
                if cnt > 0:
                    df[idx] += 1
        idf = []
        for d in df:
            value = Decimal(math.log((1 + float(n_docs)) / (1 + float(d)))) + Decimal(1)
            idf.append(value)
        return idf

    def _tfidf(self, counts: List[int], idf: List[Decimal]) -> List[Decimal]:
        total = sum(counts)
        if total == 0:
            total = 1
        total_dec = Decimal(total)
        vector = []
        for idx, cnt in enumerate(counts):
            tf = Decimal(cnt) / total_dec
            vector.append(tf * idf[idx])
        return vector

    def _cosine(self, vectors: List[List[Decimal]]) -> List[List[Decimal]]:
        sims = []
        dense = [np.array(vec, dtype=np.float64) for vec in vectors]
        norms = [np.linalg.norm(vec) for vec in dense]
        for i, vi in enumerate(dense):
            row = []
            for j, vj in enumerate(dense):
                denom = norms[i] * norms[j]
                if denom == 0.0:
                    mapped = Decimal(-1)
                else:
                    cos = float(np.dot(vi, vj) / denom)
                    cos2 = cos * cos
                    mapped = Decimal(2 * cos2 - 1)
                row.append(Decimal(max(-1.0, min(1.0, mapped))))
            sims.append(row)
        return sims

    def _sigmoid(self, value: Decimal) -> Decimal:
        # Polynomial approximation for sigmoid centered at 0.
        return Decimal("0.5") + value / Decimal(4) - (value ** 3) / Decimal(48)

    def _truthfinder(self,
                     sims: List[List[Decimal]],
                     gamma: Decimal,
                     rho: Decimal,
                     trust_seed: Decimal,
                     iterations: int = 5) -> (List[Decimal], List[Decimal]):
        n = len(sims)
        trust = [trust_seed] * n
        confidence = [Decimal(0)] * n
        for _ in range(iterations):
            # Update fact confidence
            for i in range(n):
                support = Decimal(0)
                for j in range(n):
                    if i == j:
                        continue
                    support += confidence[j] * sims[j][i]
                raw = trust[i] + rho * support
                confidence[i] = self._sigmoid(gamma * raw)
            # Update trustworthiness
            for i in range(n):
                trust[i] = confidence[i]
        return confidence, trust

    def build(self,
              texts: Sequence[str],
              gamma: float = 0.8,
              rho: float = 0.6,
              trust_init: float = 0.9,
              iterations: int = 5) -> Witness:
        documents = [self.tokenize(text) for text in texts]
        idf = self._compute_idf(documents)
        tfidf_vectors = [self._tfidf(doc, idf) for doc in documents]
        sims = self._cosine(tfidf_vectors)
        confidence, trust = self._truthfinder(
            sims,
            Decimal(gamma),
            Decimal(rho),
            Decimal(trust_init),
            iterations=iterations,
        )
        trust_seed_scaled = _scale_decimal(Decimal(trust_init))
        return Witness(
            tfidf_vectors=[[ _scale_decimal(val) for val in vec] for vec in tfidf_vectors],
            similarities=[[ _scale_decimal(val) for val in row] for row in sims],
            fact_confidence=[_scale_decimal(val) for val in confidence],
            trustworthiness=[_scale_decimal(val) for val in trust],
            idf=[_scale_decimal(val) for val in idf],
            gamma=_scale_decimal(Decimal(gamma)),
            rho=_scale_decimal(Decimal(rho)),
            trust_init=[trust_seed_scaled] * len(texts),
        )
