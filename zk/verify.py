"""Verify TF-IDF and cosine modules using fixed summaries."""

from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np

from witness_builder import TruthFinderWitnessBuilder, extract_vocabulary
from field import FIELD_MODULUS

getcontext().prec = 50

TEXTS = [
    "A new Agilent Technologies, Inc. (Symbol: A) option chain has opened with a $125 put priced "
    "at $4.50 and a $150 call at $8.10, offering opportunities for premium capturing via selling "
    "puts or opening covered calls respectively.",
    "Newly trading August 2024 options for Agilent Technologies Inc., including a $125 put with "
    "$4 premium and a $150 call at $8.10, offer opportunities to sellers of puts or calls for higher "
    "premiums due to upcoming expiration, while noting slightly lower implied volatilities compared "
    "to similar contracts.",
    "A recent round of Agilent Technologies options, trading at 241 days until expiry, now offers "
    "potential upside for option sellers and lower cost entry points for call buyers. The August "
    "$150 put has a bid of $8.10 with an implied volatility of 32%, and the August $125 call comes "
    "in at $9.10 with a premium of 13.75%.",
    "Ticker A is leading the tech stock rally, trading at all-time highs!",
]

VOCAB_SIZE = 90

def manual_tfidf(counts, idf):
    tfidf = []
    total = sum(counts) or 1
    total_dec = Decimal(total)
    for idx, cnt in enumerate(counts):
        tfidf.append((Decimal(cnt) / total_dec) * idf[idx])
    return tfidf


def cosine_from_vectors(vectors):
    sims = np.zeros((len(vectors), len(vectors)))
    dense = np.array(vectors, dtype=float)
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            vi = dense[i]
            vj = dense[j]
            denom = np.linalg.norm(vi) * np.linalg.norm(vj)
            sims[i, j] = float(np.dot(vi, vj) / denom) if denom else 0.0
    return sims

def to_signed(value: int) -> int:
    half = FIELD_MODULUS // 2
    if value > half:
        return value - FIELD_MODULUS
    return value


def main():
    vocabulary = extract_vocabulary(TEXTS, target_size=VOCAB_SIZE)
    builder = TruthFinderWitnessBuilder(vocabulary)
    documents = [builder.tokenize(text) for text in TEXTS]
    witness = builder.build(TEXTS)

    idf_decimals = [Decimal(value) / Decimal("1e8") for value in witness.idf]

    print("=== TF-IDF check ===")
    for idx, counts in enumerate(documents):
        expected = manual_tfidf(counts, idf_decimals)
        actual = [Decimal(val) / Decimal("1e8") for val in witness.tfidf_vectors[idx]]
        print(f"Document {idx} counts: {counts}")
        print(f"Expected TF-IDF: {expected}")
        print(f"Actual TF-IDF  : {actual}")
        assert all(abs(e - a) <= Decimal("1e-6") for e, a in zip(expected, actual))

    print("\n=== Cosine similarity check ===")
    tfidf_vectors = [
        [Decimal(val) / Decimal("1e8") for val in vec]
        for vec in witness.tfidf_vectors
    ]
    sims = cosine_from_vectors(tfidf_vectors)
    for i in range(len(TEXTS)):
        for j in range(len(TEXTS)):
            cos_value = sims[i, j]
            mapped = max(-1.0, min(1.0, 2 * (cos_value ** 2) - 1))
            signed = to_signed(witness.similarities[i][j])
            actual = Decimal(signed) / Decimal("1e8")
            print(f"Sim[{i}][{j}] -> cos={cos_value:.6f}, mapped={mapped:.6f}, witness={actual}")
            assert abs(Decimal(mapped) - actual) <= Decimal("1e-6")

    print("\n=== TruthFinder iteration check ===")
    sims_dec = [
        [Decimal(to_signed(val)) / Decimal("1e8") for val in row]
        for row in witness.similarities
    ]
    gamma = Decimal(witness.gamma) / Decimal("1e8")
    rho = Decimal(witness.rho) / Decimal("1e8")

    confidence = [Decimal(0)] * len(TEXTS)
    trust = [
        Decimal(to_signed(val)) / Decimal("1e8")
        for val in witness.trust_init
    ]

    def sigmoid(x):
        return Decimal("0.5") + x / Decimal(4) - (x**3) / Decimal(48)

    for _ in range(5):
        new_conf = []
        for i in range(len(TEXTS)):
            support = Decimal(0)
            for j in range(len(TEXTS)):
                if i == j:
                    continue
                support += confidence[j] * sims_dec[j][i]
            raw = trust[i] + rho * support
            new_conf.append(sigmoid(gamma * raw))
        confidence = new_conf
        trust = confidence[:]

    actual_conf = [Decimal(val) / Decimal("1e8") for val in witness.fact_confidence]
    actual_trust = [Decimal(val) / Decimal("1e8") for val in witness.trustworthiness]
    print("Expected confidence:", confidence)
    print("Actual confidence  :", actual_conf)
    print("Expected trust     :", trust)
    print("Actual trust       :", actual_trust)
    tolerance = Decimal("5e-4")
    for e, a in zip(confidence, actual_conf):
        assert abs(e - a) <= tolerance
    for e, a in zip(trust, actual_trust):
        assert abs(e - a) <= tolerance


if __name__ == "__main__":
    main()
VOCAB_SIZE = 90
