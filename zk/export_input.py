"""Convert witness builder output into Circom input JSON."""

from __future__ import annotations

import json
from pathlib import Path
from decimal import Decimal

from witness_builder import TruthFinderWitnessBuilder, extract_vocabulary

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

OUTPUT_PATH = Path(__file__).resolve().parent / "truthfinder_input.json"
SCALE = Decimal("1e8")
VOCAB_SIZE = 90


def modinv(x, p):
    return pow(x % p, p - 2, p)

def main():
    vocab = extract_vocabulary(TEXTS, target_size=VOCAB_SIZE)
    builder = TruthFinderWitnessBuilder(vocab)

    # ======== bn128 field / fixed scale ========
    P = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    SCALE_INT = 100000000  # 必须和 circom 输入 scale 一致

    # 1) counts / totals / inv_total (bn128)
    counts_list = []
    totals = []
    inv_totals = []
    for text in TEXTS:
        counts = builder.tokenize(text)
        total = sum(counts)
        if total == 0:
            raise ValueError("total_count is 0; please extend vocabulary to cover all documents.")
        counts_list.append(counts)
        totals.append(total)
        inv_totals.append(modinv(total, P))

    # 2) idf (still computed offchain), then scale to int (same SCALE)
    idf_dec = builder._compute_idf(counts_list)   # Decimal list
    idf_int = [int((x * Decimal(SCALE_INT)).to_integral_value()) % P for x in idf_dec]

    # 3) tfidf_int in FIELD: tfidf[k] = counts[k] * idf[k] * inv_total
    tfidf = []
    for i in range(len(TEXTS)):
        inv_t = inv_totals[i]
        vec = [ (counts_list[i][k] * idf_int[k]) % P * inv_t % P for k in range(len(vocab)) ]
        tfidf.append(vec)

    # 4) cosine^2 mapped matrix: similarities[i][j]
    n = len(TEXTS)
    similarities = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dot = 0
            na2 = 0
            nb2 = 0
            for k in range(len(vocab)):
                a = tfidf[i][k]
                b = tfidf[j][k]
                dot = (dot + a*b) % P
                na2 = (na2 + a*a) % P
                nb2 = (nb2 + b*b) % P
            denom2 = (na2 * nb2) % P
            dot2 = (dot * dot) % P
            if denom2 == 0:
                cos2_scaled = 0
            else:
                cos2_scaled = (dot2 * SCALE_INT) % P * modinv(denom2, P) % P
            similarities[i][j] = (2 * cos2_scaled - SCALE_INT) % P

    # 5) assemble circom input JSON
    input_data = {
        "gamma": str(int(Decimal("0.8") * Decimal(SCALE_INT))),  # 你也可以继续用 witness.gamma 但必须一致
        "rho": str(int(Decimal("0.6") * Decimal(SCALE_INT))),
        "idf": [str(x) for x in idf_int],
        "counts": [[str(c) for c in row] for row in counts_list],
        "total_counts": [str(x) for x in totals],
        "inv_total_counts": [str(x) for x in inv_totals],
        "similarities": [[str(x) for x in row] for row in similarities],
        "half_const": str(int(Decimal(SCALE_INT) * Decimal("0.5"))),
        "quarter_const": str(int(Decimal(SCALE_INT) * Decimal("0.25"))),
        "inv48_const": str(int(Decimal(SCALE_INT) / Decimal(48))),
        "scale": str(SCALE_INT),
        "trust_init": [str(int(Decimal("0.9") * Decimal(SCALE_INT)))] * len(TEXTS),
    }

    OUTPUT_PATH.write_text(json.dumps(input_data, indent=2))
    print(f"Wrote Circom input to {OUTPUT_PATH}")



if __name__ == "__main__":
    main()
