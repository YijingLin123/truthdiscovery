"""TruthFinder demo using deterministic summaries with zk-aligned math."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from field import FIELD_MODULUS
from witness_builder import TruthFinderWitnessBuilder, extract_vocabulary

AGENT_SUMMARIES = [
    (
        "Agent_1",
        "A new Agilent Technologies, Inc. (Symbol: A) option chain has opened with a $125 put "
        "priced at $4.50 and a $150 call at $8.10, offering opportunities for premium capturing "
        "via selling puts or opening covered calls respectively.",
    ),
    (
        "Agent_2",
        "Newly trading August 2024 options for Agilent Technologies Inc., including a $125 put "
        "with $4 premium and a $150 call at $8.10, offer opportunities to sellers of puts or "
        "calls for higher premiums due to upcoming expiration, while noting slightly lower "
        "implied volatilities compared to similar contracts.",
    ),
    (
        "Agent_3",
        "A recent round of Agilent Technologies options, trading at 241 days until expiry, now "
        "offers potential upside for option sellers and lower cost entry points for call buyers. "
        "The August $150 put has a bid of $8.10 with an implied volatility of 32%, and the August "
        "$125 call comes in at $9.10 with a premium of 13.75%.",
    ),
    (
        "Agent_4",
        "Ticker A is leading the tech stock rally, trading at all-time highs!",
    ),
]

VOCAB_SIZE = 90


def build_records():
    rows = []
    for website, summary in AGENT_SUMMARIES:
        rows.append(
            {
                "website": website,
                "fact": summary,
                "object": "A",
            }
        )
    return rows


def _to_decimal(value: int) -> Decimal:
    half = FIELD_MODULUS // 2
    if value > half:
        value -= FIELD_MODULUS
    return Decimal(value) / Decimal("1e8")


def run_truthfinder(records, gamma: float = 0.8, rho: float = 0.6):
    texts = [row["fact"] for row in records]
    vocabulary = extract_vocabulary(texts, target_size=VOCAB_SIZE)
    builder = TruthFinderWitnessBuilder(vocabulary)
    witness = builder.build(texts, gamma=gamma, rho=rho)
    scored = []
    for row, conf, trust in zip(records, witness.fact_confidence, witness.trustworthiness):
        scored.append(
            {
                **row,
                "fact_confidence": float(_to_decimal(conf)),
                "trustworthiness": float(_to_decimal(trust)),
            }
        )
    return sorted(scored, key=lambda r: r["fact_confidence"], reverse=True)


def main():
    records = build_records()
    print("Agent summaries:")
    for row in records:
        print(f"- {row['website']}: {row['fact']}")

    scored = run_truthfinder(records)
    print("\nTruthFinder scoring result:")
    for row in scored:
        print(
            f"{row['website']}: fact_confidence={row['fact_confidence']:.6f}, "
            f"trustworthiness={row['trustworthiness']:.6f}"
        )


if __name__ == "__main__":
    main()
