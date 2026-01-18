"""Example test showing how to build a zk witness from the provided summaries."""

from __future__ import annotations

import json
from dataclasses import asdict

from witness_builder import TruthFinderWitnessBuilder
from field import FIELD_MODULUS

AGENT_SUMMARIES = [
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

VOCABULARY = [
    "agilent",
    "options",
    "call",
    "put",
    "premium",
    "volatility",
    "trading",
    "ticker",
    "rally",
    "highs",
    "stock",
]


def test_truthfinder_witness_example(tmp_path):
    """Build a witness and dump it for manual inspection."""
    builder = TruthFinderWitnessBuilder(VOCABULARY)
    witness = builder.build(AGENT_SUMMARIES)

    # Basic sanity checks: shapes match inputs and everything sits in the field.
    assert len(witness.tfidf_vectors) == len(AGENT_SUMMARIES)
    assert all(len(vec) == len(VOCABULARY) for vec in witness.tfidf_vectors)
    assert len(witness.similarities) == len(AGENT_SUMMARIES)
    assert len(witness.fact_confidence) == len(AGENT_SUMMARIES)
    assert len(witness.trustworthiness) == len(AGENT_SUMMARIES)
    for row in witness.tfidf_vectors:
        for value in row:
            assert 0 <= value < FIELD_MODULUS

    # Dump to JSON so Circom witnesses can reuse it directly.
    output_path = tmp_path / "truthfinder_witness.json"
    output_path.write_text(json.dumps(asdict(witness), indent=2))
