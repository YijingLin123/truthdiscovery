"""Simple TruthFinder example adapted to oracle price quotes."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import List

import pandas as pd

from truthdiscovery import TruthFinder


def build_dataframe(quotes: List[dict]) -> pd.DataFrame:
    rows = []
    for q in quotes:
        rows.append(
            {
                "website": q["oracle"],
                "fact": str(q["price"]),
                "object": q.get("asset", "asset"),
            }
        )
    return pd.DataFrame(rows, columns=["website", "fact", "object"])


def price_implication_factory(tolerance: Decimal):
    tol = max(tolerance, Decimal("0.01"))

    def implication(f1, f2):
        try:
            p1 = Decimal(str(f1))
            p2 = Decimal(str(f2))
        except InvalidOperation:
            return 0.0
        diff = abs(p1 - p2)
        if diff == 0:
            return 1.0
        if diff >= tol:
            return 0.0
        return float((tol - diff) / tol)

    return implication


def filter_by_support(df: pd.DataFrame, tolerance: Decimal, min_support: int):
    """Drop facts without at least min_support corroborating quotes inside tolerance."""
    if min_support <= 1:
        return df.copy(), df.iloc[0:0].copy()

    tol = tolerance if isinstance(tolerance, Decimal) else Decimal(str(tolerance))
    df = df.copy()
    df["_support_count"] = 0

    for idx, row in df.iterrows():
        same_object = df[df["object"] == row["object"]]
        try:
            price_i = Decimal(str(row["fact"]))
        except InvalidOperation:
            continue
        count = 0
        for _, other in same_object.iterrows():
            try:
                price_j = Decimal(str(other["fact"]))
            except InvalidOperation:
                continue
            if abs(price_i - price_j) <= tol:
                count += 1
        df.at[idx, "_support_count"] = count

    mask = df["_support_count"] >= min_support
    kept = df.loc[mask].drop(columns="_support_count")
    dropped = df.loc[~mask].drop(columns="_support_count")
    return kept, dropped


def run_truthfinder(quotes: List[dict],
                    tolerance: Decimal = Decimal("50"),
                    gamma: float = 0.8,
                    rho: float = 0.6,
                    min_support: int = 1) -> pd.DataFrame:
    df = build_dataframe(quotes)
    filtered, dropped = filter_by_support(df, tolerance, min_support)
    if filtered.empty:
        raise ValueError("No price quotes satisfy the min_support constraint.")
    finder = TruthFinder(
        price_implication_factory(tolerance),
        dampening_factor=gamma,
        influence_related=rho,
    )
    result = finder.train(filtered.copy())
    ordered = result.sort_values("fact_confidence", ascending=False).reset_index(drop=True)
    if not dropped.empty:
        dropped = dropped.assign(trustworthiness=0.0, fact_confidence=0.0)
        ordered = pd.concat([ordered, dropped], ignore_index=True)
    return ordered


if __name__ == "__main__":
    # price_quotes = [
    #     {"oracle": "oracle_a", "asset": "BTC", "price": "64230.98"},
    #     {"oracle": "oracle_a", "asset": "ETH", "price": "3198.75"},
    #     {"oracle": "oracle_b", "asset": "BTC", "price": "64230.12"},
    #     {"oracle": "oracle_b", "asset": "ETH", "price": "3199.12"},
    #     {"oracle": "oracle_c", "asset": "BTC", "price": "64233.98"},
    #     {"oracle": "oracle_c", "asset": "ETH", "price": "3201.11"},
    #     {"oracle": "oracle_d", "asset": "BTC", "price": "61025.55"},
    #     {"oracle": "oracle_d", "asset": "ETH", "price": "2902.45"},
    #     {"oracle": "oracle_e", "asset": "BTC", "price": "64231.41"},
    #     {"oracle": "oracle_e", "asset": "ETH", "price": "3197.88"},
    #     {"oracle": "oracle_f", "asset": "BTC", "price": "61025.33"},
    #     {"oracle": "oracle_f", "asset": "ETH", "price": "2999.56"},
    #     {"oracle": "oracle_g", "asset": "BTC", "price": "61024.03"},
    #     {"oracle": "oracle_g", "asset": "ETH", "price": "2905.90"},
    # ]
    price_quotes = [
        {"oracle": "oracle_a", "asset": "BTC", "price": "64230.98"},
        {"oracle": "oracle_b", "asset": "BTC", "price": "64230.12"},
        {"oracle": "oracle_c", "asset": "BTC", "price": "64233.98"},
        {"oracle": "oracle_d", "asset": "BTC", "price": "61025.55"},
        {"oracle": "oracle_e", "asset": "BTC", "price": "64231.41"},
        {"oracle": "oracle_f", "asset": "BTC", "price": "61025.33"},
        {"oracle": "oracle_g", "asset": "BTC", "price": "61024.03"},
    ]
    print("Initial quotes\n", build_dataframe(price_quotes))
    scored = run_truthfinder(
        price_quotes,
        tolerance=Decimal("50"),
        min_support=4,
    )
    print("\nTruthFinder result\n", scored)
