"""Run four LLM agents to summarize a news article, then rank summaries via TruthFinder."""

from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer

from truthdiscovery import TruthFinder

try:
    from qwen_agent.agents import Assistant
except ImportError as exc:  # pragma: no cover
    Assistant = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


NEWS_TICKER = "A"
NEWS_BODY = (
    "Investors in Agilent Technologies, Inc. (Symbol: A) saw new options begin trading this "
    "week, for the August 2024 expiration. One of the key inputs that goes into the price an "
    "option buyer is willing to pay, is the time value, so with 241 days until expiration the "
    "newly trading contracts represent a possible opportunity for sellers of puts or calls to "
    "achieve a higher premium than would be available for the contracts with a closer expiration. "
    "At Stock Options Channel, our YieldBoost formula has looked up and down the A options chain "
    "for the new August 2024 contracts and identified one put and one call contract of particular "
    "interest. The put contract at the $125.00 strike price has a current bid of $4.50. If an "
    "investor was to sell-to-open that put contract, they are committing to purchase the stock at "
    "$125.00, but will also collect the premium, putting the cost basis of the shares at $120.50. "
    "Turning to the calls side of the option chain, the call contract at the $150.00 strike price "
    "has a current bid of $8.10. If an investor was to purchase shares of A stock at the current "
    "price level of $138.99/share, and then sell-to-open that call contract as a covered call, "
    "they are committing to sell the stock at $150.00. Considering the call seller will also collect "
    "the premium, that would drive a total return of 13.75% if the stock gets called away at the "
    "August 2024 expiration. The implied volatility in the put contract example is 32%, while the "
    "implied volatility in the call contract example is 29%."
)

LLM_CFG = {
    "model": "qwen2.5:1.5b",
    "model_server": "http://127.0.0.1:11434/v1",
    "api_key": "ollama",
}


def init_summary_agent(name: str) -> Assistant:
    if Assistant is None:  # pragma: no cover
        raise RuntimeError(
            f"qwen_agent is required but not installed: {IMPORT_ERROR}"
        )
    system = (
        f"You are {name}, a financial news summarization node.\n"
        f"- You will receive a news article about ticker {NEWS_TICKER}.\n"
        f"- Respond with a single concise English sentence, starting with `{NEWS_TICKER}`.\n"
        f"- Keep it under 25 words and avoid extra commentary.\n"
    )
    return Assistant(
        llm=LLM_CFG,
        name=name,
        description=f"{name} 股票摘要节点",
        system_message=system,
    )


def init_noisy_agent(name: str) -> Assistant:
    if Assistant is None:  # pragma: no cover
        raise RuntimeError(
            f"qwen_agent is required but not installed: {IMPORT_ERROR}"
        )
    system = (
        f"You are {name}, a noisy ticker commentator.\n"
        f"- Produce a short English sentence mentioning ticker {NEWS_TICKER}.\n"
        f"- The sentence should sound market-related but does not need to summarize news.\n"
        f"- Keep it under 20 words and vary the wording each time.\n"
    )
    return Assistant(
        llm=LLM_CFG,
        name=name,
        description=f"{name} 噪声节点",
        system_message=system,
    )


def _extract_assistant_text(messages: Optional[List[dict]]) -> Optional[str]:
    if not messages:
        return None
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"].strip()
    return None


def run_summary_agent(agent: Assistant, prompt_text: str, max_retries: int = 2) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt_text,
        }
    ]
    for _ in range(max_retries):
        last_response: Optional[List[dict]] = None
        for response in agent.run(messages):
            last_response = response
        summary = _extract_assistant_text(last_response)
        if summary:
            return summary
        messages.append({"role": "user", "content": "Please respond with a single English sentence summary."})
    return ""


def generate_agent_summaries(agent_names: List[str]) -> pd.DataFrame:
    records = []
    noisy_agents = {"Agent_4", "Agent_5", "Agent_6", "Agent_7"}
    summary_prompt = (
        "Read the following article and produce one English sentence summary:\n"
        f"{NEWS_BODY}"
    )
    for name in agent_names:
        if name in noisy_agents:
            agent = init_noisy_agent(name)
            summary = run_summary_agent(
                agent,
                f"Produce a random ticker-related remark about {NEWS_TICKER} unrelated to the article.",
            )
        else:
            agent = init_summary_agent(name)
            summary = run_summary_agent(agent, summary_prompt)
        records.append(
            {
                "website": name,
                "fact": summary or f"{NEWS_TICKER} summary generation failed.",
                "object": NEWS_TICKER,
            }
        )
    return pd.DataFrame(records, columns=["website", "fact", "object"])


def build_text_implication(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(min_df=1)
    vectorizer.fit(df["fact"])

    def implication(f1, f2):
        V = vectorizer.transform([f1, f2])
        v1, v2 = np.asarray(V.todense())
        denom = norm(v1) * norm(v2)
        if denom == 0:
            return -1.0
        cos_sim = float(np.dot(v1, v2) / denom)
        # Map cosine similarity [-1, 1] -> [-1, 1] but allow negative influence
        return max(-1.0, min(1.0, 2 * cos_sim - 1))

    def similarity_matrix():
        vectors = vectorizer.transform(df["fact"])
        dense = vectors.todense()
        sims = np.zeros((len(df), len(df)))
        for i in range(len(df)):
            vi = np.asarray(dense[i]).flatten()
            ni = norm(vi)
            for j in range(len(df)):
                vj = np.asarray(dense[j]).flatten()
                nj = norm(vj)
                denom = ni * nj
                sims[i, j] = float(np.dot(vi, vj) / denom) if denom else 0.0
        return sims

    return implication, similarity_matrix


def run_truthfinder(df: pd.DataFrame, gamma: float, rho: float):
    implication, sim_builder = build_text_implication(df)
    finder = TruthFinder(
        implication,
        dampening_factor=gamma,
        influence_related=rho,
    )
    result = finder.train(df.copy())
    ordered = result.sort_values("fact_confidence", ascending=False).reset_index(drop=True)
    return ordered, sim_builder()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score agent-generated summaries with TruthFinder.")
    parser.add_argument("--gamma", type=float, default=0.8, help="TruthFinder dampening factor.")
    parser.add_argument("--rho", type=float, default=0.6, help="TruthFinder influence factor.")
    parser.add_argument("--agents", type=int, default=20, help="Number of agents to spawn.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    agent_names = [f"Agent_{i+1}" for i in range(args.agents)]
    df = generate_agent_summaries(agent_names)
    print("Agent summaries:")
    for _, row in df.iterrows():
        print(f"- {row['website']}: {row['fact']}")

    result, sim_matrix = run_truthfinder(df, gamma=args.gamma, rho=args.rho)
    print("\nTruthFinder scoring result:")
    print(result[["website", "fact_confidence", "trustworthiness"]])

    print("\nTF-IDF cosine similarity matrix:")
    sim_df = pd.DataFrame(sim_matrix, columns=df["website"], index=df["website"])
    print(sim_df)


if __name__ == "__main__":
    main()
