


# Circom Modules for TruthFinder zk Pipeline

This folder will gradually host the Circom templates that mirror each stage of
`truthfinder_agent.py`. The first module implements TF-IDF constraints for a
fixed vocabulary.

## `tfidf.circom`

`TFIDFVector(VOCAB_SIZE)` takes:

- `counts[VOCAB_SIZE]`: raw token counts for a document (private witness).
- `total_count`: sum of all counts (either provided or constrained to the sum).
- `idf[VOCAB_SIZE]`: scaled IDF constants (public input or constant).

It outputs `tfidf[VOCAB_SIZE]` such that

```
tfidf[i] * total_count == counts[i] * idf[i]
```

which encodes `tfidf = (count / total) * idf` using field arithmetic.

### Example (Circom CLI)

```bash
circom tfidf.circom --r1cs --wasm --prime bn128
```

Then use a witness file like:

```json
{
  "counts": ["2", "1", "0", "0"],
  "total_count": "3",
  "idf": ["100000000", "120000000", "90000000", "80000000"]
}
```

where IDF values are scaled by `1e8` (matching `zk/witness_builder.py`). The
resulting `tfidf` outputs can feed into later circuits for cosine similarity
and TruthFinder iterations.

## `cosine.circom`

Defines three templates:

- `DotProduct(VOCAB_SIZE)` – constrained sum of element-wise products.
- `VectorNorm(VOCAB_SIZE)` – computes squared norm of a vector.
- `CosineSimilaritySq(VOCAB_SIZE)` – enforces `(sim + scale) * (||a||^2·||b||^2) = 2 * scale * dot^2`,
  which corresponds to `sim = 2 * cos^2 - 1` with all quantities kept in the field.

The cosine gadget assumes `vectorA` and `vectorB` are already scaled integers
matching the TF-IDF outputs and that `sim` uses the same `1e8` scaling as the
witness builder.

Next steps:
1. Encode the TruthFinder recurrence equations.
2. Wrap everything in a top-level circuit that exposes the final ranking as a
   public output and commits to the raw summaries via Merkle roots.

## `truthfinder.circom`

Adds:

- `SigmoidApprox` – polynomial approximation (`0.5 + x/4 - x^3/48`) with scaling
  constants passed as inputs (`half_const`, `quarter_const`, `inv48_const`).
- `TruthFinderIteration(N)` – uses the similarity matrix and current trust
  vector to produce updated `confidence` and `trust_out`. It reuses the sigmoid
  gadget and assumes all inputs are scaled integers (e.g., same 1e8 factor as
  `witness_builder.py`).

Next step: stitch TF-IDF, cosine, and TruthFinder templates into a top-level
 circuit that exposes commitments/public outputs for on-chain verification.

## `truthfinder_top.circom`

`TruthFinderTop(N, VOCAB_SIZE)` demonstrates how to wire the three modules:

- Takes term counts for each summary, IDF constants, similarity matrix, TruthFinder
  parameters, and the initial trust vector as inputs.
- Instantiates `TFIDFVector` per summary, then `CosineSimilaritySq` for each pair.
- Reuses a 5-step `TruthFinderIteration` loop that feeds the checked similarity
  matrix into successive iterations to obtain final trust scores (`trust_final`).

This template assumes the counts/norms/similarities already come from the same
scaling scheme as `zk/witness_builder.py`. In a production circuit, you would
replace the simple `trust_init` seed with either public inputs or additional
constraints (e.g., fix an initial trust vector).
