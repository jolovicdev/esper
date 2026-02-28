# ESPER PoC

## Overview

ESPER stands for **Event-Sourced Projected Epistemic Resolver**.

This PoC implements a belief layer for memory systems:
- ingest immutable evidence events,
- project them into query-time belief state,
- resolve updates/contradictions before LLM retrieval.

The repository contains:
- core engine: `esper.py`
- live demo: `poc.py`
- validation tests: `tests/test_esper_validation.py`

## Evidence Model

Each event is:

$$
e_i = (u, s, p, o, \pi, c, \epsilon, t, \tau)
$$

where:
- $u$ = user id
- $s$ = subject
- $p$ = predicate
- $o$ = object
- $\pi \in \{+1,-1\}$ = polarity
- $c \in [0,1]$ = calibrated confidence
- $\epsilon$ = episode id
- $t$ = event timestamp
- $\tau$ = source type

Source reliability weight:

$$
w(\tau)\in(0,1]
$$

with default mapping:
- TOOL $=1.0$
- EXPLICIT $=1.0$
- RULE $=0.8$
- CLASSIFIER $=0.6$
- EXTRACTOR $=0.5$

## Temporal Decay

For event age $\Delta t_i$ and decay parameter $\lambda \ge 0$:

$$
\tilde{c_i} = \mathrm{clip}\left(c_i e^{-\lambda \Delta t_i},\,0,\,1-\varepsilon\right)
$$

If $\lambda=0$, decay is disabled.

## MANY Cardinality Projection

For each $(u,s,p,o,\epsilon)$ we max-pool within episode:

$$
p_\epsilon = \max_{\pi_i=+1}\tilde{c_i},\quad
n_\epsilon = \max_{\pi_i=-1}\tilde{c_i}
$$

Positive and negative weighted Noisy-OR accumulators across episodes:

$$
L^+ = \sum_{\epsilon} w_\epsilon \ln(1-p_\epsilon),\quad
L^- = \sum_{\epsilon} w_\epsilon \ln(1-n_\epsilon)
$$

$$
P^+ = 1-e^{L^+},\quad
P^- = 1-e^{L^-}
$$

Final projected belief:

$$
P_{net} = \mathrm{clip}\left(\max(0, P^+ - P^-),\,0,\,1-\varepsilon\right)
$$

Implication:
- repeated claims in the same episode do **not** add independent reinforcement,
- negative evidence reduces belief explicitly.

## ONE Cardinality Projection

For a new positive candidate with score $s_{new}$ and incumbent $s_{old}$:

$$
s'_{old} = \mathrm{clip}\left(
s_{old} \cdot
\exp\left(-\kappa s_{new} e^{-\Delta t/\tau_{ow}}\right),
0,1-\varepsilon\right)
$$

and the new candidate is inserted/updated at $s_{new}$.

ONE semantics intentionally differ from MANY: it uses overwrite suppression between competing objects, not the MANY Noisy-OR episode accumulators.
For replay and backfill safety, overwrite recency is computed from event timestamps (new event time vs incumbent last update time), not wall-clock ingestion time.

Negative polarity on ONE suppresses current belief:

$$
s' = \mathrm{clip}(s(1-s_{neg}),\,0,\,1-\varepsilon)
$$

Ambiguity rule for top two candidates $s_1, s_2$:

$$
\mathrm{ambiguous} \iff s_1 - s_2 < m
$$

with $m$ set by `ambiguity_margin`.
When ambiguous, both top candidates are flagged as ambiguous.

## Retrieval in `poc.py`

Embeddings are stored in local SQLite + `sqlite-vec` files:
- `.blackgeorge/poc_flat_vec.db`
- `.blackgeorge/poc_esper_vec.db`

Schema:
- `docs(doc_id, text)`
- `vec_docs` virtual table (`vec0`) with dense embedding vectors

Flat retrieval score:

$$
\mathrm{score}_{flat}(q,o)=\frac{1}{1+d(q,o)}
$$

where $d(q,o)$ is the `sqlite-vec` nearest-neighbor distance.

ESPER retrieval score (over active contenders):

$$
\mathrm{score}_{esper}(q,o)=\alpha\frac{1}{1+d(q,o)}+(1-\alpha)P_{net}(o)
$$

Current demo setting:

$$
\alpha=0.4,\quad 1-\alpha=0.6
$$

## What This PoC Demonstrates

From `poc.py`:

1. Knowledge update case:
- flat retrieval can return stale Borealis,
- ESPER returns updated Atlas after overwrite projection.

2. Correlated reinforcement case:
- same 3 confidence values:
  - same episode $\rightarrow$ `reinforcement_count = 1`, lower belief,
  - separate episodes $\rightarrow$ `reinforcement_count = 3`, higher belief.

## What This PoC Does Not Prove

This PoC is a **mechanism demonstration**, not a benchmark paper result.

Not established here:
- broad generalization across domains,
- optimality of constants/weights,
- cross-provider robustness.

Those require held-out datasets, larger ablations, and controlled evaluation protocols.

## Runtime Parameters

`ESPERDatabase(...)` exposes:
- `soft_overwrite_kappa`
- `soft_overwrite_tau_seconds`
- `ambiguity_margin`

`insert_evidence(...)` exposes:
- `decay_lambda`

## Quickstart

Install:

```bash
uv sync
```

Run tests:

```bash
uv run pytest -q
```

Run live demo:

```bash
uv run python poc.py
```

`poc.py` auto-loads environment variables from `.env` via `python-dotenv`.

In this case I used OPENROUTER_API_KEY and openrouter/*model. 

You can use any model for embeddings, LLM -> blackgeorge works on LITELLM.
