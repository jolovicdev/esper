# ESPER PoC Run Report

## Scenario A: Knowledge Update (Contradiction Resolution)

Score blend used in `poc.py`:

$$
\mathrm{score}_{esper}(q,o)=0.4\cdot\frac{1}{1+d(q,o)}+0.6\cdot P_{net}(o)
$$


Question: `What is the current production codename?`

Flat retrieval:
- `#1 codename_borealis sim=0.605 score=0.605`
- `#2 codename_atlas sim=0.567 score=0.567`
- Agent answer: `Borealis` (stale)

ESPER retrieval:
- `#1 codename_atlas sim=0.567 belief=0.920 score=0.779`
- Agent answer: `Atlas` (updated)

Belief state:
- `belief(codename_borealis)=0.143`
- `belief(codename_atlas)=0.920`

Interpretation:
- Flat RAG follows pure semantic similarity and picks stale memory.
- ESPER applies epistemic state (update-aware belief), so retrieval aligns with latest trusted update.

## Scenario B: Correlated vs Independent Reinforcement
Inputs are the same confidences `[0.85, 0.88, 0.90]` with different episode structure.

Results:
- Correlated (same episode): `reinforcement_count=1`, `estimate=0.684`
- Independent (distinct episodes): `reinforcement_count=3`, `estimate=0.958`

Interpretation:
- ESPER correctly avoids overcounting repeated same-episode claims.
- Independent confirmations are valued higher than correlated repetition.

In short: ESPER resolves contradictions and deduplicates correlated evidence at retrieval time — problems flat vector search ignores entirely.

## Limits
- This is a PoC, not a full benchmark study.
- Constants and blend weights still need broader ablations on larger datasets.
- Domain-level impact must be validated with real task metrics (accuracy, reversals avoided, harm reduction proxies).
