# active-learning-loop

Simulated active-learning experiment selection over a **real, fully measured
protein fitness landscape**: does a surrogate-model-driven loop find
high-fitness variants with fewer experiments than random screening?

**Status: working demonstration.** Snakemake DAG runs fetch -> parse ->
AL-vs-random simulation -> summary report on the GB1 four-site combinatorial
landscape (Wu et al., eLife 2016, via the FLIP mirror): 149,361 variants at
sites V39/D40/G41/V54 with experimentally measured enrichment fitness.

## The question

This is the loop experimental science actually wants closed: label a small
random screen, fit a surrogate, let an acquisition function pick the next
most informative batch, repeat under a fixed budget. The landscape is fully
measured, so the oracle is ground truth — every acquisition decision can be
scored against what the experiment would really have returned.

## Method

- **Surrogate**: Gaussian process (RBF + white noise, fixed hyperparameters,
  normalized targets) over position-wise one-hot features (4 sites x 20 AAs).
  Fit on `log1p(fitness)` — enrichment is heavy-tailed (mean 0.08, max 8.76).
- **Acquisition**: UCB (`mean + kappa*std`, kappa=2); EI and greedy are
  implemented behind config.
- **Schedule**: 96-variant random initial screen, then batches of 24 up to a
  480-experiment budget.
- **Baseline**: random selection on the identical schedule, replicated over
  20 seeds. Same initial distribution, same budget — the honest counterfactual.
- **Metrics**: best-fitness-found curve, true top-100 discovery curve, AUBC
  (area under best curve, oracle-normalized), top-100 hit rate at budget.

## Result (seed 13, committed in `results/summary.json`)

| | Active (UCB-GP) | Random (20 seeds) |
|---|---|---|
| AUBC | **0.465** | 0.373 ± 0.107 |
| best fitness found | 5.77 | 5.06 mean (oracle max 8.76) |
| true top-100 hits at budget | **9** | 0.55 mean |
| acquired variants with fitness > 1.0 | 312/384 (81%) | ~4% of landscape |

Reading it honestly: the policy's *single best find* is only modestly better
than lucky random draws — GB1's top of landscape is spiky and partly
luck-driven. But the policy **concentrates experiments on the functional
region**: ~16x more true top-100 hits discovered, and 81% of chosen
experiments returned fitness > 1.0. If the goal is "learn where the good
variants live" — informative experiment selection — the loop works. If the
goal is one lucky hit, budget matters more than strategy here.

## Debugging trail (kept, it's the point)

The first run reported active *worse* than random (AUBC 0.167, zero top-100
hits, best found frozen at the initial draw). Traced to two failures:

1. Per-round GP kernel optimization degenerated on the spiky landscape
   (bounds-hitting, matmul overflow) -> NaN acquisition scores -> argsort
   silently degenerated to index order, repeatedly picking the first
   unlabeled rows.
2. `length_scale=10` was mismatched to the one-hot metric (variant
   distances are sqrt(2)-sqrt(8)), making candidates nearly indistinguishable.

Fix: fixed hyperparameters matched to the feature metric + log1p target
transform. The failure mode is documented because silent-NaN acquisition is
exactly the kind of bug that produces *plausible-looking* wrong results —
the only tell was the frozen best-fitness curve.

## Caveats

- Single seeded AL trajectory vs a 20-seed random distribution; the AL
  curve is one draw, not a mean. Multi-seed AL replication is the honest
  next step.
- Oracle = noise-free measured values; real experiments add assay noise the
  surrogate would have to absorb.
- One-hot encoding sees sites, not structure; sequence-embedding encoders
  (e.g., ESM-2) are the natural upgrade behind the same interface.
- Greedy top-k batch selection ignores batch diversity.

## Run

```bash
.venv/bin/snakemake -j1          # full DAG
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

`AL_CONFIG` env var selects an alternate config (same convention as the
other repos). Results: `results/summary.json`, `results/curves.png`,
`results/provenance.json`; per-round records and the exact acquisition
order the policy chose are in `data/processed/` (regenerable, gitignored).

## Data

FLIP `splits/gb1/four_mutations_full_data.csv.zip` (CC BY 4.0; extends Wu et
al., eLife 2016 supplement). Downloaded zip is gitignored; the parsed
variant/fitness parquet is a regenerable intermediate.
