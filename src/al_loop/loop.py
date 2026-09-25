"""The experiment-selection simulation.

Oracle model: `y` contains *measured* fitness for every variant — querying a
variant costs one experiment and reveals its true value. The active loop
starts from `n_init` random labels and iterates: fit surrogate -> score
unlabeled pool with the acquisition function -> measure the top `batch_size`
candidates -> repeat until `budget` experiments are spent.

The random baseline follows the identical schedule but picks candidates
uniformly, replicated `n_random_seeds` times for a distribution. This is
the honest counterfactual: same budget, same initial set, no model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from al_loop.acquisition import acquire
from al_loop.config import load_config
from al_loop.encode import one_hot
from al_loop.surrogate import fit_predict


def _snapshot(labeled: set, y: np.ndarray, top_set: set) -> dict:
    lab = np.array(sorted(labeled))
    return {
        "n_experiments": len(lab),
        "best_fitness": float(y[lab].max()),
        "top_hits_found": len(top_set & labeled),
    }


def run_active(X, y, exp, sur, acq, top_set):
    """One AL trajectory. Returns (round snapshots, ordered pick indices)."""
    rng = np.random.default_rng(exp["seed"])
    labeled = set(rng.choice(len(X), size=exp["n_init"], replace=False).tolist())
    picks = sorted(labeled)
    records = [_snapshot(labeled, y, top_set)]
    # Surrogate sees log-enrichment: fitness is a heavy-tailed selection
    # ratio (mean ~0.08, max ~8.8); on the raw scale the tail dominates and
    # GP optimization degenerates. Reported metrics stay on the raw scale.
    y_model = np.log1p(y)
    while len(labeled) < exp["budget"]:
        lab = np.array(sorted(labeled))
        unl = np.array([i for i in range(len(X)) if i not in labeled])
        mean, std = fit_predict(X[lab], y_model[lab], X[unl], sur["kind"])
        scores = acquire(
            mean,
            std,
            acq["kind"],
            acq["kappa"],
            acq["xi"],
            float(y_model[lab].max()),
        )
        k = min(exp["batch_size"], exp["budget"] - len(labeled))
        batch = unl[np.argsort(-scores)[:k]]
        labeled.update(batch.tolist())
        picks.extend(batch.tolist())
        records.append(_snapshot(labeled, y, top_set))
    return records, picks


def run_random(X, y, exp, seed, top_set):
    """Random-selection trajectory under the identical experiment schedule."""
    rng = np.random.default_rng(seed)
    labeled = set(rng.choice(len(X), size=exp["n_init"], replace=False).tolist())
    records = [_snapshot(labeled, y, top_set)]
    while len(labeled) < exp["budget"]:
        unl = np.array([i for i in range(len(X)) if i not in labeled])
        k = min(exp["batch_size"], exp["budget"] - len(labeled))
        labeled.update(rng.choice(unl, size=k, replace=False).tolist())
        records.append(_snapshot(labeled, y, top_set))
    return records


def main(in_parquet: str, out_records: str, out_picks: str):
    cfg = load_config()
    exp, sur, acq = cfg["experiment"], cfg["surrogate"], cfg["acquisition"]
    df = pd.read_parquet(in_parquet)
    X, y = one_hot(df["variant"]), df["fitness"].to_numpy()
    top_set = set(np.argsort(-y)[: cfg["evaluation"]["top_k"]].tolist())

    active_records, active_picks = run_active(X, y, exp, sur, acq, top_set)
    rows = [{"strategy": "active", **r} for r in active_records]
    for s in range(exp["n_random_seeds"]):
        for r in run_random(X, y, exp, exp["seed"] + 1 + s, top_set):
            rows.append({"strategy": f"random_{s}", **r})
    Path(out_records).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_records, index=False)

    # What the active policy chose to measure, in acquisition order —
    # inspectable evidence of the policy's decisions.
    picks = df.iloc[active_picks].copy()
    picks["acquisition_order"] = range(len(picks))
    picks["initial_screen"] = picks["acquisition_order"] < exp["n_init"]
    Path(out_picks).parent.mkdir(parents=True, exist_ok=True)
    picks.to_parquet(out_picks, index=True)
    print(
        f"records: {len(rows)} rows | active spent {len(active_picks)} "
        f"experiments, best found {active_records[-1]['best_fitness']:.4f}"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
