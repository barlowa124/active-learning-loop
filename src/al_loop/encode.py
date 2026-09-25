"""Variant encoders.

Default: position-wise one-hot over the mutated sites. GB1's library is a
4-site combinatorial design (V39, D40, G41, V54), so one-hot is 4*20 = 80
dims and captures all single-site effects; interaction terms are learnable
by the GP's kernel. An ESM-2 embedding encoder could be added behind the
same interface for landscapes where one-hot underdetermines structure.
"""

import numpy as np
import pandas as pd

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids, fixed order


def one_hot(variants: pd.Series, alphabet: str = AA_ALPHABET) -> np.ndarray:
    """(n_variants, n_sites * len(alphabet)) one-hot over mutated positions."""
    idx = {aa: i for i, aa in enumerate(alphabet)}
    n_sites = len(variants.iloc[0])
    X = np.zeros((len(variants), n_sites * len(alphabet)), dtype=np.float32)
    for i, v in enumerate(variants):
        if len(v) != n_sites:
            raise ValueError(f"inconsistent variant length: {v!r}")
        for site, aa in enumerate(v):
            j = idx.get(aa)
            if j is None:
                raise ValueError(f"non-standard residue {aa!r} in {v!r}")
            X[i, site * len(alphabet) + j] = 1.0
    return X
