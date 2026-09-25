import numpy as np
import pandas as pd
import pytest

from al_loop.encode import AA_ALPHABET, one_hot


def test_one_hot_shape_and_normalization():
    X = one_hot(pd.Series(["VDGV", "AAAA"]))
    assert X.shape == (2, 4 * len(AA_ALPHABET))
    # each site contributes exactly one active bit
    for site in range(4):
        block = X[:, site * len(AA_ALPHABET) : (site + 1) * len(AA_ALPHABET)]
        assert np.all(block.sum(axis=1) == 1.0)


def test_one_hot_deterministic_and_distinct():
    a = one_hot(pd.Series(["VDGV"]))
    b = one_hot(pd.Series(["VDGV"]))
    c = one_hot(pd.Series(["ADGV"]))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_one_hot_rejects_bad_input():
    with pytest.raises(ValueError, match="non-standard"):
        one_hot(pd.Series(["VDG*"]))
    with pytest.raises(ValueError, match="inconsistent"):
        one_hot(pd.Series(["VDGV", "AAA"]))


def test_one_hot_extended_alphabet_accepts_stop():
    # AAV landscape keeps '*' (stop) as a real measured dead variant
    X = one_hot(pd.Series(["VDGV", "V*GV"]), alphabet=AA_ALPHABET + "*")
    assert X.shape == (2, 4 * 21)
