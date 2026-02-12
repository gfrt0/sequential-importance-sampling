"""
Reproduce the counting result from Diaconis and Gangolli (1995):

The number of 5x3 contingency tables of non-negative integers with
row sums [10, 62, 13, 11, 39] and column sums [65, 25, 45] is
exactly 239,382,173.

We estimate this via sequential importance sampling following
Chen, Diaconis, Holmes, and Liu (2005, JASA).
"""

import numpy as np
from sequential_importance_sampling import sample_tables


def test_diaconis_gangolli():
    true_count = 239_382_173

    row_sums = np.array([10, 62, 13, 11, 39])
    col_sums = np.array([65, 25, 45])

    _, logq = sample_tables(
        row_sums, col_sums,
        num_samples=150_000,
        rng_seed=44042,
    )

    inv_w = np.exp(-logq)
    estimate = inv_w.mean()
    cv = inv_w.std() / inv_w.mean()
    ess = 150_000 / (1 + cv ** 2)

    print(f"True count:              {true_count:,}")
    print(f"SIS estimate (n=150k):   {estimate:,.0f}")
    print(f"Coefficient of variation: {cv:.4f}")
    print(f"Effective sample size:   {ess:,.0f}")

    # Allow 1% relative error
    assert abs(estimate - true_count) / true_count < 0.01, (
        f"Estimate {estimate:,.0f} too far from true count {true_count:,}"
    )


if __name__ == "__main__":
    test_diaconis_gangolli()
