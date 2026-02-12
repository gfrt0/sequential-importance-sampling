"""
Reproduce the counting result from Holmes and Jones (1996):

A 5x4 contingency table with row sums [9, 49, 182, 478, 551]
and column sums [9, 309, 355, 596].

Chen et al. (2005, JASA) estimate the total number of tables
as approximately 3.383 x 10^16 via SIS (Section 6.4).
"""

import numpy as np
from sequential_importance_sampling import sample_tables


def test_holmes_jones():
    # Chen et al. (2005) SIS estimate: ~3.383 x 10^16
    log10_reference = 16 + np.log10(3.383)

    num_samples = 1_000_000

    row_sums = np.array([9, 49, 182, 478, 551])
    col_sums = np.array([9, 309, 355, 596])

    _, logq = sample_tables(
        row_sums, col_sums,
        num_samples=num_samples,
        rng_seed=44042,
    )

    inv_w = np.exp(-logq)
    estimate = inv_w.mean()
    log10_estimate = np.log10(estimate)
    cv = inv_w.std() / inv_w.mean()
    ess = num_samples / (1 + cv ** 2)

    print(f"Reference (Chen et al.):  3.383e16")
    print(f"SIS estimate (n=1M):     {estimate:.3e}")
    print(f"log10 estimate:          {log10_estimate:.4f}")
    print(f"log10 reference:         {log10_reference:.4f}")
    print(f"Coefficient of variation: {cv:.4f}")
    print(f"Effective sample size:   {ess:,.0f}")

    # Allow 1% relative error on log10 scale
    assert abs(log10_estimate - log10_reference) / log10_reference < 0.01, (
        f"log10 estimate {log10_estimate:.4f} too far from reference {log10_reference:.4f}"
    )


if __name__ == "__main__":
    test_holmes_jones()
