# Sequential Importance Sampling for Contingency Tables

Numba-accelerated sequential importance sampling (SIS) for uniformly sampling contingency tables with fixed row and column margins, following **Chen, Diaconis, Holmes, and Liu (2005, JASA)**.

## Installation

```bash
pip install -e .
```

## Usage

```python
import numpy as np
from src.sequential_importance_sampling import sample_tables_sis_parallel

# Row and column margins from Chen et al. (2005)
row_sums = np.array([10, 62, 13, 11, 39])
col_sums = np.array([65, 25, 45])

# Draw 150,000 importance-weighted tables
tables, logq = sample_tables_sis_parallel(
    row_sums[:, None, None],
    col_sums[None, :, None],
    J=1, K=4, L=3, T=1,
    num_samples=150_000,
    rng_seed=44042,
)

# Estimate the number of tables with these margins
# True value: 239,382,173 (Diaconis and Gangolli, 1995)
print("Estimate:", np.exp(-logq).mean())
```

## API

| Function | Description |
|---|---|
| `sample_table_sis` | Sample a single contingency table via column-wise SIS |
| `sample_tables_sis` | Sample multiple tables (sequential) |
| `sample_tables_sis_parallel` | Sample multiple tables (parallel via `numba.prange`) |
| `sample_tables_sis_parallel_singleperiod` | Parallel sampling, single time-period variant |

All samplers return `(tables, logq)` where `logq` contains the log importance weights.

## Tests

Run the Diaconis-Gangolli counting verification:

```bash
python tests/test_diaconis_gangolli.py
```

Or with pytest:

```bash
pytest tests/
```

## References

- Chen, Y., Diaconis, P., Holmes, S. P., & Liu, J. S. (2005). Sequential Monte Carlo methods for statistical analysis of tables. *Journal of the American Statistical Association*, 100(469), 109-120.
- Diaconis, P., & Gangolli, A. (1995). Rectangular arrays with fixed margins. In *Discrete Probability and Algorithms* (pp. 15-41). Springer.
