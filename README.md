# Sequential Importance Sampling for Contingency Tables

Numba-accelerated sequential importance sampling (SIS) for uniformly sampling **two-way** contingency tables with fixed row and column margins, following **Chen, Diaconis, Holmes, and Liu (2005, JASA)**.

## Installation

```bash
pip install sequential-importance-sampling
```

For development (editable install from source):

```bash
pip install -e .
```

## Usage

```python
import numpy as np
from sequential_importance_sampling import sample_tables

# Row and column margins from Chen et al. (2005)
row_sums = np.array([10, 62, 13, 11, 39])
col_sums = np.array([65, 25, 45])

# Draw 150,000 importance-weighted tables
tables, logq = sample_tables(row_sums, col_sums, num_samples=150_000, rng_seed=44042)

# Estimate the number of tables with these margins
# True value: 239,382,173 (Diaconis and Gangolli, 1995)
print("Estimate:", np.exp(-logq).mean())
```

**Batch dimensions** are supported for sampling many independent two-way tables in parallel. If `row_sums` has shape `(R, d1, d2, ...)` and `col_sums` has shape `(C, d1, d2, ...)`, each combination of batch indices defines an independent R×C table problem. The output `tables` will have shape `(num_samples, d1, d2, ..., R, C)` with batch dims before table dims for C-contiguous access. Note that this does **not** extend to multi-way (3+) contingency tables with additional margin constraints — each batch element is a separate two-way table.

## API

| Function | Description |
|---|---|
| `sample_tables` | Main entry point — sample batches of tables with arbitrary batch dimensions |
| `sample_table_sis` | Sample a single contingency table via column-wise SIS |

`sample_tables` returns `(tables, logq)` where `logq` contains the log importance weights. Pass `parallel=False` to disable numba parallelism.

## Tests

Run the Diaconis-Gangolli counting verification:

```bash
python tests/test_diaconis_gangolli.py
```

```
True count:               239,382,173
SIS estimate (n=150k):    239,413,201
Coefficient of variation: 0.9512
Effective sample size:    78,750
```

Run the Holmes-Jones example from the Chen et al. paper:

```bash
python tests/test_holmes_jones.py
```

```
Reference (Chen et al.):  3.383e16
SIS estimate (n=1M):     3.382e+16
log10 estimate:           16.5291
log10 reference:          16.5293
Coefficient of variation: 1.0537
Effective sample size:    473,875
```


## References

- Chen, Y., Diaconis, P., Holmes, S. P., & Liu, J. S. (2005). Sequential Monte Carlo methods for statistical analysis of tables. *Journal of the American Statistical Association*, 100(469), 109-120.
- Diaconis, P., & Gangolli, A. (1995). Rectangular arrays with fixed margins. In *Discrete Probability and Algorithms* (pp. 15-41). Springer.
